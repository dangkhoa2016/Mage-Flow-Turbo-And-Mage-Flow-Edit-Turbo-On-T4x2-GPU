import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

MODE_ENV = "MAGE_T2I_CONDITIONING_OFFLOAD"
MODE_OFF = "off"
MODE_FULL = "full"
SUPPORTED_MODES = {MODE_OFF, MODE_FULL}
FULL_CONDITIONING_BLOCKS = 30
TARGET_DEVICE = "cuda:1"
TARGET_INDEX = 1
DEFAULT_RESERVE_MIB = 512
RESERVE_ENV = "MAGE_T2I_CONDITIONING_OFFLOAD_RESERVE_MIB"
MIB = 1024 * 1024


@dataclass(frozen=True)
class ConditioningOffloadConfig:
    mode: str = MODE_OFF
    blocks: int = 0
    target_device: str = "cpu"
    target_index: int = -1
    reserve_bytes: int = DEFAULT_RESERVE_MIB * MIB

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "ConditioningOffloadConfig":
        values = os.environ if env is None else env
        mode = values.get(MODE_ENV, MODE_OFF).strip().lower()
        if mode not in SUPPORTED_MODES:
            raise RuntimeError(f"{MODE_ENV} must be one of {sorted(SUPPORTED_MODES)}; got {mode!r}")
        raw_reserve = values.get(RESERVE_ENV, str(DEFAULT_RESERVE_MIB)).strip()
        try:
            reserve_mib = int(raw_reserve)
        except ValueError as exc:
            raise RuntimeError(f"{RESERVE_ENV} must be an integer MiB value") from exc
        if reserve_mib < 256 or reserve_mib > 2048:
            raise RuntimeError(f"{RESERVE_ENV} must be between 256 and 2048 MiB")
        if mode == MODE_OFF:
            return cls(mode=MODE_OFF, reserve_bytes=reserve_mib * MIB)
        return cls(
            mode=MODE_FULL,
            blocks=FULL_CONDITIONING_BLOCKS,
            target_device=TARGET_DEVICE,
            target_index=TARGET_INDEX,
            reserve_bytes=reserve_mib * MIB,
        )


def block_static_bytes(layers: Any, k: int) -> int:
    if k < 0 or k > len(layers):
        raise RuntimeError(f"invalid offload block count {k}; available={len(layers)}")
    seen: set[int] = set()
    total = 0
    for i in range(k):
        for _name, param in layers[i].named_parameters(recurse=True):
            ptr = int(param.data_ptr())
            if ptr == 0 or ptr in seen:
                continue
            seen.add(ptr)
            total += int(param.untyped_storage().nbytes())
    return total


def _install_boundary_hooks(layers: Any, k: int, target_device: str, torch_module: Any) -> list[Any]:
    handles: list[Any] = []

    def make_pre():
        def pre(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
            new_args = list(args)
            if new_args and isinstance(new_args[0], torch_module.Tensor) and new_args[0].device.type == "cpu":
                new_args[0] = new_args[0].to(target_device)
            pe = kwargs.get("position_embeddings")
            if pe is not None:
                kwargs["position_embeddings"] = (pe[0].to(target_device), pe[1].to(target_device))
            return tuple(new_args), kwargs

        return pre

    def make_post():
        def post(
            _module: Any,
            _args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> Any:
            if output is not None and isinstance(output, torch_module.Tensor):
                output = output.to("cpu")
            pe = kwargs.get("position_embeddings")
            if pe is not None:
                kwargs["position_embeddings"] = (pe[0].to("cpu"), pe[1].to("cpu"))
            return output

        return post

    for i in range(k):
        handles.append(layers[i].register_forward_pre_hook(make_pre(), with_kwargs=True))
        handles.append(layers[i].register_forward_hook(make_post(), with_kwargs=True))
    return handles


class T2IConditioningOffloadController:
    """Runtime-only full Qwen3-VL conditioning offload controller.

    The accepted full diagnostic design is preserved: decoder blocks [0:30] move
    CPU -> target GPU immediately before the real text-encoder forward; boundary
    hooks move hidden_states + position_embeddings to the target GPU and each
    block's output back to CPU; all moved blocks return to CPU in a finally path
    before Mage denoise can begin.

    full-topology integration targets cuda:1, not cuda:0. GPU0 is occupied by
    the accepted GGUF safety server plus the T2I transformer/VAE and does not have
    enough free memory for the 5.78 GiB full static residency. cuda:1 retains about
    6.75 GiB free while the Edit worker is resident but idle; the coordinator must
    serialize T2I-full and Edit requests whenever full is enabled.
    """

    def __init__(
        self,
        pipeline: Any,
        *,
        config: ConditioningOffloadConfig,
        torch_module: Any,
        log: Callable[[str, str], None],
    ) -> None:
        self.pipeline = pipeline
        self.config = config
        self.torch = torch_module
        self.log = log
        self.text_encoder = pipeline.model.txt_enc
        self.layers = self.text_encoder.hf_module.model.language_model.layers
        self.original_forward = self.text_encoder.forward
        self.static_bytes = block_static_bytes(self.layers, config.blocks) if config.mode == MODE_FULL else 0
        self.installed = False
        self.active = False
        self.last_metrics: dict[str, Any] = {}

        if config.mode == MODE_FULL:
            if len(self.layers) < FULL_CONDITIONING_BLOCKS:
                raise RuntimeError(
                    f"full requires at least {FULL_CONDITIONING_BLOCKS} language-model blocks; found {len(self.layers)}"
                )
            if not self.torch.cuda.is_available():
                raise RuntimeError("full conditioning offload requires CUDA")
            if self.torch.cuda.device_count() <= config.target_index:
                raise RuntimeError(
                    f"full target {config.target_device} unavailable; "
                    f"CUDA device_count={self.torch.cuda.device_count()}"
                )

    @property
    def enabled(self) -> bool:
        return self.config.mode == MODE_FULL

    def install(self) -> None:
        if self.installed:
            return
        if not self.enabled:
            self.log("INFO", "T2I conditioning offload disabled; text encoder remains CPU-resident")
            self.installed = True
            return

        original_forward = self.original_forward

        def wrapped_forward(*args: Any, **kwargs: Any):
            return self._run_full(original_forward, *args, **kwargs)

        self.text_encoder.forward = wrapped_forward
        self.installed = True
        self.log(
            "PASS",
            "T2I_FULL_CONDITIONING_OFFLOAD_INSTALLED "
            f"blocks={self.config.blocks} target={self.config.target_device} "
            f"static_bytes={self.static_bytes} reserve_mib={self.config.reserve_bytes // MIB}",
        )

    def _headroom(self) -> tuple[int, int]:
        free_bytes, total_bytes = self.torch.cuda.mem_get_info(self.config.target_index)
        return int(free_bytes), int(total_bytes)

    def _run_full(self, original_forward: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self.active:
            raise RuntimeError("nested full conditioning offload invocation is forbidden")
        self.active = True
        moved: list[Any] = []
        handles: list[Any] = []
        started = time.perf_counter()
        h2d_seconds = 0.0
        forward_seconds = 0.0
        d2h_seconds = 0.0
        free_before, total_bytes = self._headroom()
        required = self.static_bytes + self.config.reserve_bytes
        if free_before < required:
            self.active = False
            raise RuntimeError(
                "full conditioning offload headroom gate failed: "
                f"target={self.config.target_device} free_mib={free_before / MIB:.1f} "
                f"static_mib={self.static_bytes / MIB:.1f} reserve_mib={self.config.reserve_bytes / MIB:.1f}"
            )

        result: Any = None
        forward_error: BaseException | None = None
        try:
            t0 = time.perf_counter()
            for i in range(self.config.blocks):
                layer = self.layers[i]
                layer.to(self.config.target_device)
                moved.append(layer)
            self.torch.cuda.synchronize(self.config.target_index)
            h2d_seconds = time.perf_counter() - t0

            handles = _install_boundary_hooks(
                self.layers,
                self.config.blocks,
                self.config.target_device,
                self.torch,
            )
            t1 = time.perf_counter()
            result = original_forward(*args, **kwargs)
            self.torch.cuda.synchronize(self.config.target_index)
            forward_seconds = time.perf_counter() - t1
            return result
        except BaseException as exc:
            forward_error = exc
            raise
        finally:
            for handle in handles:
                try:
                    handle.remove()
                except Exception:
                    pass

            cleanup_error: BaseException | None = None
            t2 = time.perf_counter()
            for layer in reversed(moved):
                try:
                    layer.to("cpu")
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
            try:
                self.torch.cuda.synchronize(self.config.target_index)
                self.torch.cuda.empty_cache()
                self.torch.cuda.synchronize(self.config.target_index)
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            d2h_seconds = time.perf_counter() - t2
            try:
                free_after, _ = self._headroom()
            except Exception:
                free_after = -1

            self.last_metrics = {
                "mode": self.config.mode,
                "target_device": self.config.target_device,
                "blocks": self.config.blocks,
                "static_bytes": self.static_bytes,
                "free_before_mib": round(free_before / MIB, 1),
                "free_after_mib": round(free_after / MIB, 1) if free_after >= 0 else None,
                "total_mib": round(total_bytes / MIB, 1),
                "h2d_seconds": round(h2d_seconds, 6),
                "forward_seconds": round(forward_seconds, 6),
                "d2h_seconds": round(d2h_seconds, 6),
                "wall_seconds": round(time.perf_counter() - started, 6),
                "cleanup_ok": cleanup_error is None,
            }
            self.log("INFO", f"T2I_FULL_CONDITIONING_METRICS {self.last_metrics}")
            self.active = False

            # A cleanup failure is unsafe even when the forward itself succeeded.
            # If the forward already raised, preserve that primary exception while
            # logging cleanup failure for forensic visibility.
            if cleanup_error is not None:
                self.log("FAIL", f"full cleanup error: {type(cleanup_error).__name__}: {cleanup_error}")
                if forward_error is None:
                    raise RuntimeError("full cleanup failed; worker state is unsafe") from cleanup_error

    def health_payload(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "enabled": self.enabled,
            "blocks": self.config.blocks,
            "target_device": self.config.target_device,
            "reserve_mib": self.config.reserve_bytes // MIB,
            "static_bytes": self.static_bytes,
            "active": self.active,
            "last_metrics": dict(self.last_metrics),
        }


def install_t2i_conditioning_offload(
    pipeline: Any,
    *,
    log: Callable[[str, str], None],
    torch_module: Any | None = None,
    env: Mapping[str, str] | None = None,
) -> T2IConditioningOffloadController:
    config = ConditioningOffloadConfig.from_environment(env)
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    controller = T2IConditioningOffloadController(
        pipeline,
        config=config,
        torch_module=torch_module,
        log=log,
    )
    controller.install()
    return controller
