from __future__ import annotations

import pytest
from server.workers import conditioning_offload as mod


class FakeDevice:
    def __init__(self, name: str):
        self.name = name
        self.type = "cuda" if name.startswith("cuda") else "cpu"


class FakeTensor:
    def __init__(self, device: str = "cpu"):
        self.device = FakeDevice(device)

    def to(self, device: str):
        return FakeTensor(device)


class FakeStorage:
    def __init__(self, nbytes: int):
        self._nbytes = nbytes

    def nbytes(self):
        return self._nbytes


class FakeParam:
    _next = 1000

    def __init__(self, nbytes: int):
        self.ptr = FakeParam._next
        FakeParam._next += 1
        self.storage = FakeStorage(nbytes)

    def data_ptr(self):
        return self.ptr

    def untyped_storage(self):
        return self.storage


class FakeHandle:
    def __init__(self, collection, item):
        self.collection = collection
        self.item = item
        self.removed = False

    def remove(self):
        if not self.removed:
            self.collection.remove(self.item)
            self.removed = True


class FakeLayer:
    def __init__(self, index: int, nbytes: int = 1024, fail_h2d: bool = False):
        self.index = index
        self.device = "cpu"
        self.param = FakeParam(nbytes)
        self.pre_hooks = []
        self.post_hooks = []
        self.fail_h2d = fail_h2d

    def named_parameters(self, recurse=True):
        return [("weight", self.param)]

    def to(self, device: str):
        if device.startswith("cuda") and self.fail_h2d:
            raise RuntimeError(f"synthetic H2D failure layer={self.index}")
        self.device = device
        return self

    def register_forward_pre_hook(self, fn, with_kwargs=True):
        self.pre_hooks.append(fn)
        return FakeHandle(self.pre_hooks, fn)

    def register_forward_hook(self, fn, with_kwargs=True):
        self.post_hooks.append(fn)
        return FakeHandle(self.post_hooks, fn)

    def run(self, x, *, position_embeddings):
        args = (x,)
        kwargs = {"position_embeddings": position_embeddings}
        for hook in list(self.pre_hooks):
            args, kwargs = hook(self, args, kwargs)
        assert self.device == "cuda:1"
        assert args[0].device.type == "cuda"
        assert kwargs["position_embeddings"][0].device.type == "cuda"
        out = FakeTensor("cuda:1")
        for hook in list(self.post_hooks):
            out = hook(self, args, kwargs, out)
        return out


class FakeCuda:
    def __init__(self, free_bytes=8 * 1024**3, total_bytes=16 * 1024**3):
        self.free_bytes = free_bytes
        self.total_bytes = total_bytes
        self.sync_calls = 0
        self.empty_cache_calls = 0

    def is_available(self):
        return True

    def device_count(self):
        return 2

    def mem_get_info(self, index):
        assert index == 1
        return self.free_bytes, self.total_bytes

    def synchronize(self, index):
        assert index == 1
        self.sync_calls += 1

    def empty_cache(self):
        self.empty_cache_calls += 1


class FakeTorch:
    Tensor = FakeTensor

    def __init__(self, free_bytes=8 * 1024**3):
        self.cuda = FakeCuda(free_bytes=free_bytes)


class Obj:
    pass


def make_pipeline(*, fail_forward=False, fail_h2d_index=None, layer_bytes=1024):
    layers = [FakeLayer(i, layer_bytes, fail_h2d=(i == fail_h2d_index)) for i in range(36)]
    txt = Obj()
    txt.hf_module = Obj()
    txt.hf_module.model = Obj()
    txt.hf_module.model.language_model = Obj()
    txt.hf_module.model.language_model.layers = layers

    def forward(x, cu_seqlens=None, inputs=None, drop_idx_override=None):
        if fail_forward:
            raise ValueError("synthetic forward failure")
        pe = (FakeTensor("cpu"), FakeTensor("cpu"))
        out = x
        for layer in layers[:30]:
            out = layer.run(out, position_embeddings=pe)
        return out

    txt.forward = forward
    pipeline = Obj()
    pipeline.model = Obj()
    pipeline.model.txt_enc = txt
    return pipeline, layers


def test_config_defaults_off():
    cfg = mod.ConditioningOffloadConfig.from_environment({})
    assert cfg.mode == "off"
    assert cfg.blocks == 0


def test_config_full_is_frozen_to_cuda1():
    cfg = mod.ConditioningOffloadConfig.from_environment({"MAGE_T2I_CONDITIONING_OFFLOAD": "full"})
    assert cfg.blocks == 30
    assert cfg.target_device == "cuda:1"
    assert cfg.reserve_bytes == 512 * 1024 * 1024


def test_config_rejects_unknown_mode():
    with pytest.raises(RuntimeError, match="must be one of"):
        mod.ConditioningOffloadConfig.from_environment({"MAGE_T2I_CONDITIONING_OFFLOAD": "k31"})


def test_config_rejects_unsafe_reserve():
    with pytest.raises(RuntimeError, match="between 256 and 2048"):
        mod.ConditioningOffloadConfig.from_environment(
            {
                "MAGE_T2I_CONDITIONING_OFFLOAD": "full",
                "MAGE_T2I_CONDITIONING_OFFLOAD_RESERVE_MIB": "128",
            }
        )


def test_disabled_install_preserves_original_forward():
    pipe, _ = make_pipeline()
    original = pipe.model.txt_enc.forward
    logs = []
    ctl = mod.install_t2i_conditioning_offload(pipe, log=lambda *x: logs.append(x), torch_module=FakeTorch(), env={})
    assert pipe.model.txt_enc.forward is original
    assert ctl.enabled is False


def test_full_success_moves_back_to_cpu_and_cleans_hooks():
    pipe, layers = make_pipeline(layer_bytes=1024)
    torch = FakeTorch()
    ctl = mod.install_t2i_conditioning_offload(
        pipe,
        log=lambda *_: None,
        torch_module=torch,
        env={"MAGE_T2I_CONDITIONING_OFFLOAD": "full"},
    )
    result = pipe.model.txt_enc.forward(FakeTensor("cpu"))
    assert isinstance(result, FakeTensor)
    assert result.device.type == "cpu"
    assert all(layer.device == "cpu" for layer in layers[:30])
    assert all(not layer.pre_hooks and not layer.post_hooks for layer in layers[:30])
    assert torch.cuda.empty_cache_calls == 1
    assert ctl.last_metrics["cleanup_ok"] is True


def test_headroom_gate_fails_before_any_move():
    pipe, layers = make_pipeline(layer_bytes=50 * 1024 * 1024)
    # 30 * 50 MiB + 512 MiB reserve > 1 GiB free.
    torch = FakeTorch(free_bytes=1 * 1024**3)
    ctl = mod.install_t2i_conditioning_offload(
        pipe,
        log=lambda *_: None,
        torch_module=torch,
        env={"MAGE_T2I_CONDITIONING_OFFLOAD": "full"},
    )
    with pytest.raises(RuntimeError, match="headroom gate failed"):
        pipe.model.txt_enc.forward(FakeTensor("cpu"))
    assert all(layer.device == "cpu" for layer in layers[:30])
    assert ctl.active is False


def test_forward_exception_still_returns_all_layers_to_cpu():
    pipe, layers = make_pipeline(fail_forward=True)
    torch = FakeTorch()
    mod.install_t2i_conditioning_offload(
        pipe,
        log=lambda *_: None,
        torch_module=torch,
        env={"MAGE_T2I_CONDITIONING_OFFLOAD": "full"},
    )
    with pytest.raises(ValueError, match="synthetic forward failure"):
        pipe.model.txt_enc.forward(FakeTensor("cpu"))
    assert all(layer.device == "cpu" for layer in layers[:30])
    assert all(not layer.pre_hooks and not layer.post_hooks for layer in layers[:30])
    assert torch.cuda.empty_cache_calls == 1


def test_partial_h2d_failure_rolls_back_already_moved_layers():
    pipe, layers = make_pipeline(fail_h2d_index=7)
    torch = FakeTorch()
    mod.install_t2i_conditioning_offload(
        pipe,
        log=lambda *_: None,
        torch_module=torch,
        env={"MAGE_T2I_CONDITIONING_OFFLOAD": "full"},
    )
    with pytest.raises(RuntimeError, match="synthetic H2D failure"):
        pipe.model.txt_enc.forward(FakeTensor("cpu"))
    assert all(layer.device == "cpu" for layer in layers[:30])
    assert torch.cuda.empty_cache_calls == 1
