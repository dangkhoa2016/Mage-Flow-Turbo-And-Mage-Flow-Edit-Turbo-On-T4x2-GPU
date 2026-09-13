import pytest
from server.runtime import DeviceInfo, RuntimeConfig, validate_required_gpu_layout


def test_required_split_device_contract():
    result = validate_required_gpu_layout([
        DeviceInfo(0, 'Tesla T4'),
        DeviceInfo(1, 'Tesla T4'),
    ])
    assert result['t2i_device'] == 'cuda:0'
    assert result['edit_device'] == 'cuda:1'


def test_requires_two_gpus():
    with pytest.raises(RuntimeError, match='at least 2 CUDA devices'):
        validate_required_gpu_layout([DeviceInfo(0, 'Tesla T4')])


def test_rejects_cpu_fallback_policy():
    with pytest.raises(RuntimeError, match='CPU fallback'):
        validate_required_gpu_layout(
            [DeviceInfo(0, 'Tesla T4'), DeviceInfo(1, 'Tesla T4')],
            RuntimeConfig(allow_cpu_fallback=True),
        )
