
from typing import Optional


import torch_npu

from vllm_ascend.utils import AscendDeviceType


class CommonDeviceOperator(object):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu._npu_reshape_and_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_indices=slot_mapping)

class A5Operator(CommonDeviceOperator):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu.npu_scatter_pa_kv_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping)


DeviceOperator: Optional[CommonDeviceOperator.__class__] = None

def set_device(ascend_device_type):
    global DeviceOperator
    if ascend_device_type == AscendDeviceType.A5:
        DeviceOperator = A5Operator
    DeviceOperator = CommonDeviceOperator