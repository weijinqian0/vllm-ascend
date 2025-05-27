# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from vllm_ascend.ops.op_builder.builder import VllmAscendOpBuilder


class MoeTokenUnpermuteOpBuilder(VllmAscendOpBuilder):
    OP_NAME = "npu_moe_token_unpermute"

    def __init__(self):
        super(MoeTokenUnpermuteOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['csrc/kernels/npu_moe_token_unpermute.cpp']

    def include_paths(self):
        paths = super().include_paths()
        paths += ['csrc/kernels/inc']
        return paths

    def cxx_args(self):
        args = super().cxx_args()
        args += [
            '-Wno-sign-compare',
            '-Wno-deprecated-declarations',
            '-Wno-return-type',
            "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'"
        ]
        return args