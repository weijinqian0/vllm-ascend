from contextlib import contextmanager
from enum import Enum
from typing import Any, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context, set_forward_context

import vllm_ascend.envs as envs_ascend


class FusedMoEState(Enum):
    AllGather = 0
    All2All = 1
    MC2 = 2
    All2AllSeq = 3


# TODO(zzzzwwjj): add soc_version to choose branch
def get_fused_moe_state(ep_size: int, with_prefill: bool):
    if ep_size == 1:
        return FusedMoEState.AllGather
    elif envs_ascend.VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ:
        return FusedMoEState.All2AllSeq if ep_size < 16 else FusedMoEState.MC2
    # NOTE: mc2 need ep_size >= 16 & all2all can't use in torchair graph.
    elif ep_size < 16 or with_prefill:
        return FusedMoEState.All2All
    else:
        return FusedMoEState.MC2


@contextmanager
def set_ascend_forward_context(
        attn_metadata: Any,
        vllm_config: VllmConfig,
        virtual_engine: int = 0,
        num_tokens: Optional[int] = None,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        with_prefill: bool = True,
        in_profile_run: bool = False):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
    """
    with set_forward_context(attn_metadata,
                             vllm_config,
                             virtual_engine=virtual_engine,
                             num_tokens=num_tokens,
                             num_tokens_across_dp=num_tokens_across_dp):
        forward_context = get_forward_context()
        forward_context.with_prefill = with_prefill

        ep_size = torch.distributed.get_world_size(
        ) if vllm_config.parallel_config.enable_expert_parallel else 1

        fused_moe_state = get_fused_moe_state(ep_size, with_prefill)

        forward_context.fused_moe_state = fused_moe_state

        forward_context.in_profile_run = in_profile_run

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing
        forward_context.capturing = False

        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and forward_context.dp_metadata is not None:
            forward_context.max_tokens_across_dp = forward_context.dp_metadata.max_tokens_across_dp_cpu.item(
            )
        elif num_tokens is not None:
            forward_context.max_tokens_across_dp = num_tokens
        elif attn_metadata is not None:
            if hasattr(attn_metadata, 'num_actual_tokens'):
                forward_context.max_tokens_across_dp = attn_metadata.num_actual_tokens
            else:
                forward_context.max_tokens_across_dp = attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens
        else:
            forward_context.max_tokens_across_dp = None

        try:
            yield
        finally:
            pass
