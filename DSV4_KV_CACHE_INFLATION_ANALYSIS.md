# DeepSeek V4 fp8 128k KV Cache 容量差异分析

## 背景

在 DeepSeek V4 fp8 模型、128k 上下文场景下，观测到 KV cache 实际占用显著高于理论推算值。本报告通过端到端代码追踪，定位 Ascend 实现中的膨胀来源，并量化其影响。

---

## 结论概览

| 指标 | 数值 |
|---|---|
| 纯理论 KV 数据量 | **0.705 GB / 128k request** |
| 真实所需（page 粒度） | **0.731 GB / 128k request** |
| Ascend 实际 admission | **1.659 GB / 128k request** |
| 总膨胀倍数 | **2.35×** |
| 膨胀主要来源 | num_layer_tuples = max（占 94%） |
| 次要来源 | page 粒度开销（占 4%） |

---

## 代码端到端追踪

### Step 1：模型生成 KVCacheSpec

**代码位置**：`vllm_ascend/models/deepseek_v4/model.py` `DeepseekV4Attention.__init__` L465-661

每层根据压缩比生成不同的 KVCacheSpec 对象，共 6 类：

| # | Cache 类型 | 类名 | 层数 | scheduler block_size | 物理 page_size_bytes |
|---|---|---|---|---|---|
| ① | MLA Main C4 | `AscendMLAAttentionSpec` | 20 | 512 | 77,824 B |
| ② | Indexer C4 | `AscendSFAIndexerCacheSpec` | 20 | 512 | 65,536 B |
| ③ | MLA Main C128 | `AscendMLAAttentionSpec` | 23 | 16,384 | 77,824 B |
| ④ | SWA（每层都有） | `AscendSlidingWindowMLASpec` | 43 | 128 | 77,824 B |
| ⑤ | State C4 (fp32) | `AscendCompressorStateCache` | 20 | 8 | 65,536 B |
| ⑥ | State C128 (fp32) | `AscendCompressorStateCache` | 23 | 16 | 65,536 B |

**参数来源**：
- `page_size_bytes = storage_bs × num_kv_heads × head_size × dtype_size`（[`kv_cache_interface.py` L251-256](file:///workspace/vllm_ascend/core/kv_cache_interface.py#L251)）
- `storage_block_size = block_size / compress_ratio`（同 L251）
- fp8: dtype_size = 1, State: fp32 = 4
- transformers 默认 compress_ratios = `[4]*20 + [128]*23`, sliding_window = 128

### Step 2：`group_and_unify_kv_cache_specs` 分组

**代码位置**：[`patch_kv_cache_utils.py` L251-282](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_utils.py#L251)

按 `spec.block_size` 把各层放进不同 dict，然后内部用 `UniformTypeKVCacheSpecs.from_specs()` 合并。产出 5 个 UniformType group：

| Group | block_size | 内容 |
|---|---|---|
| full_mla_c4 | 512 | 20 MLA main + 20 Indexer |
| full_mla_c128 | 16,384 | 23 MLA main |
| swa | 128 | 43 SWA |
| state_c4 | 8 | 20 State C4 |
| state_c128 | 16 | 23 State C128 |

### Step 3：`_get_kv_cache_groups_uniform_groups` page_size pad

**代码位置**：[`patch_kv_cache_utils.py` L285-380](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_utils.py#L285)

- L300-311: full_mla_group 和 full_mla_c128_group 作为前两个 group
- L320-324: `_approximate_gcd` 计算各 group 的 `num_layer_tuples` 最优值

```python
num_layer_tuples_per_group = [20, 23, 43, 20, 23]
_approximate_gcd(..., lower_bound=20) = 1   # GCD=1，几乎没对齐效果
```

- L337-354: SWA-like group 的 page_size 对齐到 full_mla_group 的 canonical bucket（`{65536, 77824}`）。本场景中所有 page_size 本身已在 canonical bucket 内，不需要实际 pad。

### Step 4：最终 kv_cache_groups 结构

传递给规划器的 5 个 `KVCacheGroupSpec`：

| Group | block_size | 总层数 | buckets |
|---|---|---|---|
| full_mla_c4 | 512 | 40 | {77824: 20, 65536: 20} |
| full_mla_c128 | 16,384 | 23 | {77824: 23} |
| swa | 128 | 43 | {77824: 43} |
| state_c4 | 8 | 20 | {65536: 20} |
| state_c128 | 16 | 23 | {65536: 23} |

### Step 5：`_get_deepseek_v4_cache_layout` — 核心膨胀点

**代码位置**：[`patch_kv_cache_utils.py` L416-456](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_utils.py#L416)

```python
# L431: canonical page sizes 来自 full_mla_group
canonical_page_sizes = sorted(_page_sizes(full_mla_spec))  # = [65536, 77824]

# L437-448: 每个 group 按 page_size 分 bucket
# L454: num_layer_tuples = max(所有 bucket 的层数) + len(mtp_layer_names)
num_layer_tuples = max(20, 20, 23, 43, 20, 23) = 43
```

**🔴 关键发现**：L454 直接取 `max`，完全忽略了 Step 3 的 `_approximate_gcd` 结果。

各 bucket 被 pad 情况：

| Group | 实际最大 bucket | 被 pad 到 | 浪费 slots |
|---|---|---|---|
| full_mla_c4 | 20 层 | 43 | 23 × 2 buckets = 46 |
| full_mla_c128 | 23 层 | 43 | 20 × 1 bucket = 20 |
| swa | 43 层 | 43 | 0 |
| state_c4 | 20 层 | 43 | 23 × 1 bucket = 23 |
| state_c128 | 23 层 | 43 | 20 × 1 bucket = 20 |

### Step 6：Pool 物理大小规划

**代码位置**：[`patch_kv_cache_utils.py` L459-523](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_utils.py#L459)

```python
bytes_per_tuple = sum(canonical_page_sizes)   # = 65536 + 77824 = 143,360 B
pool_bytes_per_block = bytes_per_tuple × num_layer_tuples  # = 143,360 × 43 = 6,164,480 B
                                                           # = 5.88 MB / scheduler block
```

shared-tuple 架构的物理 pool 结构：

```
┌──────────────────────────────────────────────────────┐
│ tuple slot 0:  [Indexer 65536 B][MLA main 77824 B]   │ ← full_mla_c4 第 0 层
│ tuple slot 1:  [Indexer 65536 B][MLA main 77824 B]   │ ← full_mla_c4 第 1 层
│ ...                                                    │
│ tuple slot 19: [Indexer 65536 B][MLA main 77824 B]   │ ← full_mla_c4 第 19 层
│ tuple slot 20: [空洞       ][空洞       ]             │ ← 无层引用
│ tuple slot 21: [空洞       ][空洞       ]             │ ← 无层引用
│ ...                                                    │
│ tuple slot 42: [空洞       ][空洞       ]             │ ← 无层引用
└──────────────────────────────────────────────────────┘
          ↑ 同一 tuple slot 被 full_mla_c128 / swa / state 各 group 复用
```

### Step 7：Admission bytes 计算

**代码位置**：[`patch_kv_cache_utils.py` L595-620](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_utils.py#L595)

```python
num_layer_tuples = max(_get_max_layers_per_page_size(g) for g in kv_cache_groups)  # = 43
layer_tuple_bytes = sum(_page_sizes(full_mla_spec))  # = 143,360 B

admission_bytes = sum(
    num_layer_tuples × group.kv_cache_spec.max_memory_usage_pages(vllm_config) × layer_tuple_bytes
    for group in kv_cache_groups
)
```

逐 group 贡献（128k request）：

| Group | group_pages | 贡献 |
|---|---|---|
| full_mla_c4 | ⌈131072/512⌉ = 256 | 43 × 256 × 143360 = **1,578 MB** |
| full_mla_c128 | ⌈131072/16384⌉ = 8 | 43 × 8 × 143360 = **47 MB** |
| swa | ⌈128/128⌉ = 1 | 43 × 1 × 143360 = **5.9 MB** |
| state_c4 | ⌈128/8⌉ = 16 | 43 × 16 × 143360 = **94 MB** |
| state_c128 | ⌈128/16⌉ = 8 | 43 × 8 × 143360 = **47 MB** |
| **合计** | | **1,659 MB = 1.659 GB** |

---

## 膨胀来源分解

| 层级 | 说明 | 膨胀倍数 | 占比 |
|---|---|---|---|
| 纯理论 KV 数据量 | 真实 token × 真实 bytes/state | **1.00×** | — |
| + page 粒度开销 | `cdiv` 对齐到物理 page 边界 | **1.04×** | 4% |
| + shared-tuple padding | `num_layer_tuples=43` 对齐 | **2.27×** | 94% |
| **Ascend 实际** | | **2.35×** | 100% |

### 各 group 具体膨胀

| Group | 真实需要 | Ascend 算给它 | 膨胀 |
|---|---|---|---|
| full_mla_c4 | 700 MB | 1,505 MB | ×2.15 |
| full_mla_c128 | 13.7 MB | 47 MB | ×3.44 |
| swa | 3.2 MB | 5.9 MB | ×1.84 |
| state_c4 | 20 MB | 94 MB | ×4.7 |
| state_c128 | 11.5 MB | 47 MB | ×4.1 |

---

## 问题定性

### 问题 1：`num_layer_tuples = max` 是核心膨胀源

L454 直接取所有 bucket 层数的最大值（43，来自 SWA group），其他 group 全部被 pad 到这个数。这些被 pad 的 tuple slots 对应的物理显存**没有任何模型层读写**，是纯空洞。

### 问题 2：`_approximate_gcd` 写了等于白写

Step 3（[L320-324](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_utils.py#L320)）调用 `_approximate_gcd([20,23,43,20,23], lower_bound=20)` 计算最优 tuple 对齐数，但结果 **1** 完全没被 Step 5（[L454](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_utils.py#L454)）引用。两处逻辑脱节。

### 问题 3：shared-tuple 是上游已抛弃的设计

从代码注释看（[L637](file:///workspace/vllm_ascend/patch/platform/patch_kv_cache_utils.py#L637)）：

> "Restore Ascend's DSV4 shared-tuple planner **removed by vLLM #51718**"

vLLM PR #51718 把 shared-tuple 换成了 per-group layout，每个 KV cache group 有独立 pool、不需要层数对齐。Ascend 刻意恢复了旧设计。

---

## Runtime 依赖分析

Ascend 恢复 shared-tuple 的原因注释说是为了 rank 重规划一致性。但从 runtime 代码看：

- **普通 DSV4 fp8 场景**：不走 `layer_tuple_strides` 分支（[`attn_utils.py` L1542-1544](file:///workspace/vllm_ascend/worker/v2/attn_utils.py#L1542) 只对 DSV41 / TurboQuant 填充）
- **DSV41 / TurboQuant**：才真正使用 shared-tuple 布局

这意味着 **普通 DSV4 fp8 场景的 runtime 不依赖 shared-tuple 布局**，理论上可以改成 per-group。

---

## 优化方向

如果把 shared-tuple 改成 per-group layout：

| | 纯理论 | per-group（优化后） | Ascend 当前 |
|---|---|---|---|
| Admission / 128k req | 0.705 GB | **0.731 GB** | 1.659 GB |
| 膨胀倍数 | 1.00× | **1.04×** | 2.35× |
| KV cache 容量增益 | — | **+127%** | 基准 |

---

## 关键源码位置索引

| 功能 | 文件 | 行号 |
|---|---|---|
| AscendMLAAttentionSpec.page_size_bytes | `vllm_ascend/core/kv_cache_interface.py` | L251-256 |
| AscendSlidingWindowMLASpec.page_size_bytes | 同上 | L410-411 |
| Compressor State Cache 构造 | `vllm_ascend/models/deepseek_v4/compressor.py` | L154-175 |
| State Cache page_size 硬编码 pad | 同上 | L58-72 |
| DSV4 block sizes 表 | `vllm_ascend/models/layer/attention/layer.py` | L33-54 |
| group_and_unify_kv_cache_specs | `patch/platform/patch_kv_cache_utils.py` | L251-282 |
| _get_kv_cache_groups_uniform_groups | 同上 | L285-380 |
| _get_deepseek_v4_cache_layout | 同上 | L416-456 |
| _get_kv_cache_config_deepseek_v4_main | 同上 | L459-523 |
| num_layer_tuples = max（核心膨胀点） | 同上 | **L454** |
| pool_bytes_per_block 除数 | 同上 | L565-566, L591-592 |
| _ascend_max_memory_usage_bytes_from_groups | 同上 | L595-620 |
| admission 核心公式 | 同上 | **L616-620** |
| _ascend_get_kv_cache_config_from_groups | 同上 | L632+ |
| Runtime layer_tuple_strides（DSV41/TQ 才用） | `vllm_ascend/worker/v2/attn_utils.py` | L1542-1544 |
| v1 model_runner 里同样的 layer_tuple_strides | `vllm_ascend/worker/v1/model_runner.py` | L5760-5797 |

---

## 附：数值模拟脚本

以下 Python 脚本可复现本次分析的全部数值，纯代码推导、无实测反推：

```python
import math
from collections import defaultdict

def cdiv(a, b): return math.ceil(a / b)
def round_up(x, m): return ((x + m - 1) // m) * m
def approximate_gcd(nums, lower_bound=None):
    if lower_bound is None: lower_bound = min(nums)
    for cand in range(lower_bound, 0, -1):
        if all(n % cand == 0 for n in nums): return cand
    return 1

MAX_MODEL_LEN = 131072   # 128k
SLIDING_WINDOW = 128

# Step 4 产出的 5 个 group（对应 Step 1-4 的推演）
final_groups = [
    {"name": "full_mla_c4",   "block_size": 512,   "layers": [(None, 77824)]*20 + [(None, 65536)]*20},
    {"name": "full_mla_c128", "block_size": 16384, "layers": [(None, 77824)]*23},
    {"name": "swa",           "block_size": 128,   "layers": [(None, 77824)]*43},
    {"name": "state_c4",      "block_size": 8,     "layers": [(None, 65536)]*20},
    {"name": "state_c128",    "block_size": 16,    "layers": [(None, 65536)]*23},
]

# Step 5
canonical_page_sizes = sorted(set(ps for _, ps in final_groups[0]["layers"]))
bucketed = []
for g in final_groups:
    b = defaultdict(int)
    for _, ps in g["layers"]: b[ps] += 1
    bucketed.append(dict(b))

num_layer_tuples = max(layers for b in bucketed for layers in b.values())
bytes_per_tuple = sum(canonical_page_sizes)
pool_bytes_per_block = bytes_per_tuple * num_layer_tuples

# Step 7 admission
def group_pages(g_name, bs):
    if g_name in ("swa", "state_c4", "state_c128"): return cdiv(SLIDING_WINDOW, bs)
    return cdiv(MAX_MODEL_LEN, bs)

ascend_admission = sum(num_layer_tuples * group_pages(g["name"], g["block_size"]) * bytes_per_tuple
                        for g in final_groups)

# 真实 admission（按各 group 实际层数 × 真实 page_size）
real_admission = 0
for g in final_groups:
    buckets = defaultdict(int)
    for _, ps in g["layers"]: buckets[ps] += 1
    pages = group_pages(g["name"], g["block_size"])
    real_admission += sum(count * pages * ps for ps, count in buckets.items())

# 纯理论值
theory = (
    20 * cdiv(MAX_MODEL_LEN, 4) * 608       # MLA Main C4
    + 20 * cdiv(MAX_MODEL_LEN, 4) * 512     # Indexer C4
    + 23 * cdiv(MAX_MODEL_LEN, 128) * 608   # MLA Main C128
    + 43 * SLIDING_WINDOW * 608              # SWA
    + 20 * cdiv(128, 4) * 2048 * 4           # State C4 (fp32)
    + 23 * cdiv(128, 128) * 1024 * 4         # State C128 (fp32)
)

print(f"纯理论:   {theory/1024**3:.3f} GB")
print(f"真实:     {real_admission/1024**3:.3f} GB  (×{real_admission/theory:.2f})")
print(f"Ascend:   {ascend_admission/1024**3:.3f} GB  (×{ascend_admission/theory:.2f})")
print(f"膨胀:     {ascend_admission/real_admission:.2f}×")
```
