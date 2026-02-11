// #include <ascendc/ascendc_runtime.h>
// #include <ascendc/ascendc_op.h>
#include <cstdint>
#include <algorithm>
// #include <acl_base.h>
// #include <torch/extension.h>
// #include <torch/library.h>
#include "kernel_operator.h"
#include "types.h"
#include "kernel_operator.h"
#include "kernel_tensor_impl.h"
#include "kernel_type.h"
#include "types.h"
#include "utils.h"


__aicore__ __device__ int32_t GRID_DIM_X = 0;  // grid的X维度（num_reqs）
__aicore__ __device__ int32_t GRID_DIM_Y = 0;  // grid的Y维度（token_block_num）
// 适配昇腾NPU向量单元的块大小（可根据芯片调整，910B推荐128）
// #ifndef BLOCK_SIZE_TOKENS
#define DEFAULT_BLOCK_SIZE_TOKENS 128
// #endif

/**
 * @brief Ascend C版Eagle投机解码输入复制扩展算子
 * @并行模型：
 *   - blockIdx.x → request_idx（原Triton program_id(axis=0)）
 *   - blockIdx.y → token_batch_idx（原Triton program_id(axis=1)）
 *   - threadIdx.x → 块内token索引（原Triton tl.arange(0, BLOCK_SIZE_TOKENS)）
 */
extern "C" __global__ __aicore__ void copy_and_expand_eagle_inputs_kernel(
    // 目标模型输入（Padded）
    const int32_t* __gm__ target_token_ids_ptr,  // [total_tokens_in_batch]
    const int32_t* __gm__ target_positions_ptr,  // [total_tokens_in_batch]
    const int32_t* __gm__ next_token_ids_ptr,    // [num_reqs]
    // 草稿模型输出
    int32_t* __gm__ out_input_ids_ptr,           // [total_draft_tokens_in_batch] (output)
    int32_t* __gm__ out_positions_ptr,           // [total_draft_tokens_in_batch] (output)
    bool* __gm__ out_is_rejected_token_mask_ptr, // [total_draft_tokens_in_batch] (output)
    bool* __gm__ out_is_masked_token_mask_ptr,   // [total_draft_tokens_in_batch] (output)
    int32_t* __gm__ out_new_token_indices_ptr,   // [num_padding_slots_per_request * num_reqs] (output)
    int32_t* __gm__ out_hidden_state_mapping_ptr,// [total_tokens_in_batch]
    // 输入元数据
    const int32_t* __gm__ query_start_loc_ptr,   // [num_reqs + 1]
    const int32_t* __gm__ query_end_loc_ptr,     // [num_reqs]
    int32_t padding_token_id,                    // 填充token ID
    int32_t parallel_drafting_token_id,          // 并行草稿token ID
    // 尺寸信息
    int32_t total_input_tokens,                  // 总输入token数
    int32_t num_padding_slots_per_request,       // 每个请求的填充槽数
    bool shift_input_ids,                        // 是否偏移输入ID
    int32_t BLOCK_SIZE_TOKENS                    // 块大小（适配原Triton constexpr）
) {
    // === 1. 定位当前并行维度（对齐原Triton program_id） ===
    // 1. Block索引：CANN 8.5.0中GetBlockIdx()是全局函数，无需AscendC::前缀
    const int64_t block_idx_global = GetBlockIdx();
    // 2. 显式引用全局变量（修复未定义问题）
    const int32_t request_idx = static_cast<int32_t>(block_idx_global % ::GRID_DIM_X);  // 原blockIdx.x
    const int32_t token_batch_idx = static_cast<int32_t>(block_idx_global / ::GRID_DIM_X);  // 原blockIdx.y
    
    // 3. Thread索引：替换为CANN 8.5.0标准API GetLocalThreadId()
    const int64_t thread_idx = GetLocalThreadId();
    const int32_t tid = static_cast<int32_t>(thread_idx);
    // === 2. 边界检查：块内索引超出范围直接退出 ===
    if (tid >= BLOCK_SIZE_TOKENS) {
        return;
    }

    // === 3. 加载当前请求的token位置边界（对齐原Triton load） ===
    const int32_t query_start_loc = query_start_loc_ptr[request_idx];
    const int32_t next_query_start_loc = query_start_loc_ptr[request_idx + 1];
    const int32_t query_end_loc = query_end_loc_ptr[request_idx];

    // === 4. 计算有效token数和输入/输出偏移（完全对齐原逻辑） ===
    int32_t num_valid_tokens;
    int32_t input_offset;
    int32_t output_start;

    if (shift_input_ids) {
        num_valid_tokens = query_end_loc - query_start_loc;
        input_offset = 1;
        output_start = query_start_loc + request_idx * (num_padding_slots_per_request - 1);
    } else {
        num_valid_tokens = query_end_loc - query_start_loc + 1;
        input_offset = 0;
        output_start = query_start_loc + request_idx * num_padding_slots_per_request;
    }

    // === 5. 计算上一轮被拒绝的token数 ===
    const int32_t num_rejected = next_query_start_loc - query_end_loc - 1;

    // === 6. 计算当前请求的总输出token数 ===
    const int32_t total_output_tokens = num_valid_tokens + num_padding_slots_per_request + num_rejected;

    // === 7. 计算当前线程处理的token索引j（对齐原Triton j） ===
    const int32_t j = token_batch_idx * BLOCK_SIZE_TOKENS + tid;

    // === 8. 边界检查：j超出总输出token数直接退出 ===
    const bool in_bounds = (j < total_output_tokens);
    if (!in_bounds) {
        return;
    }

    // === 9. 划分输出区域掩码（完全对齐原Triton逻辑） ===
    const bool is_valid_region = (j < num_valid_tokens);
    const bool is_bonus_region = (j == num_valid_tokens);
    const bool is_parallel_draft_region = (j > num_valid_tokens) && 
                                         (j < num_valid_tokens + num_padding_slots_per_request);
    const bool is_rejected_region = (j >= num_valid_tokens + num_padding_slots_per_request);

    // === 10. 计算输入/输出索引 ===
    const int32_t out_idx = output_start + j;
    int32_t in_idx = query_start_loc + input_offset + j;
    // 钳位索引避免越界（对齐原tl.minimum）
    // const int32_t in_idx_clamped = std::min(in_idx, total_input_tokens - 1);
    const int32_t in_idx_clamped = (in_idx < (total_input_tokens - 1)) ? in_idx : (total_input_tokens - 1);
    // 或用：const int32_t in_idx_clamped = acl_min(in_idx, total_input_tokens - 1);（需包含acl_base.h）
    // === 11. 加载输入数据（对齐原Triton masked load） ===
    int32_t token_ids = 0;
    if (is_valid_region) { // 仅有效区域加载原始token
        token_ids = target_token_ids_ptr[in_idx_clamped];
    }

    // 加载请求的起始位置
    const int32_t start_pos = target_positions_ptr[query_start_loc];
    // 加载bonus token
    const int32_t bonus_token = next_token_ids_ptr[request_idx];

    // === 12. 构建最终token IDs（对齐原tl.where逻辑） ===
    if (is_bonus_region) {
        token_ids = bonus_token;
    } else if (is_parallel_draft_region) {
        token_ids = parallel_drafting_token_id;
    } else if (is_rejected_region) {
        token_ids = padding_token_id;
    }

    // === 13. 构建位置信息 ===
    int32_t positions = start_pos + j;
    if (is_rejected_region) { // 被拒绝区域位置设为0
        positions = 0;
    }

    // === 14. 构建输出掩码 ===
    const bool is_rejected_out = is_rejected_region;
    const bool is_masked_out = is_parallel_draft_region;

    // === 15. 计算新token索引（用于采样） ===
    const bool is_new_token_region = (j >= num_valid_tokens) && 
                                    (j < num_valid_tokens + num_padding_slots_per_request);
    if (is_new_token_region) {
        const int32_t new_token_local_idx = j - num_valid_tokens;
        const int32_t new_token_out_idx = request_idx * num_padding_slots_per_request + new_token_local_idx;
        out_new_token_indices_ptr[new_token_out_idx] = out_idx;
    }

    // === 16. 构建隐藏状态映射 ===
    if (shift_input_ids) {
        const int32_t num_input_tokens_this_request = next_query_start_loc - query_start_loc;
        const bool is_input_region = (j < num_input_tokens_this_request);
        if (is_input_region) {
            const int32_t src_idx = query_start_loc + j;
            out_hidden_state_mapping_ptr[src_idx] = out_idx;
        }
    }

    // === 17. 存储所有输出（对齐原Triton masked store） ===
    out_input_ids_ptr[out_idx] = token_ids;
    out_positions_ptr[out_idx] = positions;
    out_is_rejected_token_mask_ptr[out_idx] = is_rejected_out;
    out_is_masked_token_mask_ptr[out_idx] = is_masked_out;
}

// === 算子封装调用函数（供PyTorch调用） ===
void copy_and_expand_eagle_inputs_impl(
    // PyTorch Tensor对应的NPU全局内存指针
    const int32_t* target_token_ids_ptr,
    const int32_t* target_positions_ptr,
    const int32_t* next_token_ids_ptr,
    int32_t* out_input_ids_ptr,
    int32_t* out_positions_ptr,
    bool* out_is_rejected_token_mask_ptr,
    bool* out_is_masked_token_mask_ptr,
    int32_t* out_new_token_indices_ptr,
    int32_t* out_hidden_state_mapping_ptr,
    const int32_t* query_start_loc_ptr,
    const int32_t* query_end_loc_ptr,
    int32_t padding_token_id,
    int32_t parallel_drafting_token_id,
    int32_t total_input_tokens,
    int32_t num_padding_slots_per_request,
    bool shift_input_ids,
    int32_t BLOCK_SIZE_TOKENS,
    int32_t num_reqs,                // 请求数（grid.x维度）
    int32_t total_draft_tokens_in_batch, // 总草稿token数（计算grid.y维度）
    void* stream              // NPU计算流
) {
    // === 计算Grid维度（对齐原Triton grid） ===
    // grid.x = num_reqs（每个请求一个block）
    // grid.y = 每个请求的token块数 = ceil(total_output_tokens_per_req / BLOCK_SIZE_TOKENS)
    // const int32_t total_token_blocks = (total_draft_tokens_in_batch + BLOCK_SIZE_TOKENS - 1) / BLOCK_SIZE_TOKENS;
    // const dim3 grid_dim(num_reqs, total_token_blocks, 1);  // 2D Grid（对应原Triton program_id axis0/1）
    // const dim3 block_dim(BLOCK_SIZE_TOKENS, 1, 1);         // 1D Block（块内线程数=BLOCK_SIZE_TOKENS）
    // 1. 计算grid维度（不变）
    const int32_t token_block_num = (total_draft_tokens + BLOCK_SIZE_TOKENS - 1) / BLOCK_SIZE_TOKENS;
    const dim3 grid_dim(num_reqs, token_block_num, 1);
    const dim3 block_dim(BLOCK_SIZE_TOKENS, 1, 1);

    // 赋值全局变量（核函数需用）
    ::GRID_DIM_X = num_reqs;          // 全局作用域赋值
    ::GRID_DIM_Y = token_block_num;

    // === 启动Ascend C算子 ===
    copy_and_expand_eagle_inputs_kernel<<<grid_dim, block_dim, 0, stream>>>(
        target_token_ids_ptr,
        target_positions_ptr,
        next_token_ids_ptr,
        out_input_ids_ptr,
        out_positions_ptr,
        out_is_rejected_token_mask_ptr,
        out_is_masked_token_mask_ptr,
        out_new_token_indices_ptr,
        out_hidden_state_mapping_ptr,
        query_start_loc_ptr,
        query_end_loc_ptr,
        padding_token_id,
        parallel_drafting_token_id,
        total_input_tokens,
        num_padding_slots_per_request,
        shift_input_ids,
        BLOCK_SIZE_TOKENS
    );

    // === 检查算子启动状态 ===
    // ACL_CHECK_RET(aclrtSynchronizeStream(stream)); // 可选：同步流确保执行完成（根据场景选择）
}
