/**
 * @file copy_and_expand_eagle_inputs_tiling.cpp
 * @brief CopyAndExpandEagleInputs TilingFunc implementation
 */

#include "copy_and_expand_eagle_inputs_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdio>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // ========== 1. Get hardware core count ==========
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNum();

    // ========== 2. Derive num_reqs from query_start_loc shape ==========
    // query_start_loc is the 4th input (index 3), shape [num_reqs + 1]
    auto queryStartLocShape = context->GetInputShape(3);
    uint32_t numReqs = 0;
    if (queryStartLocShape != nullptr &&
        queryStartLocShape->GetStorageShape().GetDimNum() > 0) {
        int64_t dim0 = queryStartLocShape->GetStorageShape().GetDim(0);
        numReqs = (dim0 > 1) ? static_cast<uint32_t>(dim0 - 1) : 0;
    }

    // ========== 3. Get operator attributes ==========
    auto attrs = context->GetAttrs();

    int32_t paddingTokenId = *(attrs->GetAttrPointer<int32_t>(0));
    int32_t parallelDraftingTokenId = *(attrs->GetAttrPointer<int32_t>(1));
    int32_t numPaddingSlotsPerReq = *(attrs->GetAttrPointer<int32_t>(2));
    bool shiftInputIds = *(attrs->GetAttrPointer<bool>(3));
    int32_t totalInputTokens = *(attrs->GetAttrPointer<int32_t>(4));

    // ========== 4. Compute core distribution ==========
    uint32_t usedCoreNum = std::min(coreNum, numReqs);
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    uint32_t reqsPerCore   = numReqs / usedCoreNum;
    uint32_t remainderReqs = numReqs % usedCoreNum;

    // ========== 5. Set tiling_key ==========
    context->SetTilingKey(1);

    // ========== 6. Get output shape ==========
    uint32_t totalDraftTokens = 0;
    auto outShape = context->GetOutputShape(0);
    if (outShape != nullptr &&
        outShape->GetStorageShape().GetDimNum() > 0) {
        totalDraftTokens = static_cast<uint32_t>(outShape->GetStorageShape().GetDim(0));
    }

    // ========== 7. Fill TilingData ==========
    CopyAndExpandEagleInputsTilingData tiling;
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_numReqs(numReqs);
    tiling.set_reqsPerCore(reqsPerCore);
    tiling.set_remainderReqs(remainderReqs);
    tiling.set_paddingTokenId(paddingTokenId);
    tiling.set_parallelDraftingTokenId(parallelDraftingTokenId);
    tiling.set_numPaddingSlotsPerReq(static_cast<uint32_t>(numPaddingSlotsPerReq));
    tiling.set_totalInputTokens(static_cast<uint32_t>(totalInputTokens));
    tiling.set_shiftInputIds(shiftInputIds ? 1u : 0u);
    tiling.set_totalDraftTokens(totalDraftTokens);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // ========== 8. Set block_dim ==========
    context->SetBlockDim(usedCoreNum);

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
