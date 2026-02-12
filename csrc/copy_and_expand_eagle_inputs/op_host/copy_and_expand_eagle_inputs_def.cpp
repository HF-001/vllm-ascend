/**
 * @file copy_and_expand_eagle_inputs_def.cpp
 * @brief CopyAndExpandEagleInputs OpDef, InferShape, InferDataType
 */

#include "register/op_def_registry.h"

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    return GRAPH_SUCCESS;
}

}  // namespace ge


namespace optiling {
// Forward declaration of TilingFunc (defined in tiling cpp)
static ge::graphStatus TilingFunc(gert::TilingContext* context);
}  // namespace optiling


namespace ops {

class CopyAndExpandEagleInputs : public OpDef {
public:
    explicit CopyAndExpandEagleInputs(const char* name) : OpDef(name)
    {
        // -------------------- Inputs --------------------
        this->Input("target_token_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("target_positions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("next_token_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("query_start_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("query_end_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- Outputs --------------------
        this->Output("out_input_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_positions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_is_rejected_token_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_is_masked_token_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_new_token_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_hidden_state_mapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- Attributes --------------------
        this->Attr("padding_token_id").Int();
        this->Attr("parallel_drafting_token_id").Int();
        this->Attr("num_padding_slots_per_request").Int();
        this->Attr("shift_input_ids").Bool();
        this->Attr("total_input_tokens").Int();

        // -------------------- InferShape / InferDataType --------------------
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        // -------------------- Tiling --------------------
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(CopyAndExpandEagleInputs);

}  // namespace ops
