// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/mobile_transformer.h"
#include "core/mlas/inc/mlas.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
namespace {

namespace selectors {
const Node* GetLoneConsumerNode(const GraphViewer& graph_viewer, const Node& node) {
  if (!optimizer_utils::CheckOutputEdges(graph_viewer.GetGraph(), node, 1)) {
    return nullptr;
  }
  return &*node.OutputNodesBegin();
}

class ConvAddActivation : public NodeSelector {
 public:
  ConvAddActivation() = default;

  std::optional<NodesToOptimizeIndices> Select(const GraphViewer& graph_viewer, const Node& node) const override {
    const std::string_view node_ep = node.GetExecutionProviderType();
    if (node_ep != kCpuExecutionProvider) {
      return std::nullopt;
    }
    //we can't assign `node` as the pre-conv-node, even it is, because the node in this branch may not satisfy requirements
    const Node* conv_node = nullptr;
    const auto* add_node = GetLoneConsumerNode(graph_viewer, node);
    if (!add_node) {
      return std::nullopt;
    }
    // The following transforms only run when the input edge count has already
    // been decremented to zero by earlier transforms.
    // any element-wise op could be fused here.
    if (graph_utils::IsSupportedOptypeVersionAndDomain(*add_node, "Add", {7, 13, 14}) ||
        graph_utils::IsSupportedOptypeVersionAndDomain(*add_node, "Sum", {6, 8, 13})) {
      conv_node = MultiSelectBinary(*add_node, true);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*add_node, "Mul", {7, 13, 14})) {
      conv_node = MultiSelectBinary(*add_node, false);
    }
    if (!conv_node) {
      return std::nullopt;
    }
    const auto* act_node = GetLoneConsumerNode(graph_viewer, *add_node);
    //even the next node is not a activation node, it's also fine.
    if (!act_node) {
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*act_node, "Relu", {6, 13, 14}) ||
                     graph_utils::IsSupportedOptypeVersionAndDomain(*act_node, "Sigmoid", {6, 13}) ||
                     graph_utils::IsSupportedOptypeVersionAndDomain(*act_node, "Tanh", {6, 13}) ||
                     graph_utils::IsSupportedOptypeVersionAndDomain(*act_node, "LeakyRelu", {6})) {
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*act_node, "Clip", {6, 11, 12, 13})) {
      float min, max;
      if (!optimizer_utils::GetClipConstantMinMax(graph_viewer.GetGraph(), *act_node, min, max)) {
        act_node = nullptr;
      }  
    } else {
      act_node = nullptr;
    }


    NodesToOptimizeIndicesBuilder builder{};
    builder.target_node = conv_node->Index();
    builder.output_nodes = {add_node->Index()};
    if (act_node) {
      builder.output_nodes.push_back(act_node->Index());
    }
    return builder.Build();
  }

  const Node* MultiSelectBinary(const Node& node, bool add_node) const {
    InlinedVector<Node*> inputs_node;
    if (!add_node) {
      // FIX ME, not sure if it deserve to implement multiplication-relu fusion
      return nullptr;
    }
    constexpr int32_t kTensorDims = 4;
    const auto& input_defs = node.InputDefs();


    for (auto producer_node_ptr = node.InputNodesBegin(); producer_node_ptr != node.InputNodesEnd(); ++producer_node_ptr) {
      Node* producer_node = const_cast<Node*>(&(*producer_node_ptr));
      inputs_node.push_back(producer_node);
    }
    if (inputs_node.size() == 0) {
      return nullptr;
    }
    size_t input_defs_count = input_defs.size();

    // Test if all of inputs have an equal shape.
    bool all_shapes_match = true;
    auto* input_0_shape = input_defs[0]->Shape();
    for (size_t n = 1; n < input_defs_count; n++) {
      for (int i = 0; i < kTensorDims; i++) {
        // Check if ONNX shape inferencing has computed a precise dimension value.
        auto* input_n_shape = input_defs[n]->Shape();
        if ((input_0_shape == nullptr) || (input_n_shape == nullptr)) {
          all_shapes_match = false;
        } else {
          auto& input_0_dim = input_0_shape->dim(i);
          auto& input_n_dim = input_n_shape->dim(i);
          if (!utils::HasDimValue(input_0_dim) ||
              !utils::HasDimValue(input_n_dim) ||
              (input_0_dim.dim_value() <= 0) ||
              (input_0_dim.dim_value() != input_n_dim.dim_value())) {
            if (!utils::HasDimParam(input_0_dim) ||
                !utils::HasDimParam(input_n_dim) ||
                (input_0_dim.dim_param() != input_n_dim.dim_param())) {
              all_shapes_match = false;
              break;
            }
          }
        }
      }
    }
    if (all_shapes_match) {
      // If one of the inputs to the Add/Sum node is a convolution, then
      // attempt to fuse the addition into the convolution itself.
      if (add_node && input_defs_count == 2) {
        for (size_t n = 0; n < 2; n++) {
          auto& producer_input_defs = inputs_node[n]->MutableInputDefs();
          auto& producer_input_args_count = inputs_node[n]->MutableInputArgsCount();
          size_t pre_input_defs_count = producer_input_defs.size();
          // Check if this is a single use convolution that hasn't already
          // been fused with another Add/Sum node. The Add/Sum can also only be
          // fused if the convolution isn't itself fused with an activation.
          if ((inputs_node[n]->OpType() == "Conv") &&
              (pre_input_defs_count < 4) && (producer_input_args_count.size() < 4) &&
              (graph_utils::GetNodeAttribute(*inputs_node[n], "activation") == nullptr)) {
            if (pre_input_defs_count < 3) {
              // The optional bias parameter is empty so set to an empty string.
              //TODO, add a new null arguments for bias
              continue;
            }
            return inputs_node[n];
          }
        }
      }

    }
    return nullptr;
  }

};

}  // namespace selectors

namespace actions {
using NTO = NodesToOptimize;

class FuseConvAddActivation : public ReplaceWithNew {
 private:
  std::string OpType(const RuntimeState&) const override { return "FusedConv"; }

  std::string Domain(const RuntimeState&) const override { return kMSMobileDomain; }

  NodeAttributes ExtraAttributes(const RuntimeState& state) const override {
    NodeAttributes extra_fused_conv_attributes;

    const auto* activation = state.selected_nodes.Output(state.selected_nodes.num_outputs-1);
    ORT_ENFORCE(activation != nullptr, "Expected activation node.");

    const auto& activation_op_type = activation->OpType();
    utils::SetNodeAttribute(utils::MakeAttribute("activation", activation_op_type), extra_fused_conv_attributes);

    InlinedVector<float> activation_params;
    if (activation_op_type == "LeakyRelu") {
      activation_params.push_back(graph_utils::GetNodeAttribute(*activation, "alpha")->f());
    } else if (activation_op_type == "Clip") {
      float min, max;
      ORT_ENFORCE(optimizer_utils::GetClipConstantMinMax(state.graph, *activation, min, max),
                  "Failed to get Clip min/max constants.");
      activation_params.push_back(min);
      activation_params.push_back(max);
    } else if (activation_op_type == "HardSigmoid") {
      auto* alpha_attr = graph_utils::GetNodeAttribute(*activation, "alpha");
      auto* beta_attr = graph_utils::GetNodeAttribute(*activation, "beta");
      float alpha = (alpha_attr == nullptr ? 0.2f : alpha_attr->f());
      float beta = (beta_attr == nullptr ? 0.5f : beta_attr->f());
      activation_params.push_back(alpha);
      activation_params.push_back(beta);
    }

    if (!activation_params.empty()) {
      utils::SetNodeAttribute(utils::MakeAttribute("activation_params", activation_params),
                              extra_fused_conv_attributes);
    }

    return extra_fused_conv_attributes;
  }

  std::vector<NodeAndMoveInfo> ValueMoves(const RuntimeState& state) const override {
    const auto& conv = state.selected_nodes.Target();
    ORT_ENFORCE(conv.GetOutputEdgesCount() == 1 && conv.OutputNodesBegin()->OpType() == "Add",
                "Expected Conv then Add.");

    const auto add_input_idx = 1 - conv.OutputEdgesBegin()->GetDstArgIndex();

    const auto conv_location = NTO::NodeLocation{NTO::NodeType::kTarget, 0};
    const auto add_location = NTO::NodeLocation{NTO::NodeType::kOutput, 0};
    const auto relu_location = NTO::NodeLocation{NTO::NodeType::kOutput, 1};

    return {
        MoveAll(conv_location, ArgType::kInput),                                       // move all inputs from conv
        MoveAndAppend(add_location, ArgType::kInput, add_input_idx, ArgType::kInput),  // append add input
        MoveAll(relu_location, ArgType::kOutput),                                      // move all outputs from relu
    };
  }
};
}  // namespace actions

void RegisterConvAddActivationFusionRules(SelectorActionRegistry& registry) {
  const auto name = "ConvAddAct";
  auto action = std::make_unique<actions::FuseConvAddActivation>();
  auto selector = std::make_unique<selectors::ConvAddActivation>();
  registry.RegisterSelectorAndAction(name, {{"Conv", {1, 11}}},
                                     std::move(selector), std::move(action));
}


SelectorActionRegistry CreateSelectorActionRegistry() {
  SelectorActionRegistry registry{};
  RegisterConvAddActivationFusionRules(registry);
  return registry;
}

}  // namespace
ConvAddActivationMobileFusion::ConvAddActivationMobileFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers,
                                                             const SatApplyContextVariant& apply_context)
    : SelectorActionTransformer{
          "ConvAddActivationMobileFusion", CreateSelectorActionRegistry(), apply_context, compatible_execution_providers} {
}
}  // namespace onnxruntime
