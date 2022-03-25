// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/graph_transformer_level.h"

namespace onnxruntime {

class DataTransferManager;
struct SessionOptions;

/**
@class GraphTransformer

The interface for in-place transformation of a Graph.
*/
class GraphTransformer {
 public:
  GraphTransformer(std::string_view name)
      : name_(name), compatible_provider_types_() {
  }

  GraphTransformer(std::string_view name,
                   const InlinedHashSet<std::string_view>& compatible_execution_providers)
      : name_(name), compatible_provider_types_(compatible_execution_providers) {
  }

  GraphTransformer(std::string_view name,
                   const DataTransferManager& dt_manager,
                   const SessionOptions& session_options,
                   const InlinedHashSet<std::string_view>& compatible_execution_providers = {})
      : name_(name), dt_manager_(&dt_manager), sess_options_(&session_options), compatible_provider_types_(compatible_execution_providers) {
  }

  virtual ~GraphTransformer() = default;

  /** Gets the name of this graph transformer. */
  std::string_view Name() const noexcept {
    return name_;
  }

  const InlinedHashSet<std::string_view>& GetCompatibleExecutionProviders() const noexcept {
    return compatible_provider_types_;
  }

  /** Apply the in-place transformation defined by this transformer to the provided Graph instance.
  @param[out] modified Set to true if the Graph was modified.
  @returns Status with success or error information.
  */
  common::Status Apply(Graph& graph, bool& modified, const logging::Logger& logger) const;

  virtual bool ShouldOnlyApplyOnce() const { return false; }

 protected:
  /** Helper method to call ApplyImpl on any subgraphs in the Node. */
  common::Status Recurse(Node& node, bool& modified, int graph_level, const logging::Logger& logger) const {
    int subgraph_level = ++graph_level;
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      auto& subgraph = *entry.second;
      ORT_RETURN_IF_ERROR(ApplyImpl(subgraph, modified, subgraph_level, logger));
    }

    return Status::OK();
  }

  const DataTransferManager& GetDataTransferManager() const {
    ORT_ENFORCE(dt_manager_ != nullptr);
    return *dt_manager_;
  }

  const SessionOptions& GetSessionOptions() const {
    ORT_ENFORCE(sess_options_ != nullptr);
    return *sess_options_;
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphTransformer);

  // Apply the transform to the graph.
  // graph_level is 0 for the main graph, and is incremented when descending into the subgraph of a node.
  // You MUST call Recurse for all valid Nodes in the graph to ensure any subgraphs in control flow nodes
  // (Scan/If/Loop) are processed as well.
  // You should avoid calling Graph::Resolve in ApplyImpl unless you are 100% sure it's required. In most cases
  // the call to Graph::Resolve in Apply prior to ApplyImpl being called, and after ApplyImpl fore the main graph
  // completes (if 'modified' is true) should suffice.
  virtual common::Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger)
      const = 0;

  const std::string name_;
  const DataTransferManager* dt_manager_ = nullptr;
  const SessionOptions* sess_options_ = nullptr;
  const InlinedHashSet<std::string_view> compatible_provider_types_;
};
}  // namespace onnxruntime
