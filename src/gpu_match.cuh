#pragma once

#include "graph.h"
#include "pattern.h"
#include "callstack.h"
#include "job_queue.h"
#include "neugn_bridge.h"

namespace STMatch {
  void launch_parallel_match(Graph* dev_graph, Pattern* dev_pattern,
                             CallStack* dev_callstack, JobQueue* job_queue, size_t* res,
                             int* idle_warps, int* idle_warps_count, int* global_lock,
                             int* found, unsigned long long* fms,
                             NeuGNRequest* neugn_requests = nullptr, int* neugn_request_count = nullptr,
                             int neugn_request_capacity = 0, int* active_warps = nullptr);
}
