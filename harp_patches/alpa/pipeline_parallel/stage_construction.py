"""
Core implementations for stage construction algorithms.
The algorithm groups layers into pipeline stages.
"""
from dataclasses import dataclass
import logging
import tqdm
from typing import Sequence, List, Tuple, Dict, Union, Optional
from collections import defaultdict

from jax._src.lib import xla_extension as xe
from jax.core import Var
import numpy as np
import ray
from ray.util.actor_pool import ActorPool
import time

from alpa.device_mesh import VirtualPhysicalMesh
from alpa.global_env import global_config
from alpa.pipeline_parallel.computation import (
    JaxPipelineComputation, merge_marked_jaxprs_with_named_call)
from alpa.pipeline_parallel.stage_profiling import (get_compute_cost,
                                                    last_compute_cost_file_name)
from alpa.shard_parallel.auto_sharding import AutoShardingOption
from alpa.timer import timers
from alpa.util import OrderedSet, maybe_numba_jit, jaxpr_to_hlo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class AutoStageOption:
    """Options of auto stage construction algorithm."""
    # The search space of the physical submesh shapes.
    # Possible choices: {"power_of_two", "small_power_of_two", "all"}.
    submesh_physical_shape_space: str = "power_of_two"
    # The search space of the logical mesh shapes.
    # Possible choices: {"same_as_physical", "data_parallel_only",
    #                    "single_node_model_parallel", "all", "manual"}.
    # If "manual", the user needs to specify the logical mesh shape.
    manually_specified_submeshes: Sequence[Tuple[int, int]] = None
    # The search space for the logical mesh shapes.
    # Possible choices: {"all", "single_node_model_parallel",
    #                    "same_as_physical", "data_parallel_only",
    #                    "model_parallel_only"}.
    submesh_logical_shape_space: str = "single_node_model_parallel"
    # Profile only individual layers or composition different layers.
    # Possible choices: {"individual", "composition"}.
    layer_profile_mode: str = "composition"
    # The tolerance of imbalance in the auto-stage construction.
    stage_imbalance_tolerance: float = np.inf
    # Use HLO cost model for computational cost or profile for the cost.
    use_hlo_cost_model: bool = False
    # The filename of profiling result database.
    profiling_database_filename: Optional[str] = None
    # The file name of the cached compute cost.
    cached_profile_result: Optional[str] = None # Dispatched

    # heterogeneous cluster support for Harp
    gpu_flops: Optional[Sequence[float]] = None
    cluster_key: Optional[Union[str, Sequence[str]]] = None
    cached_compute_communication_cost_load_path: Optional[str] = None
    cached_compute_communication_cost_store_path_prefix: Optional[str] = None


@dataclass
class ManualStageOption:
    """Options of manual stage assignment."""
    # Layer IDs of each forward stage.
    forward_stage_layer_ids: Sequence[Sequence[int]]
    # The physical shapes of submeshes of each stage.
    submesh_physical_shapes: Sequence[Sequence[int]]
    # The logical shapes of submeshes of each stage.
    submesh_logical_shapes: Sequence[Sequence[int]]
    # The auto-sharding options of each stage.
    submesh_autosharding_option_dicts: Sequence[dict]

    # Added by H1F1B
    maunal_schedule_strategy: Sequence[int] = None


@dataclass
class UniformStageOption:
    # The number of stages.
    num_stages: int = None
    # The physical shape of all submeshes.
    submesh_physical_shape: Sequence[int] = None
    # The logical shape of all submeshes.
    submesh_logical_shape: Sequence[int] = None
    # The auto-sharding option of all stages.
    submesh_autosharding_option: dict = None


StageOption = Union[AutoStageOption, ManualStageOption, UniformStageOption]

# Get results for debugging
last_forward_stage_layer_ids = None
last_submesh_shapes = None
last_logical_mesh_shapes = None
last_autosharding_option_dicts = None


def get_last_dp_result():
    """Gets the DP result of the last run."""
    return (last_compute_cost_file_name, last_forward_stage_layer_ids,
            last_submesh_shapes, last_logical_mesh_shapes,
            last_autosharding_option_dicts)


@maybe_numba_jit
def get_optimal_submeshes(best_s, f_argmin, num_devices, num_layers,
                          submesh_n_devices):
    current_s = best_s
    current_layer = 0
    current_devices = num_devices

    res = []
    while current_s > 0 and current_layer < num_layers and current_devices > 0:
        next_start_layer, submesh_choice, autosharding_choice = (
            f_argmin[current_s, current_layer, current_devices])
        assert next_start_layer != -1 and current_devices != -1
        res.append(((current_layer, next_start_layer), submesh_choice,
                    autosharding_choice))
        current_s -= 1
        current_layer = next_start_layer
        current_devices -= submesh_n_devices[submesh_choice]
    assert (current_s == 0 and current_layer == num_layers and
            current_devices == 0)

    return res


@maybe_numba_jit
def training_dp_impl_2(num_layers, num_devices, submesh_sizes,
                       valid_idxs_and_costs, max_n_succ_stages):
    f = np.full((num_layers + 1, num_layers + 1, num_devices + 1),
                np.inf,
                dtype=np.float32)
    f_stage_max = np.full((num_layers + 1, num_layers + 1, num_devices + 1),
                          0.0,
                          dtype=np.float32)
    f_argmin = np.full((num_layers + 1, num_layers + 1, num_devices + 1, 3),
                       -1,
                       dtype=np.int32)
    f[0, num_layers, 0] = 0
    for d in range(1, num_devices + 1):
        for l, i, submesh_id, n_config, stage_cost in valid_idxs_and_costs:
            l, i, submesh_id, n_config = map(int, (l, i, submesh_id, n_config))
            n_submesh_devices = submesh_sizes[submesh_id]
            if n_submesh_devices <= d:
                for s in range(1, num_layers + 1):
                    if s - 1 > max_n_succ_stages[l, i, submesh_id, n_config]:
                        continue

                    new_cost = f[s - 1, i + 1,
                                 d - n_submesh_devices] + stage_cost
                    if new_cost < f[s, l, d]:
                        f[s, l, d] = new_cost
                        f_argmin[s, l, d] = (i + 1, submesh_id, n_config)
                        f_stage_max[s, l, d] = max(
                            f_stage_max[s - 1, i + 1, d - n_submesh_devices],
                            stage_cost)

    return f, f_stage_max, f_argmin


def training_dp_2(
    num_devices,
    num_microbatches,
    submesh_choices,
    compute_cost,
    max_n_succ_stages,
):
    """Faster implementation of the training DP algorihtm."""
    # TODO(zhuohan): Further verify the correctness of this implementation.
    timers("stage-construction-dp").start()

    num_layers = len(compute_cost)
    all_possible_stage_costs = np.sort(np.unique(compute_cost))
    best_cost = np.inf
    best_solution = None
    last_max_stage_cost = 0.0
    # FIXME(zhuohan): Set this gap as a tunable parameter in global config
    gap = 1e-6
    assert len(
        all_possible_stage_costs), "no solution in auto stage construction."

    submesh_sizes = np.array([n * m for (n, m) in submesh_choices],
                             dtype=np.int64)

    for max_stage_cost in all_possible_stage_costs:
        if max_stage_cost - last_max_stage_cost < gap:
            continue
        if max_stage_cost * num_microbatches >= best_cost:
            break

        # Lifts check for stage_cost <= t_max_stage_cost out of the inner dp
        # loop.
        valid_cost_idxs = np.transpose(
            (compute_cost <= max_stage_cost).nonzero())
        # This corresponds to the i of k <= i <= K from eqn. 3 in the alpa
        # paper.
        valid_cost_idxs = valid_cost_idxs[
            valid_cost_idxs[:, 0] <= valid_cost_idxs[:, 1]]
        if len(valid_cost_idxs) == 0:
            continue
        valid_costs = compute_cost[tuple(valid_cost_idxs.T)]
        valid_idxs_and_costs = np.hstack(
            [valid_cost_idxs, valid_costs[:, np.newaxis]])
        # Sort by descending layer idx because DP initializes
        # F[0, num_layers, 0] = 0
        valid_idxs_and_costs = valid_idxs_and_costs[np.flip(
            valid_cost_idxs[:, 1].argsort())]

        # Don't perform backtracking each time (do it only for the best
        # solution).
        f, f_stage_max, f_argmin = training_dp_impl_2(
            num_layers,
            num_devices,
            submesh_sizes,
            valid_idxs_and_costs,
            max_n_succ_stages,
        )

        best_s = f[:, 0, num_devices].argmin()
        best_total_cost = f[best_s, 0, num_devices]
        if np.isinf(best_total_cost):
            continue
        stage_cost = (num_microbatches - 1) * f_stage_max[best_s, 0,
                                                          num_devices]

        if best_total_cost + stage_cost < best_cost:
            best_cost = best_total_cost + stage_cost
            best_solution = best_s, f_argmin
        last_max_stage_cost = max_stage_cost

    assert best_solution is not None, (
        "Unable to find any solution to inter-op dp.")
    best_s, f_argmin = best_solution
    best_solution = get_optimal_submeshes(best_s, f_argmin, num_devices,
                                          num_layers, submesh_sizes)

    timers("stage-construction-dp").stop()
    return best_cost, best_solution


@maybe_numba_jit
def training_dp_impl(num_layers, num_devices, num_microbatches, submesh_choices,
                     num_autosharding_configs, compute_cost, max_n_succ_stages,
                     max_stage_cost):
    """The core implementation of the DP algorithm."""
    # For f, layer ID start from 0
    # f[#pipeline stages,
    #   layer id that is currently being considered,
    #   number of devices used]
    f = np.full((num_layers + 1, num_layers + 1, num_devices + 1),
                np.inf,
                dtype=np.float32)
    f_stage_max = np.full((num_layers + 1, num_layers + 1, num_devices + 1),
                          0.0,
                          dtype=np.float32)
    f_argmin = np.full((num_layers + 1, num_layers + 1, num_devices + 1, 3),
                       -1,
                       dtype=np.int32)
    f[0, num_layers, 0] = 0
    for s in range(1, num_layers + 1):  # pylint: disable=too-many-nested-blocks
        for i in range(num_layers - 1, -1, -1):
            for j in range(1, num_devices + 1):
                for k in range(num_layers, i, -1):
                    for m, submesh in enumerate(submesh_choices):
                        n_submesh_devices = np.prod(np.array(submesh))
                        if n_submesh_devices <= j:
                            # TODO(zhuohan): This level of for loop is not
                            #   necessary. It can be optimized by sorting
                            #   the logical mesh shapes.
                            for n_config in range(num_autosharding_configs):
                                if s - 1 <= max_n_succ_stages[i, k - 1, m,
                                                              n_config]:
                                    stage_cost = compute_cost[i, k - 1, m,
                                                              n_config]
                                    new_cost = f[s - 1, k, j -
                                                 n_submesh_devices] + stage_cost
                                    if (stage_cost <= max_stage_cost and
                                            new_cost < f[s, i, j]):
                                        f[s, i, j] = new_cost
                                        f_stage_max[s, i, j] = max(
                                            f_stage_max[s - 1, k,
                                                        j - n_submesh_devices],
                                            stage_cost)
                                        f_argmin[s, i, j] = (k, m, n_config)

    best_s = -1
    best_total_cost = np.inf
    for s in range(1, num_layers + 1):
        if f[s, 0, num_devices] < best_total_cost:
            best_s = s
            best_total_cost = f[s, 0, num_devices]

    if np.isinf(best_total_cost):
        return np.inf, None

    total_cost = f[best_s, 0, num_devices] + (
        num_microbatches - 1) * f_stage_max[best_s, 0, num_devices]
    current_s = best_s
    current_layer = 0
    current_devices = num_devices

    res = []
    while current_s > 0 and current_layer < num_layers and current_devices > 0:
        next_start_layer, submesh_choice, autosharding_choice = (
            f_argmin[current_s, current_layer, current_devices])
        assert next_start_layer != -1 and current_devices != -1
        res.append(((current_layer, next_start_layer), submesh_choice,
                    autosharding_choice))
        current_s -= 1
        current_layer = next_start_layer
        current_devices -= np.prod(np.array(submesh_choices[submesh_choice]))
    assert (current_s == 0 and current_layer == num_layers and
            current_devices == 0)

    return total_cost, res


def training_dp(num_layers, num_devices, num_microbatches, submesh_choices,
                num_autosharding_configs, compute_cost, max_n_succ_stages):
    """Auto stage dynamic programming."""
    timers("stage-construction-dp").start()

    all_possible_stage_costs = np.sort(np.unique(compute_cost))
    best_cost = np.inf
    best_solution = None
    last_max_stage_cost = 0.0
    # FIXME(zhuohan): Set this gap as a tunable parameter in global config
    gap = 1e-6
    assert len(
        all_possible_stage_costs), "no solution in auto stage construction."
    for max_stage_cost in all_possible_stage_costs:
        if max_stage_cost * num_microbatches >= best_cost:
            break
        if max_stage_cost - last_max_stage_cost < gap:
            continue
        cost, solution = training_dp_impl(num_layers, num_devices,
                                          num_microbatches, submesh_choices,
                                          num_autosharding_configs,
                                          compute_cost, max_n_succ_stages,
                                          max_stage_cost)
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
        last_max_stage_cost = max_stage_cost

    timers("stage-construction-dp").stop()
    return best_cost, best_solution


@maybe_numba_jit
def inference_dp_impl(num_layers, num_devices, submesh_choices,
                      num_autosharding_configs, compute_cost):
    """The core implementation of the DP algorithm."""
    # For f, layer ID start from 0
    # f[#pipeline stages,
    #   layer id that is currently being considered,
    #   number of devices used]
    f = np.full((num_layers + 1, num_layers + 1, num_devices + 1),
                np.inf,
                dtype=np.float32)
    f_argmin = np.full((num_layers + 1, num_layers + 1, num_devices + 1, 3),
                       -1,
                       dtype=np.int32)
    f[0, 0, 0] = 0
    for s in range(1, num_layers + 1):  # pylint: disable=too-many-nested-blocks
        for i in range(1, num_layers + 1):
            for j in range(1, num_devices + 1):
                for k in range(0, i):
                    for m, submesh in enumerate(submesh_choices):
                        n_submesh_devices = np.prod(np.array(submesh))
                        if n_submesh_devices <= j:
                            for n_config in range(num_autosharding_configs):
                                stage_cost = compute_cost[k, i - 1, m, n_config]
                                new_cost = max(
                                    f[s - 1, k, j - n_submesh_devices],
                                    stage_cost)
                                if new_cost < f[s, i, j]:
                                    f[s, i, j] = new_cost
                                    f_argmin[s, i, j] = (k, m, n_config)

    best_s = -1
    best_total_cost = np.inf
    for s in range(1, num_layers + 1):
        if f[s, num_layers, num_devices] * s < best_total_cost:
            best_s = s
            best_total_cost = f[s, num_layers, num_devices] * s

    if np.isinf(best_total_cost):
        return np.inf, None

    current_s = best_s
    current_layer = num_layers
    current_devices = num_devices

    res = []
    while current_s > 0 and current_layer > 0 and current_devices > 0:
        next_end_layer, submesh_choice, autosharding_choice = (
            f_argmin[current_s, current_layer, current_devices])
        assert next_end_layer != -1
        res.append(((next_end_layer, current_layer), submesh_choice,
                    autosharding_choice))
        current_s -= 1
        current_layer = next_end_layer
        current_devices -= np.prod(np.array(submesh_choices[submesh_choice]))
    assert (current_s == 0 and current_layer == 0 and current_devices == 0)

    return best_total_cost, res


def inference_dp(num_layers, num_devices, submesh_choices,
                 num_autosharding_configs, compute_cost):
    """Auto stage dynamic programming."""
    timers("stage-construction-dp").start()
    cost, solution = inference_dp_impl(num_layers, num_devices, submesh_choices,
                                       num_autosharding_configs, compute_cost)
    solution = list(reversed(solution))
    timers("stage-construction-dp").stop()
    return cost, solution



@maybe_numba_jit
def get_optimal_submeshes_hetero(best_s, best_last_cluster, f_argmin, 
                                 num_layers, submesh_sizes_s, bounds, strides):
    """Backtracking to reconstruct solution for N Hetero Clusters."""
    current_s = best_s
    current_layer = 0
    current_cluster = best_last_cluster
    
    # Calculate Full Device State
    current_device_state = 0
    num_clusters = len(bounds)
    for i in range(num_clusters):
         max_used = bounds[i] - 1
         current_device_state += max_used * strides[i]
         
    res = []

    while current_s > 0 and current_layer < num_layers:
        record = f_argmin[current_s, current_layer, current_device_state, current_cluster]
        
        next_start_layer = record[0]
        prev_cluster = record[1]
        submesh_choice = record[2]
        autosharding_choice = record[3]
        
        if next_start_layer == -1: break
            
        res.append(((current_layer, next_start_layer), 
                    current_cluster, submesh_choice, autosharding_choice))
        
        current_s -= 1
        current_layer = next_start_layer
        
        used_devices = submesh_sizes_s[current_cluster][submesh_choice]
        current_device_state -= used_devices * strides[current_cluster]
        
        current_cluster = prev_cluster

    return res



@maybe_numba_jit
def training_dp_hetero_impl(
    num_layers, num_clusters, strides, bounds,
    submesh_sizes_flat, submesh_offsets,
    valid_idxs_flat, valid_idxs_offsets, valid_idxs_counts,
    max_n_succ_stages_s,
    num_device_per_host_s,
    communication_cost, t_max):
    
    # Calculate total states
    total_device_states = 1
    for b in bounds:
        total_device_states *= b
    
    # DP Tables: f[s, layer, device_state, last_cluster]
    f = np.full((num_layers + 1, num_layers + 1, total_device_states, num_clusters), np.inf, dtype=np.float32)
    f_stage_max = np.full((num_layers + 1, num_layers + 1, total_device_states, num_clusters), 0.0, dtype=np.float32)
    f_stage_comm_max = np.full((num_layers + 1, num_layers + 1, total_device_states, num_clusters), 0.0, dtype=np.float32)
    
    # f_argmin: next_l, prev_cluster, submesh_id, config_id
    f_argmin = np.full((num_layers + 1, num_layers + 1, total_device_states, num_clusters, 4), -1, dtype=np.int32)
    
    # Base case: 0 stages, End of layers, 0 devices used, Last cluster irrelevant
    f[0, num_layers, 0, :] = 0.0
    
    # Iterate Stage Counts (1 to num_layers)
    for s in range(1, num_layers + 1):
        # Iterate Device States
        for state_idx in range(total_device_states):
            # Iterate Target Cluster 'c' (where we place the NEW stage)
            for c in range(num_clusters):
                
                # Check constraints and find best previous state
                start = valid_idxs_offsets[c]
                cnt = valid_idxs_counts[c]
                
                for k in range(cnt):
                    base = start + k * 5
                    l_start = int(valid_idxs_flat[base])
                    i_end = int(valid_idxs_flat[base + 1])
                    submesh_id = int(valid_idxs_flat[base + 2])
                    n_config = int(valid_idxs_flat[base + 3])
                    stage_cost = valid_idxs_flat[base + 4]
                    
                    next_l = i_end + 1
                    
                    if stage_cost > t_max: continue
                    
                    # Max Successors Constraint
                    max_limit = max_n_succ_stages_s[c][l_start, i_end, submesh_id, n_config]
                    if 2 * (s - 1) > max_limit - 1:
                        continue
                        
                    # Device Check
                    subsize = submesh_sizes_flat[submesh_offsets[c] + submesh_id]
                    current_usage_c = (state_idx // strides[c]) % bounds[c]
                    
                    # Alloc check
                    dev_per_host = num_device_per_host_s[c]
                    valid_alloc = False
                    if subsize <= current_usage_c:
                         rem = current_usage_c % dev_per_host
                         if subsize <= rem or (rem == 0 and subsize > 1):
                             valid_alloc = True
                         elif subsize == dev_per_host:
                             valid_alloc = True
                    
                    if not valid_alloc: continue
                    
                    # Calculate Prev State
                    prev_state_idx = state_idx - subsize * strides[c]
                    
                    # Constraint: Segmented Usage (Cannot re-enter a cluster once left)
                    # If we switch from prev_c (!= c) to c, 'c' must be empty in prev_state.
                    # 'current_usage_c' is usage in 'state_idx'.
                    # 'subsize' is what we just added.
                    # So usage in 'prev_state' is (current_usage_c - subsize).
                    prev_usage_c = current_usage_c - subsize

                    # Comm Cost
                    intra = communication_cost[c, i_end]
                    inter = communication_cost[-1, i_end]
                    
                    best_prev_val = np.inf
                    best_prev_c = -1
                    best_stg_mx = 0.0
                    best_com_mx = 0.0
                    
                    for prev_c in range(num_clusters):
                        # Enforce Monotonic Order (prev_c >= c)
                        # We want Cluster(Head) <= Cluster(Tail) => c <= prev_c
                        # So pruning if prev_c < c
                        if prev_c < c:
                            continue

                        prev_cost = f[s-1, next_l, prev_state_idx, prev_c]
                        if np.isinf(prev_cost): continue
                        
                        cc = intra if prev_c == c else inter
                        if cc > t_max: continue
                        
                        total = prev_cost + stage_cost + cc
                        
                        if total < best_prev_val:
                            best_prev_val = total
                            best_prev_c = prev_c
                            best_stg_mx = max(f_stage_max[s-1, next_l, prev_state_idx, prev_c], stage_cost)
                            best_com_mx = max(f_stage_comm_max[s-1, next_l, prev_state_idx, prev_c], stage_cost, cc)
                            
                    if best_prev_c != -1:
                        # Update DP
                        if best_prev_val < f[s, l_start, state_idx, c]:
                            f[s, l_start, state_idx, c] = best_prev_val
                            f_argmin[s, l_start, state_idx, c] = (next_l, best_prev_c, submesh_id, n_config)
                            f_stage_max[s, l_start, state_idx, c] = best_stg_mx
                            f_stage_comm_max[s, l_start, state_idx, c] = best_com_mx

    # *** Optimization: Prune Invalid States for the next stage 's+1' ***
    # If using Monotonic Order (prev_c >= c), it means Cluster indices are NON-DECREASING 
    # as we go from First Stage (Head) to Last Stage (Tail).
    # 0 -> 1 -> 2.
    # However, 'state_idx' encodes usage. 
    # The pure 'prev_c > c' check inside the loop only prunes edges. 
    # It does not prevent iterating over all 'state_idx'.
    
    return f, f_stage_max, f_stage_comm_max, f_argmin

def run_dp_batch_func(init_args, max_stage_costs):
    """Standalone DP batch runner for both Ray and local execution."""
    (num_layers, num_clusters, strides, bounds,
     submesh_sizes_flat, submesh_offsets,
     compute_cost_s, max_n_succ_stages_s,
     num_device_per_host_s, communication_cost,
     num_microbatches, num_devices_s) = init_args
     
    best_cost = np.inf
    best_solution = None

    for max_stage_cost in max_stage_costs:
        valid_idxs_flat = []
        valid_idxs_offsets = [0] * num_clusters
        valid_idxs_counts = [0] * num_clusters
        curr_off = 0
        
        has_sol = True
        for c in range(num_clusters):
            valid = np.transpose((compute_cost_s[c] <= max_stage_cost).nonzero())
            valid = valid[valid[:, 0] <= valid[:, 1]]
            
            if len(valid) == 0:
                has_sol = False
                break
                
            costs = compute_cost_s[c][tuple(valid.T)]
            block = np.hstack([valid, costs[:, np.newaxis]])
            block = block[np.flip(valid[:, 1].argsort())]
            
            valid_idxs_offsets[c] = curr_off
            valid_idxs_counts[c] = len(block)
            valid_idxs_flat.append(block.flatten())
            curr_off += len(block) * 5
            
        if not has_sol:
            continue
        
        valid_idxs_flat_arr = np.concatenate(valid_idxs_flat)
        
        f, _, f_stage_comm_max, f_argmin = training_dp_hetero_impl(
            num_layers, num_clusters, strides, bounds,
            submesh_sizes_flat, submesh_offsets,
            valid_idxs_flat_arr, 
            np.array(valid_idxs_offsets, dtype=np.int64),
            np.array(valid_idxs_counts, dtype=np.int64),
            max_n_succ_stages_s,
            np.array(num_device_per_host_s, dtype=np.int64),
            communication_cost,
            max_stage_cost
        )
        
        full_idx = 0
        for c in range(num_clusters):
            full_idx += num_devices_s[c] * strides[c]
            
        candidates = f[:, 0, full_idx, :]
        min_v = np.min(candidates)
        
        if np.isinf(min_v):
            continue
        
        flat_idx = np.argmin(candidates)
        bs, bc = np.unravel_index(flat_idx, candidates.shape)
        
        stage_ov = (num_microbatches - 1) * f_stage_comm_max[bs, 0, full_idx, bc]
        total = min_v + stage_ov
        
        if total < best_cost:
            best_cost = total
            best_solution = (bs, bc, f_argmin)
    
    return best_cost, best_solution


@ray.remote(num_cpus=1)
class DPWorker:
    def __init__(self, init_args):
        self.init_args = init_args

    def run_dp_batch(self, max_stage_costs):
        return run_dp_batch_func(self.init_args, max_stage_costs)


class DPWorkerPool:
    def __init__(self, init_args, num_cpus):
        # We start num_cpus workers
        self.workers = [DPWorker.remote(init_args) for _ in range(num_cpus)]
        self.pool = ActorPool(self.workers)

    def map_unordered(self, fn, items):
        return self.pool.map_unordered(fn, items)
    
    def submit_and_get_first(self, max_stage_cost):
        return ray.get(self.workers[0].run_dp_batch.remote([max_stage_cost]))


def training_dp_hetero(num_layers, num_microbatches, num_devices_s, submesh_choices_s,
                       compute_cost_s, max_n_succ_stages_s,
                       num_device_per_host_s, communication_cost=None, 
                       debug_pass=0, use_ray=True):
    """Auto stage dynamic programming (Generalized for N Clusters)."""
    timers("stage-construction-dp").start()
    num_clusters = len(num_devices_s)
    
    # 1. Costs
    all_costs = []
    for c_costs in compute_cost_s:
        all_costs.append(np.unique(c_costs))
    all_possible_stage_costs = np.sort(np.concatenate(all_costs))
    
    # Filter costs that are too close
    unique_costs = [all_possible_stage_costs[0]]
    gap = 1e-6
    for c in all_possible_stage_costs[1:]:
        if c - unique_costs[-1] > gap:
            unique_costs.append(c)
    all_possible_stage_costs = np.array(unique_costs)

    # 2. Strides & Bounds
    bounds = np.array(num_devices_s, dtype=np.int64) + 1
    strides = np.ones(num_clusters, dtype=np.int64)
    for i in range(1, num_clusters):
        strides[i] = strides[i-1] * bounds[i-1]
        
    # 3. Submesh Flattening
    submesh_sizes_flat = []
    submesh_offsets = [0] * num_clusters
    submesh_sizes_s_list = []
    curr = 0
    for i in range(num_clusters):
        submesh_offsets[i] = curr
        sizes = np.array([np.prod(x) for x in submesh_choices_s[i]], dtype=np.int64)
        submesh_sizes_s_list.append(sizes)
        submesh_sizes_flat.extend(sizes)
        curr += len(sizes)
    
    submesh_sizes_flat = np.array(submesh_sizes_flat, dtype=np.int64)
    submesh_offsets = np.array(submesh_offsets, dtype=np.int64)
    
    if communication_cost is None:
        communication_cost = np.zeros((num_clusters + 1, num_layers - 1), dtype=np.float32)

    init_args = (
        num_layers, num_clusters, strides, bounds,
        submesh_sizes_flat, submesh_offsets,
        compute_cost_s, max_n_succ_stages_s,
        num_device_per_host_s, communication_cost,
        num_microbatches, num_devices_s
    )

    best_feasible_cost = np.inf
    best_feasible_sol = None
    feasible_idx = -1
    pool = None

    if use_ray:
        # Estimate num_cpus based on available resources or default to something reasonable
        num_cpus = int(ray.available_resources().get("CPU", 1))
        num_cpus = max(1, num_cpus)
        
        print(f"Initializing DPWorkerPool with {num_cpus} workers...")
        pool = DPWorkerPool(init_args, num_cpus)

        # === Optimization: Binary Search for smallest feasible t_s ===
        print("Starting Binary Search for minimal feasible Max Stage Cost...")
        
        left, right = 0, len(all_possible_stage_costs) - 1
        
        # Check feasibility function
        def check_feasible(idx):
            t_max = all_possible_stage_costs[idx]
            return pool.submit_and_get_first(t_max)

        while left <= right:
            mid = (left + right) // 2
            cost, sol = check_feasible(mid)
            
            if np.isinf(cost):
                # Infeasible, need larger max_stage_cost
                left = mid + 1
            else:
                # Feasible, try smaller
                feasible_idx = mid
                best_feasible_cost = cost
                best_feasible_sol = sol
                right = mid - 1
    else:
        # Local Sequential Search (No Ray)
         print("Using Local Sequential Search (use_ray=False)...")
         left, right = 0, len(all_possible_stage_costs) - 1
         while left <= right:
            mid = (left + right) // 2
            t_max = all_possible_stage_costs[mid]
            cost, sol = run_dp_batch_func(init_args, [t_max])
            
            if np.isinf(cost):
                left = mid + 1
            else:
                feasible_idx = mid
                best_feasible_cost = cost
                best_feasible_sol = sol
                right = mid - 1
            
    if feasible_idx == -1:
         raise RuntimeError("No feasible solution found even with largest stage cost.")
         
    print(f"Found feasible solution at index {feasible_idx}. Cost: {best_feasible_cost:.2f}")

    # === Optimization: Pruning by Upper Bound t_E ===
    # t_E = ceil(T(t_s) / B)
    # Any t_max > t_E is pruned.
    
    cutoff_val = best_feasible_cost / num_microbatches
    # We find the index where t_max > cutoff_val
    cutoff_idx = np.searchsorted(all_possible_stage_costs, cutoff_val, side='right')
    
    # The range to search is [feasible_idx + 1, cutoff_idx)
    search_candidates = all_possible_stage_costs[feasible_idx + 1 : cutoff_idx]
    
    print("\n" + "="*40)
    print(f"Bidirectional Pruning Results:")
    print(f"  - Feasible t_min found at index {feasible_idx}, cost={best_feasible_cost:.2f}")
    print(f"  - Upper Bound t_max (cutoff) = {cutoff_val:.6f}")
    if len(search_candidates) > 0:
        print(f"  - Efficient Search Range: [{search_candidates[0]:.6f}, {search_candidates[-1]:.6f}]")
        print(f"  - Candidates to check: {len(search_candidates)} (Original: {len(all_possible_stage_costs)})")
    else:
        print(f"  - Efficient Search Range: Empty (Optimal is best_feasible)")
    print("="*40 + "\n")
    
    best_cost = best_feasible_cost
    best_solution = best_feasible_sol
    
    if len(search_candidates) > 0:
        if use_ray:
            # Batching strategy:
            num_cpus = len(pool.workers)
            batch_size = max(1, len(search_candidates) // (num_cpus * 4)) # 4 batches per worker
            candidate_batches = [search_candidates[i:i + batch_size] for i in range(0, len(search_candidates), batch_size)]
            
            pbar = tqdm.tqdm(total=len(candidate_batches), desc="DP Search (Parallel)")
            
            for cost, sol in pool.map_unordered(lambda a, v: a.run_dp_batch.remote(v), candidate_batches):
                if cost < best_cost:
                    best_cost = cost
                    best_solution = sol
                pbar.update(1)
            pbar.close()
        else:
             # Local sequential
             pbar = tqdm.tqdm(search_candidates, desc="DP Search (Local)")
             # Batch somewhat larger for local to reduce function call overhead? Or just 1 by 1.
             # Just 1 by 1 for simplicity or small batches.
             local_batch_size = 10
             for i in range(0, len(search_candidates), local_batch_size):
                 batch = search_candidates[i : i+local_batch_size]
                 cost, sol = run_dp_batch_func(init_args, batch)
                 if cost < best_cost:
                    best_cost = cost
                    best_solution = sol
                 pbar.update(len(batch))
             pbar.close()

    assert best_solution is not None, "No solution found"
    
    bs, bc, f_argmin = best_solution
    stages = get_optimal_submeshes_hetero(
        bs, bc, f_argmin,
        num_layers, submesh_sizes_s_list, bounds, strides
    )
    
    print(f"+++ Best solution found. Cost: {best_cost:.2f}")
    
    res_compute_cost = np.zeros(len(stages)) 
    res_comm_cost = np.zeros(len(stages))
    
    if use_ray and pool:
        for actor in pool.workers:
            ray.kill(actor)
    
    timers("stage-construction-dp").stop()
    
    return best_cost, stages, res_compute_cost, res_comm_cost



def get_submesh_choices(
        num_hosts: int,
        num_devices_per_host: int,
        space: str,
        manually_specified_submeshes: Optional[Sequence[Tuple[int,
                                                              int]]] = None):
    """Gets the valid choices of submesh shapes."""
    if global_config.overwrite_submesh_choices is not None:
        return global_config.overwrite_submesh_choices
    submesh_choices = []

    # smaller submeshes:
    i = 1
    while i <= num_devices_per_host:
        submesh_choices.append((1, i))
        i *= 2
    assert submesh_choices[-1][1] == num_devices_per_host, (
        "Only supports the cases where num_devices_per_host is power of two, "
        f"while now num_devices_per_host = {num_devices_per_host}")
    
    if global_config.nccl_mode == "xla_extension":
        assert space == "single_node_power_of_two", (
            "For xla_extension nccl mode, only single_node_power_of_two "
            "is supported.")

    # larger meshes:
    if space == "all":
        for i in range(2, num_hosts + 1):
            submesh_choices.append((i, num_devices_per_host))
    elif space == "power_of_two":
        i = 2
        while i <= num_hosts:
            submesh_choices.append((i, num_devices_per_host))
            i *= 2
    elif space == "small_power_of_two":
        i = 2
        while i <= min(num_hosts, 4):
            submesh_choices.append((i, num_devices_per_host))
            i *= 2
    elif space == "single_node_power_of_two":  # xla_extension 
        pass
    elif space == "manual":
        submesh_choices = manually_specified_submeshes
    else:
        raise ValueError(f"Invalid submesh space: {space}")

    return tuple(submesh_choices)


def get_one_submesh_autosharding_config_choices(
        virtual_submesh: VirtualPhysicalMesh, space: str, batch_size: int, cluster_id: Optional[int] = None):
    """
    Return a list of logical meshes and autosharding configs.
    Which will be used by the auto stage construction algorithm.

    Args:
        virtual_submesh: a submesh.
        space: The search space of the logical mesh shapes.
            possible choices: {"same_as_physical", "data_parallel_only",
                               "single_node_model_parallel", "all"}.
        batch_size: the batch size used.
    """
    results = []
    num_devices = virtual_submesh.num_devices
    if space in ["all", "single_node_model_parallel", "single_node_model_parallel_and_safememory_dp"]:
        min_mp_dimension = 1
        if space == "all":
            max_mp_dimension = num_devices
        else:
            max_mp_dimension = virtual_submesh.num_devices_per_host
            if (space == "single_node_model_parallel_and_safememory_dp" and 
                virtual_submesh.num_devices_per_host > 1):
                assert global_config.enable_Hetero, f"Only support heterogeneous cluster for {space}."
                
                min_mp_dimension = min(2 ** global_config.profile_memory_safe_level[cluster_id],
                                       virtual_submesh.num_devices_per_host)
        # print(f"set min_mp_dimension={min_mp_dimension}, max_mp_dimension={max_mp_dimension} for {space} with num_devices={num_devices}, num_devices_per_host={virtual_submesh.num_devices_per_host}"

        for mp_size in range(min_mp_dimension, max_mp_dimension + 1):
            if num_devices % mp_size == 0:
                dp_size = num_devices // mp_size
                if batch_size % dp_size == 0:
                    results.append((virtual_submesh.get_logical_mesh(
                        (dp_size, mp_size)), {
                            "force_batch_dim_to_mesh_dim": 0
                        }))
        if space != "single_node_model_parallel_and_safememory_dp":
            results.append((virtual_submesh.get_logical_mesh((num_devices, 1)), {}))
            
    elif space == "same_as_physical":
        results.append((virtual_submesh.get_logical_mesh(), {}))
    elif space == "data_parallel_only":
        results.append((virtual_submesh.get_logical_mesh((num_devices, 1)), {
            "force_batch_dim_to_mesh_dim": 0
        }))
    elif space == "model_parallel_only":
        results.append((virtual_submesh.get_logical_mesh((1, num_devices)), {
            "force_batch_dim_to_mesh_dim": 0
        }))
    else:
        raise ValueError(f"Invalid space for get_one_submesh_autosharding"
                         f"_config_choices: {space}")
    return results


def get_all_submesh_autosharding_config_choices(virtual_mesh, submesh_choices,
                                                space, batch_size, cluster_id=None):
    """Get all possible auto sharding config choices for all possible submesh
    shapes."""
    # A config is: Tuple(logical_mesh_shape, autosharding_option_dict).
    # Enumerate all (2D Mesh with force batch dim) + one (1D Mesh with mix batch
    # dim).
    autosharding_configs = []
    for submesh in submesh_choices:
        num_hosts, num_devices_per_host = submesh
        virtual_submesh = virtual_mesh.slice_2d(
            tuple(range(num_hosts)),
            (tuple(range(num_devices_per_host)),) * num_hosts)
        submesh_autosharding_configs = (
            get_one_submesh_autosharding_config_choices(virtual_submesh, space,
                                                        batch_size, cluster_id))
        autosharding_configs.append(submesh_autosharding_configs)

    # Pad all submesh to the maximum number of configs
    max_num_autosharding_configs = max(
        len(configs) for configs in autosharding_configs)
    for configs in autosharding_configs:
        configs += [None] * (max_num_autosharding_configs - len(configs))

    return autosharding_configs


def get_sliced_virtual_submeshes(virtual_mesh, submesh_shapes):
    """Slice the origin mesh into submeshes given submesh shapes."""
    num_hosts = virtual_mesh.num_hosts
    num_devices_per_host = virtual_mesh.num_devices_per_host
    submesh_sizes = [np.prod(submesh) for submesh in submesh_shapes]
    virtual_submeshes = [None] * len(submesh_shapes)
    assert sum(submesh_sizes) == virtual_mesh.num_devices
    sorted_submesh_indices = np.argsort(submesh_sizes, kind="stable")
    current_host_id = 0
    current_device_id = 0
    for i in reversed(sorted_submesh_indices):
        required_num_hosts, required_num_devices = submesh_shapes[i]
        if required_num_devices == num_devices_per_host:
            assert current_device_id == 0
            assert current_host_id + required_num_hosts <= num_hosts, (
                "Do not have enough hosts for the solution.")
            virtual_submeshes[i] = virtual_mesh.slice_2d(
                tuple(
                    range(current_host_id,
                          current_host_id + required_num_hosts)),
                (tuple(range(num_devices_per_host)),) * required_num_hosts)
            current_host_id += required_num_hosts
        else:
            assert required_num_hosts == 1
            assert required_num_devices < num_devices_per_host
            assert (current_device_id + required_num_devices <=
                    num_devices_per_host), (
                        "Do not have enough devices in a host for the solution")
            virtual_submeshes[i] = virtual_mesh.slice_2d([current_host_id], [
                tuple(
                    range(current_device_id,
                          current_device_id + required_num_devices))
            ])
            current_device_id += required_num_devices
            if current_device_id == num_devices_per_host:
                current_host_id += 1
                current_device_id = 0
    assert current_host_id == num_hosts
    assert current_device_id == 0
    return virtual_submeshes



def get_sliced_virtual_submeshes_hetero(virtual_mesh, submesh_shapes):
    """Slice hetero virtual meshes into submeshes.

    submesh_shapes: Sequence of (cluster_id, (num_hosts, num_devices_per_host))
    """
    num_clusters = len(virtual_mesh)
    if num_clusters < 2:
        raise ValueError("Hetero slicing requires at least 2 clusters.")

    virtual_submeshes = [None] * len(submesh_shapes)

    cluster_to_indices = [[] for _ in range(num_clusters)]
    for idx, submesh in enumerate(submesh_shapes):
        cluster_id, _ = submesh
        if cluster_id < 0 or cluster_id >= num_clusters:
            raise ValueError(f"Invalid sub_cluster_id: {cluster_id}")
        cluster_to_indices[cluster_id].append(idx)

    for cluster_id, indices in enumerate(cluster_to_indices):
        submesh_sizes = [np.prod(submesh_shapes[i][1]) for i in indices]
        total_size = sum(submesh_sizes)
        assert total_size == virtual_mesh[cluster_id].num_devices, (
            "Do not have enough hosts for the solution "
            f"{submesh_shapes}.")

    for cluster_id, indices in enumerate(cluster_to_indices):
        if not indices:
            continue

        num_hosts = virtual_mesh[cluster_id].num_hosts
        num_devices_per_host = virtual_mesh[cluster_id].num_devices_per_host
        submesh_sizes = [np.prod(submesh_shapes[i][1]) for i in indices]
        sorted_submesh_indices = np.argsort(submesh_sizes, kind="stable")
        current_host_id = 0
        current_device_id = 0

        for k in reversed(sorted_submesh_indices):
            global_idx = indices[k]
            required_num_hosts, required_num_devices = submesh_shapes[global_idx][
                1]

            if required_num_devices == num_devices_per_host:
                assert current_device_id == 0
                assert current_host_id + required_num_hosts <= num_hosts, (
                    "Do not have enough hosts for the solution.")
                virtual_submeshes[global_idx] = virtual_mesh[cluster_id].slice_2d(
                    tuple(
                        range(current_host_id,
                              current_host_id + required_num_hosts)),
                    (tuple(range(num_devices_per_host)),) * required_num_hosts)
                current_host_id += required_num_hosts
            else:
                assert required_num_hosts == 1
                assert required_num_devices < num_devices_per_host
                assert (current_device_id + required_num_devices <=
                        num_devices_per_host), (
                            "Do not have enough devices in a host for the "
                            "solution")
                virtual_submeshes[global_idx] = virtual_mesh[cluster_id].slice_2d(
                    [current_host_id], [
                        tuple(
                            range(current_device_id,
                                  current_device_id + required_num_devices))
                    ])
                current_device_id += required_num_devices
                if current_device_id == num_devices_per_host:
                    current_host_id += 1
                    current_device_id = 0

        assert current_host_id == num_hosts
        assert current_device_id == 0

    return virtual_submeshes
    

def print_autosharding_configs(vm, autosharding_configs):
    print(f"\n\033[1;33m==> VirtualMesh Info\033[0m")
    print(f"    \033[1;36mShape:\033[0m   {vm.shape}")
    print(f"    \033[1;36mDevices:\033[0m {vm.device_strs}")
    print(f"    \033[1;36mAutosharding Configs:\033[0m")

    for submesh_id, tmp_as_configs in enumerate(autosharding_configs):
        valid_configs = [c for c in tmp_as_configs if c is not None]
        if not valid_configs:
            continue
            
        print(f"      \033[1;35m[Submesh {submesh_id}]\033[0m ({len(valid_configs)} configurations)")
        
        for i, config in enumerate(valid_configs):
            logical_mesh, options = config
            print(f"        \033[1;32m- Config {i}:\033[0m "
                  f"Local Shape={logical_mesh.shape}")
                #   f"Alpha={logical_mesh.mesh_alpha}, "
                #   f"Beta={logical_mesh.mesh_beta}")
            if options:
                print(f"          \033[90mOptions: {options}\033[0m")
    print("")


def get_heter_cluster_flops_ratio(virtual_mesh, stage_option):
    heter_cluster_flops = []
    heter_cluster_flops_ratio = []
    
    # 计算每个 cluster 的 flops
    for i, vmesh in enumerate(virtual_mesh):
        heter_cluster_flops.append(vmesh.num_devices * stage_option.gpu_flops[i])
    for i, vmesh in enumerate(virtual_mesh):
        heter_cluster_flops_ratio.append(heter_cluster_flops[i] / sum(heter_cluster_flops))
    return heter_cluster_flops_ratio


def get_communication_cost(layers: Sequence[JaxPipelineComputation]):
    """
    Calculate communication cost for all possible cuts (stage boundaries) 
    across all defined network configurations.
    """
    
    def _get_cut_vars(layers: Sequence[JaxPipelineComputation]):
        """Identify variables that must cross the boundary after each layer."""
        assert len(layers) % 2 == 0, "get_communication_cost() Only support training mode."
        num_forward_layers = len(layers) // 2
        forward_layers = layers[:num_forward_layers]

        # Map each variable to its producer layer index
        var_producer = {}
        for idx, layer in enumerate(forward_layers):
            for var in layer.outvars:
                var_producer[var] = idx

        # Map each variable to the maximum layer index that consumers it
        var_max_consumer = {}
        for idx, layer in enumerate(forward_layers):
            for var in layer.invars:
                # Only track variables produced by previous layers (activations)
                if var in var_producer:
                    var_max_consumer[var] = max(var_max_consumer.get(var, -1), idx)

        # For each cut 's' (boundary after forward layer 's'), identify crossing variables.
        # Valid cuts are 0 to num_forward_layers - 2 (since no cut after the last layer).
        cut_vars = defaultdict(list)
        for s in range(num_forward_layers - 1):
            for var, prod_idx in var_producer.items():
                # A variable crosses boundary 's' if it is produced at or before 's'
                # and consumed strictly after 's'.
                if prod_idx <= s:
                    max_cons = var_max_consumer.get(var, -1)
                    if max_cons > s:
                        cut_vars[s].append(var)
        return cut_vars

    cut_vars = _get_cut_vars(layers)
    
    # Consolidate all network configurations
    network_configs = []
    if hasattr(global_config, "intra_cluster_network_performance") and global_config.intra_cluster_network_performance:
        network_configs.extend(global_config.intra_cluster_network_performance)
    
    if hasattr(global_config, "inter_cluster_network_performance") and global_config.inter_cluster_network_performance:
        network_configs.extend(global_config.inter_cluster_network_performance)

    # Fallback to legacy mesh_performance if new configs are missing
    if not network_configs and hasattr(global_config, "mesh_performance"):
        # Legacy support: iterate values of the dict
        network_configs = list(global_config.mesh_performance.values())

    num_networks = len(network_configs)
    num_layers = len(layers) // 2
    
    # rows: network configurations, cols: cut boundaries (layer index)
    total_cost = np.zeros((num_networks, num_layers))
    
    GB = 1 << 30

    for net_idx, config in enumerate(network_configs):
        if config is None:
            # Indicates disjoint/invalid link; set infinite cost
            total_cost[net_idx, :] = np.inf
            continue

        bandwidth = config["bandwidth"]
        latency = config["latency"]
        
        for s in range(num_layers - 1):
            vars_in_cut = cut_vars.get(s, [])
            if not vars_in_cut:
                cost = 0.0
            else:
                total_bytes = sum(v.aval.size * v.aval.dtype.itemsize for v in vars_in_cut)
                # Cost model: Transfer Time + Latency
                # (Assuming infinite bandwidth for gradients or symmetric cost included in DP)
                cost = (total_bytes / GB / bandwidth) + latency
            
            total_cost[net_idx, s] = cost

    return total_cost


import os
import pickle
from datetime import datetime

def load_cached_compute_communication_cost(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cached compute and communication cost file not found: {path}")
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    print("\033[93mWarning, user should make sure the cached compute and communication cost is consistent with the current setting, including virtual mesh, submesh choices, autosharding configs, etc. \033[0m")
    compute_cost = data.get("compute_cost")
    max_n_succ_stages = data.get("max_n_succ_stages")
    
    if compute_cost is None or max_n_succ_stages is None:
        raise ValueError(f"Invalid cache file format: {path}")
    
    print(f"Cached compute and communication cost loaded from: {path}")
    return compute_cost, max_n_succ_stages


def store_cached_compute_communication_cost(
    compute_cost, max_n_succ_stages, virtual_mesh,
    num_micro_batches, cluster_id, store_path_prefix
):

    os.makedirs(store_path_prefix, exist_ok=True)

    timenow = datetime.now().strftime("%m%d-%H%M%S")
    device_str = "_".join(virtual_mesh.device_strs)
    store_path = os.path.join(store_path_prefix, f"{timenow}_cluster-{cluster_id}_vm({device_str})_mb{num_micro_batches}.pkl")
    
    data = {
        "compute_cost": compute_cost,
        "max_n_succ_stages": max_n_succ_stages,
    }
    
    with open(store_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Cached compute and communication cost stored at: {store_path}")


def cluster_layers_and_slice_mesh(
        layers: Sequence[JaxPipelineComputation],
        virtual_mesh: VirtualPhysicalMesh, accumulator_mapping: Dict[Var, Var],
        acc_grad_invars: Sequence[Var], acc_grad_outvars: Sequence[Var],
        num_micro_batches: int, batch_size: int,
        jax_apply_layers: Sequence[JaxPipelineComputation],
        apply_grad_global_info: Tuple, pipeline_schedule: str,
        default_as_option: AutoShardingOption, stage_option: StageOption):
    """
    Stage-mesh assignment.

    This function clusters pipeline layers into stages, slice the device
    mesh into multiple submeshes, and assign the stages to the submeshes.
    We first profile the compute cost of layers on different choices
    of submeshes and find the optimal solution with DP.

    Args:
        layers: All the layers.
        virtual_mesh: The virtual device mesh.
        accumulator_mapping: The donation_mapping for the layers.
        acc_grad_invars: invars of the gradient accumulation layers.
        acc_grad_outvars: outvars of the gradient accumulation layers.
        num_micro_batches: The number of microbatches.
        batch_size: The micro batch size.
        jax_apply_layers: The apply gradient computations corresponding
          to each forward layers.
        pipeline_schedule: The pipeline schedule.
        default_as_option: The default auto-sharding option.
        stage_option: The options controling how to construct stages.
    """
    timers("stage-construction").start()

    assert pipeline_schedule != "inference", f"Harp does not support"
    if global_config.enable_Hetero:
        for vm in virtual_mesh:
            assert vm.launched_physical_mesh_group is None, "Harp does not support given mesh."

    # Assume each forward layer corresponds to a backward layer
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2

    if isinstance(stage_option, AutoStageOption):
            
        if global_config.enable_Hetero:
            submesh_choices = []
            autosharding_configs = []

            for i, vm in enumerate(virtual_mesh):
                submesh_choice = get_submesh_choices(vm.num_hosts, 
                                                            vm.num_devices_per_host,
                                                            stage_option.submesh_physical_shape_space,
                                                            stage_option.manually_specified_submeshes)
                

                autosharding_config = get_all_submesh_autosharding_config_choices(vm, submesh_choice, stage_option.submesh_logical_shape_space, 
                                                                                 batch_size, cluster_id=i)
                submesh_choices.append(submesh_choice)
                autosharding_configs.append(autosharding_config)

                print_autosharding_configs(vm, autosharding_config)

            num_autosharding_configs = [len(autosharding_configs[i][0]) for i in range(len(virtual_mesh))]


            compute_cost = []
            max_n_succ_stages = []
            
            # 根据 stage_option 计算 heter_cluster_flops_ratio
            heter_cluster_flops_ratio = get_heter_cluster_flops_ratio(virtual_mesh, stage_option)


            # ======

            # # get communication cost Added by lssyes
            # communication_cost = get_communication_cost(layers)

            
            for i, vm in enumerate(virtual_mesh):
                # 感觉这里可以加个多线程啊? 不太可以
                if stage_option.cached_compute_communication_cost_load_path is not None:
                    assert stage_option.cached_compute_communication_cost_load_path[i] is not None, \
                        "cached_compute_communication_cost_load_path should be provided for all clusters in hetero setting."
                    compute_cost_i, max_n_succ_stages_i = load_cached_compute_communication_cost(
                        stage_option.cached_compute_communication_cost_load_path[i])
                else:
                    compute_cost_i, max_n_succ_stages_i = get_compute_cost(
                        vm, submesh_choices[i], autosharding_configs[i], layers,
                        accumulator_mapping, acc_grad_invars, acc_grad_outvars,
                        jax_apply_layers, apply_grad_global_info, num_micro_batches,
                        default_as_option, stage_option, i,
                        heter_cluster_flops_ratio[i])
                    if stage_option.cached_compute_communication_cost_store_path_prefix is not None:
                        store_cached_compute_communication_cost(
                            compute_cost_i, max_n_succ_stages_i,
                            vm, num_micro_batches, i, 
                            stage_option.cached_compute_communication_cost_store_path_prefix
                        )

                compute_cost.append(compute_cost_i)
                max_n_succ_stages.append(max_n_succ_stages_i)
                
            compute_cost = tuple(np.array(x) for x in compute_cost)
            max_n_succ_stages = tuple(np.array(x) for x in max_n_succ_stages)
            

            # communication_cost = get_communication_cost(layers)
            communication_cost = None
            (_, solution,
             solution_compute_cost,
             solution_communication_cost) = training_dp_hetero(num_layers, num_micro_batches,
                                                               [vm.num_devices for vm in virtual_mesh],
                                                               submesh_choices,
                                                               compute_cost, max_n_succ_stages,
                                                               [vm.num_devices_per_host for vm in virtual_mesh],
                                                               communication_cost,
                                                               use_ray=False)
            assert solution is not None, "no solution in auto stage construction."
            
            # Parse solution
            forward_stage_layer_ids = [
                list(range(start_id, end_id))
                for (start_id, end_id), _, _, _ in solution
            ]
            submesh_shapes = [
                    (sub_cluster_id, submesh_choices[sub_cluster_id][submesh_id])
                    for _, sub_cluster_id, submesh_id, _ in solution
            ]
            selected_autosharding_configs = [
                autosharding_configs[sub_cluster_id][submesh_id][autosharding_config_id]            ## 思索 autosharding_configs
                for _, sub_cluster_id, submesh_id, autosharding_config_id in solution
            ]

            schedule_args = (solution_compute_cost, solution_communication_cost)

            
            

        else:
            submesh_choices = get_submesh_choices(
                virtual_mesh.num_hosts, virtual_mesh.num_devices_per_host,
                stage_option.submesh_physical_shape_space,
                stage_option.manually_specified_submeshes)
            autosharding_configs = get_all_submesh_autosharding_config_choices(
                virtual_mesh, submesh_choices,
                stage_option.submesh_logical_shape_space, batch_size)
            num_autosharding_configs = len(autosharding_configs[0])

            # Use DP to find the optimal solution.
            compute_cost, max_n_succ_stages = get_compute_cost(
                virtual_mesh, submesh_choices, autosharding_configs, layers,
                accumulator_mapping, acc_grad_invars, acc_grad_outvars,
                jax_apply_layers, apply_grad_global_info, num_micro_batches,
                default_as_option, stage_option)
            

            _, solution = training_dp(num_layers, virtual_mesh.num_devices,
                                        num_micro_batches, submesh_choices,
                                        num_autosharding_configs, compute_cost,
                                        max_n_succ_stages)

            assert solution is not None, "no solution in auto stage construction."

            forward_stage_layer_ids = [
                list(range(start_id, end_id))
                for (start_id, end_id), _, _ in solution
            ]
            submesh_shapes = [
                submesh_choices[submesh_id] for _, submesh_id, _ in solution
            ]
            selected_autosharding_configs = [
                autosharding_configs[submesh_id][autosharding_config_id]
                for _, submesh_id, autosharding_config_id in solution
            ]

        logical_mesh_shapes = [
            mesh.shape for mesh, _ in selected_autosharding_configs
        ]
        autosharding_option_dicts = [
            option_dict for _, option_dict in selected_autosharding_configs
        ]

        # Print and store the results
        _print_dp_results(forward_stage_layer_ids, submesh_shapes, logical_mesh_shapes, autosharding_option_dicts, stage_option.cluster_key)
        global last_forward_stage_layer_ids, last_submesh_shapes
        global last_logical_mesh_shapes, last_autosharding_option_dicts
        last_forward_stage_layer_ids = forward_stage_layer_ids
        last_submesh_shapes = submesh_shapes
        last_logical_mesh_shapes = logical_mesh_shapes
        last_autosharding_option_dicts = autosharding_option_dicts


    elif isinstance(stage_option, ManualStageOption):
        # Check forward_stage_layer_ids is a partition of range(num_layers)
        forward_stage_layer_ids = stage_option.forward_stage_layer_ids
        last_layer_id = 0
        for stage_layer_ids in forward_stage_layer_ids:
            for layer_id in stage_layer_ids:
                assert layer_id == last_layer_id
                last_layer_id += 1
        assert last_layer_id == num_layers, (
            f"{last_layer_id} layers in stage option, but {num_layers} marked")
        submesh_shapes = stage_option.submesh_physical_shapes
        logical_mesh_shapes = (stage_option.submesh_logical_shapes or
                               submesh_shapes)
        autosharding_option_dicts = (
            stage_option.submesh_autosharding_option_dicts)
    elif isinstance(stage_option, UniformStageOption):
        raise NotImplementedError("Harp does not support uniform stage option yet.")
    else:
        raise ValueError(f"Invalid pipeline stage option: {stage_option}")

    get_sliced_virtual_impl = get_sliced_virtual_submeshes_hetero if global_config.enable_Hetero else get_sliced_virtual_submeshes
    sliced_meshes = get_sliced_virtual_impl(virtual_mesh, submesh_shapes)

    num_forward_stages = len(forward_stage_layer_ids)

    backward_stage_layer_ids = [[
        2 * num_layers - 1 - i for i in reversed(layer_ids)
    ] for layer_ids in reversed(forward_stage_layer_ids)]
    stage_layer_ids = forward_stage_layer_ids + backward_stage_layer_ids
    stage_to_mesh = list(range(num_forward_stages)) + list(reversed(range(num_forward_stages)))

    stage_outvars = get_stage_outvars(layers, stage_layer_ids, acc_grad_outvars)
    merged_stages = []
    for stage_id, layer_ids in enumerate(stage_layer_ids):
        if len(layer_ids) == 1:
            merged_stages.append(layers[layer_ids[0]])
            continue

        stage_layer_jaxprs = [layers[i].closed_jaxpr() for i in layer_ids]
        stage_name = str(stage_id)
        merged_stage_jaxpr = merge_marked_jaxprs_with_named_call(
            stage_layer_jaxprs,
            stage_outvars[stage_id],
            accumulator_mapping,
            stage_name,
            wrap_with_marker=True)
        merged_stage = JaxPipelineComputation.from_closed_jaxpr(
            stage_name, merged_stage_jaxpr)
        merged_stages.append(merged_stage)
    stages = merged_stages

    # Check the validity of logical mesh shapes
    assert len(logical_mesh_shapes) == len(sliced_meshes)
    for logical_mesh_shape, submesh in zip(logical_mesh_shapes, sliced_meshes):
        assert np.prod(logical_mesh_shape) == submesh.num_devices

    if autosharding_option_dicts is not None:
        assert len(autosharding_option_dicts) == len(sliced_meshes)
    else:
        autosharding_option_dicts = [{}] * len(sliced_meshes)

    manual_stage_option = ManualStageOption(
        forward_stage_layer_ids, tuple(x.shape for x in sliced_meshes),
        logical_mesh_shapes, autosharding_option_dicts)

    timers("stage-construction").stop()
    return stages, stage_to_mesh, sliced_meshes, manual_stage_option


def get_stage_outvars(layers: Sequence[JaxPipelineComputation],
                      layer_assignment, global_outvars) -> List[OrderedSet]:
    """
    Get the outvars of a stage used by another stage by liveness analysis.

    Args:
        layers: clustered layers
        layer_assignment: the assignment of layers to stages
        global_outvars: global outvars

    Returns:
        A list of outvars for each stage
    """
    n_stages = len(layer_assignment)
    used = OrderedSet(global_outvars)
    stage_outvars = [OrderedSet() for _ in range(n_stages)]
    for stage_id, layer_ids in reversed(list(enumerate(layer_assignment))):
        for layer_id in layer_ids:
            for var in layers[layer_id].outvars:
                if var in used:
                    stage_outvars[stage_id].add(var)
            for var in layers[layer_id].invars:
                used.add(var)
    return stage_outvars


def _cluster_layers_with_even_tflops(layers, num_stage):
    # prefix sum: total flops till layer_i
    flops = [0]
    for layer in layers:
        hlo = jaxpr_to_hlo("tmp", layer.closed_jaxpr(),
                           [False] * len(layer.invars))
        layer_flops = xe.hlo_module_count_flop_dot_conv_only(hlo.get_module())
        flops.append(flops[-1] + layer_flops)
    avg_flop = flops[-1] / num_stage
    # the last one is to avoid IndexError
    flops = flops[1:] + [flops[-1] + 1]
    forward_layer_ids = [[-1]]
    nxt_bound = avg_flop
    for i in range(len(layers)):
        # if flops already exceeds threshold or cutting at current layer is
        # closer to the ideal average, then choose it to cut.
        # The first condition is to avoid a too large layer that occupies
        # several times of average flops
        if ((flops[i] >= nxt_bound * (1 - 1e-5)) or
            (flops[i + 1] >= nxt_bound and
             abs(flops[i + 1] - nxt_bound) > abs(flops[i] - nxt_bound))):
            nxt_bound += avg_flop
            forward_layer_ids.append(
                tuple(range(forward_layer_ids[-1][-1] + 1, i + 1)))
    forward_layer_ids = forward_layer_ids[1:]
    return forward_layer_ids


def _print_dp_results(forward_stage_layer_ids, submesh_shapes, logical_mesh_shapes, autosharding_option_dicts, cluster_keys):
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    
    # TODO: Support it
    # raise NotImplementedError("Harp does not support print dp results yet.")
    print(f"{GREEN}Result forward_stage_layer_ids:{RESET}", forward_stage_layer_ids)
    print(f"{GREEN}Result mesh_shapes:{RESET}", submesh_shapes)
    print(f"{GREEN}Result logical_mesh_shapes:{RESET}", logical_mesh_shapes)
    print(f"{GREEN}Result autosharding_option_dicts:{RESET}", autosharding_option_dicts)

    if global_config.enable_Hetero:
        readable_list = []
        for item in submesh_shapes:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                cid, shape = item
                if cluster_keys is not None and cid < len(cluster_keys):
                    cname = cluster_keys[cid]
                    readable_list.append(f"({RED}{cname}{RESET}, {shape})")
                else:
                    readable_list.append(f"({cid}, {shape})")
            else:
                readable_list.append(str(item))
        
        readable_str = f"[{', '.join(readable_list)}]"
        print(f"{GREEN}Result mesh_shapes(readable){RESET}", readable_str)
