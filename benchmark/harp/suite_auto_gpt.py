"""Benchmark suites for gpt with auto parallelization."""
from collections import namedtuple
from benchmark_parallel_utils import BenchmarkCase, SearchParallelArgs, LoadSolutionParallelArgs
from alpa.config.utils import get_auto_stage_option

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# RS = prefer_reduce_scatter, Remat = use_rematerialization,
# FM = force_batch_dim_mapping,
GPTModelConfig = namedtuple(
    "GPTModelConfig",
    ["seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size"])

gpt_specs = {
    #                      S，   H,   L,  head,   V,
    "1M": GPTModelConfig(32, 128, 6, 2, 1000),
    "125M": GPTModelConfig(1024, 768, 12, 12, 51200),
    "350M": GPTModelConfig(1024, 1024, 24, 16, 51200),
    "760M": GPTModelConfig(1024, 1536, 24, 16, 51200),
    "1.3B": GPTModelConfig(1024, 2048, 24, 32, 51200),
    "2.6B": GPTModelConfig(1024, 2560, 32, 32, 51200),
    "6.7B": GPTModelConfig(1024, 4096, 32, 32, 51200),
    "15B": GPTModelConfig(1024, 5120, 48, 40, 51200),
    "39B": GPTModelConfig(1024, 8192, 48, 64, 51200),
    "76B": GPTModelConfig(1024, 10240, 60, 80, 51200),
}
max_global_batch_size = 1024

prefer_reduce_scatter = True
use_remat = True


def get_search_cases(model_spec, num_micro_batches_list, num_auto_layers_list):
    auto_stage_option = get_auto_stage_option()
    return [
        BenchmarkCase(
            max_global_batch_size, model_spec, num_micro_batches, "search",
            SearchParallelArgs(prefer_reduce_scatter, use_remat,
                               num_auto_layers, auto_stage_option))
        for num_micro_batches in num_micro_batches_list
        for num_auto_layers in num_auto_layers_list
    ]


def get_solution_case(model_spec, num_micro_batches, num_auto_layers,
                      forward_stage_layer_ids, submesh_physical_shapes,
                      submesh_logical_shapes,
                      submesh_autosharding_option_dicts,
                      maunal_schedule_strategy=None):
    return [
        BenchmarkCase(
            max_global_batch_size, model_spec, num_micro_batches,
            "load_solution",
            LoadSolutionParallelArgs(prefer_reduce_scatter, use_remat,
                                     num_auto_layers, forward_stage_layer_ids,
                                     submesh_physical_shapes,
                                     submesh_logical_shapes,
                                     submesh_autosharding_option_dicts,
                                     maunal_schedule_strategy=maunal_schedule_strategy))
    ]


force_dp_dict = {"force_batch_dim_to_mesh_dim": 0}

# Temporary debug suite
tmp_suite = {}

perf_test_suite_heterogeneous = {
    (4, 4): get_solution_case(gpt_specs["1M"], 512, 3, 
                                 [[0], [1], [2]],
                                 [(0, (1, 4)), (1, (1, 2)), (1, (1, 2))],
                                 [(2, 2), (2, 1), (2, 1)],
                                 [force_dp_dict] * 3,
                                 maunal_schedule_strategy=None),
    (4, 4): get_solution_case(gpt_specs["1M"], 512, 3, 
                                 [[0], [1], [2]],
                                 [(0, (1, 4)), (1, (1, 2)), (1, (1, 2))],
                                 [(2, 2), (2, 1), (2, 1)],
                                 [force_dp_dict] * 3,
                                 maunal_schedule_strategy=None),


    (2, 1, 1): get_solution_case(gpt_specs["1M"], 64, 2,             #
                                 [list(range(0, 5)), list(range(5, 10)), list(range(10, 14))],
                                 [(0, (1, 2)), (1, (1, 1)), (2, (1, 1))],
                                 [(2, 1), (1, 1), (1, 1)],
                                 [force_dp_dict] * 3,
                                 maunal_schedule_strategy=None),

    (8, 2, 2): get_solution_case(gpt_specs["2.6B"], 64, 2,       
                                 [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], 
                                  [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], 
                                  [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]],
                                  [[0, [1, 8]], [1, [1, 2]], [2, [1, 2]]],
                                  [[8, 1], [2, 1], [2, 1]],
                                 [force_dp_dict] * 3,
                                 maunal_schedule_strategy=None),

}

grid_search_suite_heterogeneous = {
    (2, 2): get_search_cases(gpt_specs["1M"], [32], [2]),
    (2, 1, 2): get_search_cases(gpt_specs["1M"], [32], [2]),
    (8, 2, 2): get_search_cases(gpt_specs["2.6B"], [64], [2]),
    (8, 4): get_search_cases(gpt_specs["2.6B"], [64], [2])
}

# Performance test with search solutions found for p3.16xlarge
perf_test_suite = {
    1:
        get_solution_case(gpt_specs["350M"], 512, 1, [[0]], [(1, 1)], [(1, 1)],
                          [{}]),
    2:
        get_solution_case(gpt_specs["760M"], 128, 6, [[0, 1, 2], [3, 4, 5]],
                          [(1, 1)] * 2, [(1, 1)] * 2, [force_dp_dict] * 2),
    4:
        get_solution_case(gpt_specs["1M"], 128, 6, [[0, 1, 2], [3, 4, 5]],
                          [(1, 2)] * 2, [(2, 1)] * 2, [force_dp_dict] * 2,
                          maunal_schedule_strategy=None),
    8:
        get_solution_case(gpt_specs["350M"], 128,
                          8, [[0, 1], [2, 3], [4, 5, 6, 7]], [(1, 2), (1, 2),
                                                              (1, 4)], [(2, 1),
                                                                        (2, 1),
                                                                        (4, 1)],
                          [force_dp_dict, {}, {}]),
    16:
        get_solution_case(gpt_specs["6.7B"], 64, 8,
                          [[0, 1, 2, 3], [4, 5, 6, 7]], [(1, 8)] * 2,
                          [(2, 4)] * 2, [force_dp_dict] * 2),
    32:
        get_solution_case(
            gpt_specs["15B"], 128, 16,
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            [(1, 8)] * 4, [(2, 4)] * 4, [force_dp_dict] * 4),
    64:
        get_solution_case(gpt_specs["39B"], 1024,
                          16, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9],
                               [10], [11], [12], [13], [14], [15]],
                          [(1, 4)] * 16, [(1, 4)] * 16, [force_dp_dict] * 16),
}

# Grid search on hyperparameters
grid_search_suite = {
    2: (get_search_cases(gpt_specs["760M"], [32, 64, 128, 256], [6]) +
        get_search_cases(gpt_specs["760M"], [32, 64], [12])),
    4: (get_search_cases(gpt_specs["350M"], [32], [2]) +
        get_search_cases(gpt_specs["350M"], [32], [4])),
    # 4: (get_search_cases(gpt_specs["1.3B"], [32, 64, 128], [6]) +
    #     get_search_cases(gpt_specs["1.3B"], [32, 64], [12])),
    8: (get_search_cases(gpt_specs["2.6B"], [64, 128, 256], [8]) +
        get_search_cases(gpt_specs["2.6B"], [64, 128], [16])),
    16: get_search_cases(gpt_specs["6.7B"], [32, 64, 128, 256], [8]),
    32: get_search_cases(gpt_specs["15B"], [64, 128, 256, 512], [16]),
    64: get_search_cases(gpt_specs["39B"], [128, 256, 512, 1024], [8]),
}

# Small test cases for correctness test
correctness_test_suite = {
    8: get_search_cases(gpt_specs["2.6B"], [128], [8]),
}
