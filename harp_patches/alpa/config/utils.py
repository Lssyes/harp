
import gin

@gin.configurable
def setup_global_envs(
    self: object,
    
    # 目前仅支持同时开启 gd 和 heter, 没有做分离

    enable_geo_distribued: bool = False,
    enable_zero_redundant_profiler : bool = False,
    enable_heterogeneous: bool = False,
    
    layer_construction_eps: float = 0.5,
    NCCL_MAX_CTAS_CROSS_RESHARDING: int = -1,
    pipeline_use_signal_send_recv: bool = False,
    
    # 0 表示不开启, n 表示 min mp size 的大小为 2^n, 设置超过单个节点上的 device数量时 会被限制为 tp only.
    # only useful when submesh_logical_shape_space = "single_node_model_parallel_and_safememory_dp",
    profile_memory_safe_level: list = [0, 0],
        
    DeivceMesh_Network_Performance: dict = None,
    enable_intra_ib: list = [0, 0],
    enable_inter_ib: bool = False,
    
    
    enable_stage_cache: bool = False,
    enable_profile_cache: bool = False,
    load_stage_file: list = [],
    load_profile_file: list = [],
    dp_tmp_file : str = "tmp/dp_init_args.npz",
    enable_nsys: bool = False,
):
    self.enable_geo_distribued = enable_geo_distribued
    self.enable_zero_redundant_profiler = enable_zero_redundant_profiler    
    self.enable_heterogeneous = enable_heterogeneous
    self.layer_construction_eps = layer_construction_eps
    self.NCCL_MAX_CTAS_CROSS_RESHARDING = NCCL_MAX_CTAS_CROSS_RESHARDING
    self.pipeline_use_signal_send_recv = pipeline_use_signal_send_recv
    self.profile_memory_safe_level = profile_memory_safe_level
    if DeivceMesh_Network_Performance is not None:
        self.mesh_performance = DeivceMesh_Network_Performance
        self.communciation_scale = 1 
    self.intra_use_ib = enable_intra_ib
    self.inter_use_ib = enable_inter_ib
    self.enable_stage_cache = enable_stage_cache
    self.enable_profile_cache = enable_profile_cache
    if enable_stage_cache or enable_profile_cache:
         print(f"\033[1;32m[Global Env] Enable stage cache: {enable_stage_cache}, enable profile cache: {enable_profile_cache}\033[0m")
         # 使用 stage_cache 时，需要自己确保是正确的
         self.load_stage_file = load_stage_file
         self.load_profile_file = load_profile_file
    self.exp_name = "exp-"
        
    
    self.dp_tmp_file = dp_tmp_file
    
    
    self.nvtx_trace = enable_nsys
    if enable_nsys:
        from datetime import datetime
        nsight_output_file = f"{datetime.now().strftime('%m-%d____%H:%M:%S')}-p-%p"
        self.nsight_config = {
                                    "t": "cuda,nvtx",
                                    "o": f"'{nsight_output_file}'",
                                    "x": "true", "f": "true",
                                    "trace-fork-before-exec": "true"
                                    }

    
    
# homo
# auto_stage_option = {
#     "submesh_physical_shape_space": "power_of_two",
#     "submesh_physical_shape_space": "single_node_power_of_two",
#     "submesh_logical_shape_space": "all",
#     "stage_imbalance_tolerance": 0.5,
#     "use_hlo_cost_model": True,
#     "profiling_database_filename": DATABASE_PATH[0],
#     "cluster_key": "RTX4080",
# }

# hetero
# auto_stage_option = {
#     # "submesh_physical_shape_space": "power_of_two",
#     "submesh_physical_shape_space": "single_node_power_of_two",
#     "submesh_logical_shape_space": "single_node_model_parallel_and_safememory_dp",
#     "stage_imbalance_tolerance": 0.3,
#     "use_hlo_cost_model": True,
#     "profiling_database_filename": [DATABASE_PATH[0], DATABASE_PATH[1]],
#     "gpu_flops": [180, 113],
#     "cluster_key": ["RTX4080", "RTX4090"],
# }

    
def set_auto_stage_option(
    
):
    pass