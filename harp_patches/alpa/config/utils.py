import gin


@gin.configurable
def setup_global_envs(
    self: object,


    enable_H1F1B: bool,
    enable_ZRP  : bool,  # zero redundant profiler     # 目前仅支持同时开启 h1f1b 和 heter, 没有做分离
    enable_Hetero: bool,
    


    
    layer_construction_eps: float,
    NCCL_MAX_CTAS_CROSS_RESHARDING: int,

    # 0 表示不开启, n 表示 min mp size 的大小为 2^n, 设置超过单个节点上的 device数量时 会被限制为 tp only.
    # only useful when submesh_logical_shape_space = "single_node_model_parallel_and_safememory_dp",
    profile_memory_safe_level: list,    #TODO: 移动到 autosharding 里更合适一些
        
    intra_cluster_network_performance: list,
    inter_cluster_network_performance: list,
    enable_intra_ib: list,    #num = num of clusters
    enable_inter_ib: list,    #num = num of clusters, 假设 enable_inter_ib[i] 代表 cluster i 和 cluster (i+1)%num 之间的连接是否使用 ib    
    
    
    # enable_stage_cache: bool = False,
    # enable_profile_cache: bool = False,
    # load_stage_file: list = [],
    # load_profile_file: list = [],
    # dp_tmp_file : str = "tmp/dp_init_args.npz",

    enable_nsys: bool = False,
):
    self.enable_H1F1B = enable_H1F1B
    self.enable_ZRP = enable_ZRP    
    self.enable_Hetero = enable_Hetero

    self.layer_construction_eps = layer_construction_eps
    self.NCCL_MAX_CTAS_CROSS_RESHARDING = NCCL_MAX_CTAS_CROSS_RESHARDING
    self.profile_memory_safe_level = profile_memory_safe_level
    if intra_cluster_network_performance is not None:
        assert inter_cluster_network_performance is not None, "If intra_cluster_network_performance is set, inter_cluster_network_performance must also be set"
        self.intra_cluster_network_performance = intra_cluster_network_performance
        self.inter_cluster_network_performance = inter_cluster_network_performance
        self.communication_scale = 1 


        
    self.enable_intra_ib = enable_intra_ib
    self.enable_inter_ib = enable_inter_ib
    # self.enable_stage_cache = enable_stage_cache
    # self.enable_profile_cache = enable_profile_cache
    # if enable_stage_cache or enable_profile_cache:
    #      print(f"\033[1;32m[Global Env] Enable stage cache: {enable_stage_cache}, enable profile cache: {enable_profile_cache}\033[0m")
    #      # 使用 stage_cache 时，需要自己确保是正确的
    #      self.load_stage_file = load_stage_file
    #      self.load_profile_file = load_profile_file
    # self.dp_tmp_file = dp_tmp_file

    from datetime import datetime
    datatime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    self.exp_name = f"exp-{datatime_str}"
        

    self.enable_nsys = enable_nsys
    if enable_nsys:
        nsys_output_file = f"{datatime_str}-p-%p"
        self.nsys_config = {
                                "t": "cuda,nvtx",
                                "o": f"'{nsys_output_file}'",
                                "x": "true", "f": "true",
                                "trace-fork-before-exec": "true"
                            }