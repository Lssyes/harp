"""The entry point of intra-op + inter-op parallelism benchmark."""
import os
import argparse
from datetime import datetime
import time
import gin
import numpy as np

from alpa.util import (write_tsv, get_num_hosts_and_num_devices, to_str_round, GB)
from alpa import init, global_config
from alpa.util import disable_tqdm_globally
from alpa.config.utils import setup_global_envs, get_auto_stage_option

from benchmark_one_case_gpt_bert import benchmark_gpt_bert_3d_internal
from benchmark_parallel_utils import get_benchmark_memory_per_mesh

benchmark_suites = {}


def _build_benchmark_suites():
    import suite_auto_gpt
    # import suite_auto_moe
    # import suite_wresnet

    return {
        "gpt.tmp_auto": suite_auto_gpt.tmp_suite,
        "gpt.perf_test_auto": suite_auto_gpt.perf_test_suite,
        "gpt.grid_search_auto": suite_auto_gpt.grid_search_suite,
        "gpt.correctness_test_auto": suite_auto_gpt.correctness_test_suite,
        "gpt.perf_test_auto_heterogeneous": suite_auto_gpt.perf_test_suite_heterogeneous,
        "gpt.grid_search_auto_heterogeneous": suite_auto_gpt.grid_search_suite_heterogeneous,

        # "moe.tmp_auto": suite_auto_moe.tmp_suite,
        # "moe.perf_test_auto": suite_auto_moe.perf_test_suite,
        # "moe.grid_search_auto": suite_auto_moe.grid_search_suite,
        # "wresnet.perf_test_2d": suite_wresnet.perf_test_2d_suite,
        # "wresnet.perf_test_auto": suite_wresnet.perf_test_auto_suite,
        # "wresnet.grid_search_auto": suite_wresnet.grid_search_auto_suite,
    }

def _check_and_print_global_config(args, cluster_info):
    print("\n\033[1;32m[Global Config]\033[0m")
    for attr in dir(global_config):
        if not attr.startswith("_"):
            value = getattr(global_config, attr)
            print(f"  - {attr}: {value}")
    print("")

    assert global_config.backend == "gpu", "Only GPU backend is supported for benchmark."
    assert global_config.resharding_mode == "broadcast", "Harp only support resharding_mode to be 'broadcast'"
    assert global_config.nccl_mode == "xla_extension", "Harp only support nccl_mode to be 'xla_extension'"
    assert not global_config.use_local_allgather, "Harp only support use_local_allgather to be False"

    if args.enable_hetero:
        assert global_config.enable_Hetero, "--enable-hetero requires global_config.enable_Hetero = True"
        assert cluster_info is not None, "--enable-hetero requires cluster_info to be provided"
        assert len(cluster_info) >= 2, "--enable-hetero requires at least two clusters"
        assert len(cluster_info) == len(global_config.enable_intra_ib), "Length of cluster_info must match length of global_config.enable_intra_ib"
        assert len(cluster_info) == len(global_config.enable_inter_ib), "Length of cluster_info must match length of global_config.enable_inter_ib"
        assert len(cluster_info) == len(global_config.profile_memory_safe_level), "Length of cluster_info must match length of global_config.profile_memory_safe_level"
    else:
        assert not global_config.enable_Hetero, "Heterogeneous cluster config provided without --enable-hetero"
        assert not global_config.enable_H1F1B, "H1F1B config provided without --enable-hetero"

    if global_config.enable_H1F1B:
        assert global_config.enable_Hetero, "H1F1B requires global_config.enable_Hetero = True"
        assert global_config.intra_cluster_network_performance is not None, "H1F1B requires intra_cluster_network_performance to be set"
        assert global_config.inter_cluster_network_performance is not None, "H1F1B requires inter_cluster_network_performance to be set"
        assert global_config.NCCL_MAX_CTAS_CROSS_RESHARDING > 0, "H1F1B requires NCCL_MAX_CTAS_CROSS_RESHARDING to be positive"
        assert global_config.enable_overlapping, "H1F1B requires enable_overlapping to be True"

def _parse_int_list(arg_value, arg_name, parser):
    if arg_value is None:
        return None
    if isinstance(arg_value, list):
        values = arg_value
    else:
        parts = [p.strip() for p in str(arg_value).split(",")]
        if any(p == "" for p in parts):
            parser.error(f"{arg_name} contains empty items. Use comma-separated integers.")
        try:
            values = [int(p) for p in parts]
        except ValueError:
            parser.error(f"{arg_name} must be a comma-separated list of integers.")
    if len(values) == 0:
        parser.error(f"{arg_name} cannot be empty.")
    if any(v <= 0 for v in values):
        parser.error(f"{arg_name} must contain positive integers.")
    return values


def _parse_str_list(arg_value, arg_name, parser):
    if arg_value is None:
        return None
    parts = [p.strip() for p in str(arg_value).split(",")]
    parts = [p for p in parts if p != ""]
    if len(parts) == 0:
        parser.error(f"{arg_name} cannot be empty.")
    return parts


def _get_cluster_configs(args, parser):
    num_hosts_list = _parse_int_list(args.num_hosts, "--num-hosts", parser)
    num_devices_list = _parse_int_list(args.num_devices_per_host,
                                       "--num-devices-per-host", parser)
    gpu_info_list = _parse_str_list(args.gpu_info, "--gpu-info", parser)

    if num_hosts_list is None and num_devices_list is None:
        if args.enable_hetero or args.num_hetero_clusters is not None or gpu_info_list is not None:
            parser.error("Heterogeneous options require --num-hosts and --num-devices-per-host.")
        num_hosts, num_devices_per_host = get_num_hosts_and_num_devices(args)
        return [{
            "num_hosts": num_hosts,
            "num_devices_per_host": num_devices_per_host,
            "gpu_info": None,
        }]

    if num_hosts_list is None or num_devices_list is None:
        parser.error("--num-hosts and --num-devices-per-host must be provided together.")

    if len(num_hosts_list) != len(num_devices_list):
        parser.error("--num-hosts and --num-devices-per-host must have the same length.")

    if args.enable_hetero:
        if getattr(args, "suite", None) not in ["gpt.perf_test_auto_heterogeneous", "gpt.grid_search_auto_heterogeneous"]:
            parser.error("--enable-hetero requires --suite gpt.perf_test_auto_heterogeneous.")
        if args.num_hetero_clusters is None:
            parser.error("--num-hetero-clusters is required when --enable-hetero is set.")
        if args.num_hetero_clusters < 2:
            parser.error("--enable-hetero requires at least two clusters.")
        if args.num_hetero_clusters != len(num_hosts_list):
            parser.error("--num-hetero-clusters must match the number of clusters.")
        if gpu_info_list is None:
            parser.error("--gpu-info is required when --enable-hetero is set.")
        if len(gpu_info_list) != len(num_hosts_list):
            parser.error("--gpu-info must have the same length as --num-hosts.")
    else:
        if args.num_hetero_clusters is not None:
            parser.error("--num-hetero-clusters requires --enable-hetero.")
        if gpu_info_list is not None:
            parser.error("--gpu-info requires --enable-hetero.")
        if len(set(num_devices_list)) != 1:
            parser.error("Heterogeneous devices-per-host detected. Use --enable-hetero.")

    cluster_configs = []
    for idx, (num_hosts, num_devices_per_host) in enumerate(
            zip(num_hosts_list, num_devices_list)):
        cluster_configs.append({
            "num_hosts": num_hosts,
            "num_devices_per_host": num_devices_per_host,
            "gpu_info": gpu_info_list[idx] if gpu_info_list else None,
        })
    return cluster_configs


def _select_suite(suite_name, num_gpus, cluster_info=None):
    suite_dict = benchmark_suites[suite_name]
    if num_gpus in suite_dict:
        return suite_dict[num_gpus]
    if cluster_info is None:
        return None
    num_hosts_list = [c["num_hosts"] for c in cluster_info]
    num_devices_list = [c["num_devices_per_host"] for c in cluster_info]
    key = tuple(h * d for h, d in zip(num_hosts_list, num_devices_list))
    if key in suite_dict:
        return suite_dict[key]
    return None


def benchmark_one_case(model,
                       case,
                       niter,
                       num_hosts,
                       num_devices_per_host,
                       cluster_info,
                       profile_driver_time=False,
                       profile_stage_execution_time=False,
                       disable_tqdm=False):
    if disable_tqdm:
        disable_tqdm_globally()

    global_config.use_dummy_value_for_benchmarking = True
    global_config.pipeline_sync_for_timer = True
    if profile_stage_execution_time:
        global_config.collect_trace = True

    enable_hetero=True if cluster_info is not None else False

    init(cluster_info, enable_hetero)
    # Run benchmark
    if model in ["gpt", "bert"]:
        result = benchmark_gpt_bert_3d_internal(model,
                                                case,
                                                niter,
                                                num_hosts,
                                                num_devices_per_host,
                                                enable_hetero,
                                                profile_driver_time=profile_driver_time)
    # elif model == "moe":
    #     result = benchmark_moe_3d_internal(
    #         case,
    #         niter,
    #         num_hosts,
    #         num_devices_per_host,
    #         profile_driver_time=profile_driver_time)
    # elif model == "wresnet":
    #     global_config.xla_client_mem_fraction = 0.88
    #     # Due to legacy issues, we turn off auto-tuning. Although the
    #     # performance will be much better if we turn it on
    #     global_config.xla_gpu_autotune_level = 0
    #     result = benchmark_wresnet_3d_internal(
    #         case,
    #         niter,
    #         num_hosts,
    #         num_devices_per_host,
    #         profile_driver_time=profile_driver_time)
    else:
        raise ValueError(f"Invalid model: {model}")

    return result




def benchmark_suite(suite_name,
                    num_hosts=None,
                    num_devices_per_host=None,
                    cluster_info=None,
                    exp_name="default",
                    niter=3,
                    profile_driver_time=False,
                    profile_stage_execution_time=False,
                    disable_tqdm=False):
    if cluster_info is not None:
        num_hosts_list = [c["num_hosts"] for c in cluster_info]
        num_devices_list = [c["num_devices_per_host"] for c in cluster_info]
        num_gpus_total = sum(h * d for h, d in zip(num_hosts_list, num_devices_list))
        num_gpus = tuple(h * d for h, d in zip(num_hosts_list, num_devices_list))
        num_hosts = num_hosts_list
        num_devices_per_host = num_devices_list
    else:
        num_gpus_total = num_hosts * num_devices_per_host
        num_gpus = num_gpus_total

    suite = _select_suite(suite_name, num_gpus_total, cluster_info=cluster_info)
    if suite is None:
        key_desc = f"#gpu={num_gpus}"
        print(f"No benchmark suite for {key_desc}")
        return

    os.makedirs("tmp", exist_ok=True)

    model_type = suite_name.split(".")[0]
    output_name = f"{exp_name}.tsv"

    # Run all cases
    for benchmark_case in suite:
        model_config = benchmark_case.model_config
        num_micro_batches = benchmark_case.num_micro_batches
        parallel_args = benchmark_case.parallel_args

        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        result = benchmark_one_case(
            model_type,
            benchmark_case,
            niter,
            num_hosts,
            num_devices_per_host,
            cluster_info,
            profile_driver_time=profile_driver_time,
            profile_stage_execution_time=profile_stage_execution_time,
            disable_tqdm=disable_tqdm)

        (parameter_count, peak_mem, latencies, tflops, metadata) = result

        heads = [
            "Type", "Model Config", "#Microbatch", "#GPU", "Parallel Config",
            "Mean Time (s)", "Std Time (s)", "#Params (Billion)", "TFLOPs",
            "Peak Mem (GB)", "Metadata"
        ]
        values = [
            model_type, model_config, num_micro_batches, num_gpus,
            parallel_args, f"{np.mean(latencies):.3f}",
            f"{np.std(latencies):.3f}", f"{parameter_count/1e9:.3f}B",
            f"{tflops:.2f}", f"{get_benchmark_memory_per_mesh(peak_mem)}",
            to_str_round(metadata, 2)
        ]
        write_tsv(heads, values, output_name)

        time.sleep(0.1)  # for ctrl+c to work


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",
                        type=str,
                        required=True)
    parser.add_argument("--niter",
                        type=int,
                        default=3,
                        help="The number of benchmark iterations")
    parser.add_argument("--num-hosts",
                        type=str,
                        default=None,
                        help="Comma-separated list of host counts per cluster, e.g. 1,2")
    parser.add_argument("--num-devices-per-host",
                        type=str,
                        default=None,
                        help="Comma-separated list of device counts per host, e.g. 8,8")
    parser.add_argument("--enable-hetero",
                        action="store_true",
                        help="Enable heterogeneous cluster mode.")
    parser.add_argument("--num-hetero-clusters",
                        type=int,
                        default=None,
                        help="Number of heterogeneous clusters.")
    parser.add_argument("--gpu-info",
                        type=str,
                        default=None,
                        help="Comma-separated GPU type per cluster, e.g. A100,V100")
    parser.add_argument("--gin-file", type=str, default=None, help="Path to gin config file.")
    parser.add_argument("--profile-driver-time",
                        action="store_true",
                        help="Profile the execution time on the driver instead "
                        "of the workers.")
    parser.add_argument(
        "--profile-stage-execution-time",
        action="store_true",
        help="Profile the execution timestamps of each pipeline "
        "stage")
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    cluster_info = _get_cluster_configs(args, parser)
    if args.gin_file:
        gin.parse_config_file(args.gin_file)
    benchmark_suites = _build_benchmark_suites()
    if args.suite not in benchmark_suites:
        parser.error(f"--suite must be one of: {sorted(benchmark_suites.keys())}")
    setup_global_envs(global_config)
    _check_and_print_global_config(args, cluster_info)    # 由于 Alpa 没有采用 gin.config, 为了保证 args 和 gin 一致, 需要在 setup_global_envs 之后检查

    print("\n\033[1;32m[cluster_info]\033[0m")
    for idx, c in enumerate(cluster_info):
        print(f"  - Cluster {idx}: num_hosts={c['num_hosts']}, num_devices_per_host={c['num_devices_per_host']}, gpu_info={c['gpu_info']}")

    if args.enable_hetero:
        selected_cluster_info = cluster_info
        selected_num_hosts = None
        selected_num_devices_per_host = None
    else:
        selected_cluster_info = None
        selected_num_hosts = cluster_info[0]["num_hosts"]
        selected_num_devices_per_host = cluster_info[0]["num_devices_per_host"]

    benchmark_suite(args.suite,
                    selected_num_hosts,
                    selected_num_devices_per_host,
                    selected_cluster_info,
                    args.exp_name,
                    args.niter,
                    args.profile_driver_time,
                    args.profile_stage_execution_time,
                    args.disable_tqdm)
