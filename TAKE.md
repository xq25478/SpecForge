# GIT 操作

# TRT 

- trtllm-serve 

```
"""
TensorRT-LLM 服务启动脚本使用说明

三种使用场景及对应参数配置：

1. PyTorch 后端模式 (需手动配置参数)
----------------------------------
适用场景：使用原生PyTorch后端运行模型
参数要求：必须指定 --backend pytorch
示例命令：
    python3 trtllm-serve.py \
    --model_path /path/to/model/ \
    --backend pytorch \
    [其他可选参数]

2. TRT 引擎自动模式 (自动读取配置)
----------------------------------
适用场景：使用预编译的TRT引擎文件
参数要求：
  - 不指定 --backend 参数
  - 模型路径必须包含有效的.engine文件
示例命令：
    python3 trtllm-serve.py \
    --model_path /path/to/trt_engine/

3. TRT 后端手动模式 (手动配置参数)
----------------------------------
适用场景：动态生成TRT引擎并运行
参数要求：
  - 不指定 --backend 参数
  - 必须提供运行参数配置
示例命令：
    python3 trtllm-serve.py \
    --model_path /path/to/model/ \
    --max_batch_size 32 \
    --max_num_tokens 8192 \
    [其他TRT参数]

服务管理命令：
终止所有trtllm服务进程：
    ps -ef | grep "trtllm-serve" | grep -v grep | awk '{print $2}' | xargs kill -9

注意事项：
1. 模式2和模式3通过是否提供运行参数自动区分
2. PyTorch模式需要额外依赖torch环境
3. 所有路径建议使用绝对路径
"""

import os
import sys
import argparse
import subprocess
import json
import yaml
import torch
import pty
import select
import signal
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from tabulate import tabulate

# 颜色定义
COLOR = {
    'GREEN': '\033[0;32m',
    'RED': '\033[0;31m',
    'YELLOW': '\033[0;33m',
    'BLUE': '\033[0;34m',
    'NC': '\033[0m'
}


def print_colored(text: str, color: str) -> None:
    """彩色打印输出"""
    print(f"{color}{text}{COLOR['NC']}")


def print_dict(data: dict, color: str = COLOR['GREEN']) -> None:
    """格式化打印字典"""
    formatted = json.dumps(data, indent=2, sort_keys=True)
    print(f"{color}{formatted}{COLOR['NC']}")


def kill_trtllm_serve() -> None:
    """终止所有 trtllm-serve 进程"""
    try:
        subprocess.run(
            "ps -ef | grep 'trtllm-serve' | grep -v grep | awk '{print $2}' | xargs kill -9",
            shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print_colored("Successfully killed trtllm-serve processes", COLOR['YELLOW'])
    except subprocess.CalledProcessError as e:
        print_colored(f"Failed to kill processes: {e.stderr.decode()}", COLOR['RED'])


def check_for_engine_files(model_path: str) -> bool:
    """检查模型路径中是否存在TRT引擎文件"""
    for root, _, files in os.walk(model_path):
        if any(f.endswith('.engine') for f in files):
            return True
    return False


def read_config_json(model_path: str) -> Optional[Dict[str, Any]]:
    """读取模型配置文件"""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def setup_logging(model_path: str, args) -> str:
    """设置日志目录和文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path.rstrip('/'))
    log_dir = "/tmp"
    
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir,
            f"{model_name}_{timestamp}_"
            f"tp{args.tp_size}_pp{args.pp_size}_ep{args.ep_size}_"
            f"mbs{args.max_batch_size}.log"
        )
        return log_file
    except OSError as e:
        print_colored(f"Error creating log directory: {str(e)}", COLOR['RED'])
        sys.exit(1)


def run_command(command: List[str], log_file: str) -> int:
    """执行命令并实时输出日志"""
    def handle_interrupt(signum, frame):
        print_colored(f"\nReceived signal {signum}, cleaning up...", COLOR['YELLOW'])
        if 'process' in locals():
            process.terminate()
        kill_trtllm_serve()
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    try:
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            command,
            stdout=slave_fd,
            stderr=slave_fd,
            stdin=subprocess.PIPE,
            close_fds=True,
            start_new_session=True
        )
        
        with open(log_file, 'a') as log_f:
            while True:
                r, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in r:
                    data = os.read(master_fd, 1024)
                    if not data:
                        break
                    decoded = data.decode(errors='replace')
                    sys.stdout.write(decoded)
                    log_f.write(decoded)
                    log_f.flush()
                
                if process.poll() is not None:
                    break
        
        return process.wait()
    
    except Exception as e:
        print_colored(f"Error executing command: {str(e)}", COLOR['RED'])
        kill_trtllm_serve()
        return 1
    finally:
        if 'master_fd' in locals(): os.close(master_fd)
        if 'slave_fd' in locals(): os.close(slave_fd)


def print_config_table(args) -> None:
    """打印配置表格"""
    config_data = [
        ("model_path", args.model_path),
        ("host:port", f"{args.host}:{args.port}"),
        ("tp_size", args.tp_size),
        ("ep_size", args.ep_size),
        ("pp_size", args.pp_size),
        ("kv_cache_fraction", args.kv_cache_free_gpu_memory_fraction),
        ("max_batch_size", args.max_batch_size),
        ("max_num_tokens", args.max_num_tokens),
        ("max_seq_len", args.max_seq_len if args.max_seq_len is not None else "Not set"),
        ("attn_backend", args.attn_backend),
        ("cuda_graph_max_batch_size", args.cuda_graph_max_batch_size),
        ("dtype", args.dtype),
        ("backend", "TRT" if args.backend == "trt" else "PyTorch")
    ]
    
    print(f"\n{COLOR['GREEN']}Configuration Parameters:{COLOR['NC']}")
    print(tabulate(
        config_data,
        headers=["Parameter", "Value"],
        tablefmt="grid",
        stralign="left",
        colalign=("left", "left")
    ))


def build_base_command(model_path: str, args) -> List[str]:
    """构建基础命令"""
    command = ["trtllm-serve", model_path]
    command.extend([
        "--host", args.host,
        "--port", str(args.port),
        "--kv_cache_free_gpu_memory_fraction", str(args.kv_cache_free_gpu_memory_fraction),
        "--trust_remote_code",
        "--log_level", "info"
    ])

    if args.tracing_level:
        command.extend(["--otlp_traces_endpoint",args.otlp_traces_endpoint])

    if args.reasoning_prefix:
        command.extend(["--reasoning_prefix",args.reasoning_prefix])
        
    return command


def build_trt_command(model_path: str, args) -> List[str]:
    """构建TRT后端命令"""
    command = build_base_command(model_path, args)
    params = {}
    
    if check_for_engine_files(model_path):
        tokenizer_path = os.path.join(model_path, "tokenizer")
        config = read_config_json(model_path)
        if config:
            build_config = config.get("build_config", {})
            pretrained_config = config.get("pretrained_config", {}).get("mapping", {})
            params = {
                "max_batch_size": build_config.get("max_batch_size"),
                "max_num_tokens": build_config.get("max_num_tokens"),
                "max_seq_len": build_config.get("max_seq_len"),
                "tp_size": pretrained_config.get("tp_size"),
                "pp_size": pretrained_config.get("pp_size"),
                "ep_size": pretrained_config.get("moe_ep_size")
            }
    else:
        params = {
            "max_batch_size": args.max_batch_size,
            "max_num_tokens": args.max_num_tokens,
            "tp_size": args.tp_size,
            "pp_size": args.pp_size,
            "ep_size": args.ep_size
        }
        tokenizer_path = model_path
        if args.max_seq_len is not None:
            params["max_seq_len"] = str(args.max_seq_len)

    if os.path.exists(tokenizer_path):
        command.extend(["--tokenizer", tokenizer_path])

    for param, value in params.items():
        if value is not None:
            command.extend([f"--{param}", str(value)])
    
    return command


def build_torch_command(model_path: str, args) -> List[str]:
    """构建PyTorch后端命令"""
    command = build_base_command(model_path, args)
    command.extend([
        "--tp_size", str(args.tp_size),
        "--ep_size", str(args.ep_size),
        "--pp_size", str(args.pp_size),
        "--max_batch_size", str(args.max_batch_size),
        "--max_num_tokens", str(args.max_num_tokens),
        "--tokenizer", model_path,
        "--backend", "pytorch",
        "--extra_llm_api_options", "/tmp/extra_llm_api_options.yaml"  
    ])
    
    if args.max_seq_len is not None:
        command.extend(["--max_seq_len",str(args.max_seq_len)])

    if args.tracing_level > 1:
        command.extend(["--use_enhanced_tracing",str(args.use_enhanced_tracing)])
    
    return command


def get_trtllm_version() -> Optional[str]:
    """获取TensorRT-LLM版本"""
    try:
        result = subprocess.run(
            ["pip", "show", "tensorrt_llm"],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                version = line.split(':')[1].strip()
                return version.split('-')[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return None


def create_extra_llm_config(args, trtllm_version="v1.0.0rc3") -> Dict[str, Any]:
    VERSION_CONFIGS = {
        "0.20.0rc1": {
            "kv_cache": {
                "config_key": "kv_cache_config",
                "dtype_key": "kv_cache_dtype",
                "cuda_graph": {"padding_key": "padding_enabled"}
            },
            "cuda_graph":{
                "padding_key":"cuda_graph_padding_enabled",
                "use_cuda_graph":"use_cuda_graph",
                "cuda_graph_max_batch_size":"cuda_graph_max_batch_size"
            },
            "speculative": {"config_key": "speculative_config"},
        },
        "0.20.0rc3": {
            "kv_cache": {
                "config_key": "kv_cache_config",
                "dtype_key": "kv_cache_dtype",
                "cuda_graph": {"padding_key": "padding_enabled"}
            },
            "cuda_graph":{
                "padding_key":"cuda_graph_padding_enabled",
                "use_cuda_graph":"use_cuda_graph",
                "cuda_graph_max_batch_size":"cuda_graph_max_batch_size"
            },
            "speculative": {"config_key": "speculative_config"},
        },
        "1.0.0rc4": {
            "kv_cache": {
                "config_key": "kv_cache_config",
                "dtype_in_config": True,     
            },
            "cuda_graph": {"padding_key": "enable_padding"},
            "speculative": {"config_key": "speculative_config"}
        },
        "1.0.0rc6": {
            "kv_cache": {
                "config_key": "kv_cache_config",
                "dtype_key": "dtype",
                "dtype_in_config": True,     
            },
            "cuda_graph": {"padding_key": "enable_padding"},
            "speculative": {"config_key": "speculative_config"}
        }
    }

    # 检查版本是否支持
    if trtllm_version not in VERSION_CONFIGS:
        raise ValueError(f"Unsupported TensorRT-LLM version: {trtllm_version}")
    version_config = VERSION_CONFIGS[trtllm_version]
    
    # 基础配置 (所有版本通用)
    config = {
        "enable_chunked_prefill": not args.disable_chunked_prefill,
        "disable_overlap_scheduler": args.disable_overlap_scheduler,
        "enable_trtllm_sampler": not args.disable_trtllm_sampler,
        "allreduce_strategy": args.allreduce_strategy,
        "attn_backend": args.attn_backend,
        "enable_iter_perf_stats": not args.disable_iter_stats,
        "enable_iter_req_stats": not args.disable_iter_stats,
        "enable_attention_dp": args.enable_attention_dp,
        "print_iter_log": args.print_iter_log,
        "dtype": args.dtype
    }

    # KV缓存配置
    kv_cache_config = {
        "enable_block_reuse": not args.disable_block_reuse,
        "free_gpu_memory_fraction": args.kv_cache_free_gpu_memory_fraction
    }
    
    if version_config["kv_cache"].get("dtype_in_config", False):
        kv_cache_config[version_config["kv_cache"].get("dtype_key","kv_cache_dtype")] = args.kv_cache_dtype
        config["kv_cache_config"] = kv_cache_config
    else:
        config["kv_cache_config"] = kv_cache_config
        config[version_config["kv_cache"].get("dtype_key","kv_cache_dtype")] = args.kv_cache_dtype

    # CUDA Graph配置
    if not args.disable_cuda_graph:
        if trtllm_version in [ "0.20.0rc1","0.20.0rc3"]:
            pytorch_backend_config = {
                "use_cuda_graph":True,
                "cuda_graph_max_batch_size":args.cuda_graph_max_batch_size,
                "cuda_graph_padding_enabled": True
            }
            config["pytorch_backend_config"] = pytorch_backend_config
        else:
            cuda_graph_config = {
                version_config["cuda_graph"]["padding_key"]: not args.padding_disabled
            }
            if args.cuda_graph_batch_sizes is not None:
                cuda_graph_config["batch_sizes"] = args.cuda_graph_batch_sizes
            else:
                cuda_graph_config["max_batch_size"] = args.cuda_graph_max_batch_size
            
            config["cuda_graph_config"] = cuda_graph_config

    # 投机采样配置
    if getattr(args, 'spec_algo', None):
        if args.spec_algo == "eagle3":
            assert args.draft_model_path is not None, "draft_model_path must be specified for eagle3"
            assert args.max_draft_len is not None, "max_draft_len must be specified for eagle3"
            config[version_config["speculative"]["config_key"]] = {
                "decoding_type":"Eagle",
                "max_draft_len":args.max_draft_len,
                "speculative_model_dir":args.draft_model_path,
                "eagle3_one_model":True
            }
        else:
            print_colored("Invalid spec_config JSON format", COLOR['RED'])
            sys.exit(1)

    # Scheduler 配置
    scheduler_config = {"context_chunking_policy": args.context_chunking_policy}
    config['scheduler_config'] = scheduler_config

    # PD 分离配置
    if args.enable_disaggregation_mode:
        if trtllm_version in ["0.20.0rc1", "0.20.0rc3"]:
            print_colored(f"Enable disaggregation mode with trtllm version: {trtllm_version}, "
                          f"disaggregation params should be set through environment vars", COLOR['YELLOW'])
        else:
            cache_transceiver_config = {
                "backend": args.disaggregation_transfer_backend,
                "max_tokens_in_buffer": args.disaggregation_transfer_buffer_size
            }
            config["cache_transceiver_config"] = cache_transceiver_config

    return config


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="TensorRT-LLM Model Serving",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False
    )

    # 基础参数
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS,
                      help="Show this help message and exit")
    
    parser.add_argument("--backend", choices=["trt", "pytorch"], default="trt",
                      help="Select backend type (default: TRT)")
    
    parser.add_argument("--model_path", type=str, required=True,
                      help="Model path (TRT: Engine file path, PyTorch: Model weight path)")
    
    parser.add_argument("--host", type=str, default="127.0.0.1",
                      help="Service listening address (default: 127.0.0.1)")
    
    parser.add_argument("--port", type=int, default=9122,
                      help="Service listening port (default: 9122)")

    # 性能参数
    parser.add_argument("--max_batch_size", type=int, default=256,
                      help="Max batch size (default: 256)")
    
    parser.add_argument("--max_num_tokens", type=int, default=32768,
                      help="Max number of tokens (default: 32768)")

    parser.add_argument("--kv_cache_free_gpu_memory_fraction", type=float, default=0.85,
                      help="KV cache GPU memory fraction (default: 0.85)")
    
    parser.add_argument("--max_seq_len", type=int, default=None,
                      help="Maximum sequence length")

    # 并行参数
    parser.add_argument("--tp_size", type=int, default=torch.cuda.device_count(),
                      help="Tensor parallel size (default: GPU count)")
    
    parser.add_argument("--pp_size", type=int, default=1,
                      help="Pipeline parallel size (default: 1)")
    
    parser.add_argument("--ep_size", type=int, default=1,
                      help="Expert parallel size (default: 1)")

    # Torch Backend 参数
    parser.add_argument("--attn_backend", default="TRTLLM",
                      choices=["FLASHINFER", "TRTLLM"],
                      help="Attention mechanism backend (default: TRTLLM)")
    
    parser.add_argument("--cuda_graph_max_batch_size", type=int, default=64,
                      help="CUDA graph batch size (default: 64)")
    
    parser.add_argument('--cuda_graph_batch_sizes', nargs='+', type=int, default=None)
    
    parser.add_argument("--dtype", default="auto",
                      choices=["bfloat16", "float16", "auto"],
                      help="Precision type (default: auto)")
    
    parser.add_argument("--kv_cache_dtype", default="auto",
                      choices=["fp8", "auto"],
                      help="KV cache precision type (default: auto)")

    # 功能开关
    parser.add_argument("--disable_block_reuse", action="store_true", default=False,
                      help="Disable block reuse")
    
    parser.add_argument("--disable_chunked_prefill", action="store_true", default=False,
                      help="Disable chunked prefill")
    
    parser.add_argument("--padding_disabled", action="store_true", default=False,
                      help="Disable padding")
    
    parser.add_argument("--disable_overlap_scheduler", action="store_true", default=False,
                      help="Disable overlap scheduler")
    
    parser.add_argument("--disable_trtllm_sampler", action="store_true", default=False,
                      help="Disable TRTLLM sampler")
    
    parser.add_argument("--disable_cuda_graph", action="store_true", default=False,
                      help="Disable CUDA graph")
    
    parser.add_argument("--print_iter_log", action="store_true", default=False,
                      help="Print iteration log")
    
    parser.add_argument("--disable_iter_stats", action="store_true", default=False,
                      help="Disable iteration perf and req stats")

    parser.add_argument("--enable_disaggregation_mode", action="store_true", default=False,
                      help="Enable disaggregation mode")

    parser.add_argument("--enable_attention_dp", action="store_true", default=False,
                      help="Enable attention dp")

    # 调度参数
    parser.add_argument("--context_chunking_policy", default="FIRST_COME_FIRST_SERVED",
                      help="context_chunking_policy")
                
    # Antmointor参数
    parser.add_argument("--tracing_level", type = int, default=0,
                      help="0 for no tracing, 1 for normal tracing, 2 for enhanced tracing")

    parser.add_argument("--otlp_traces_endpoint", type = str, default="https://antcollector.alipay.com/namespace/aicloud/task/otlptrace/otlp/api/v1/traces",
                      help="otlp_traces_endpoint")
                    
    parser.add_argument("--use_enhanced_tracing", type = int, default=5,
                      help="use_enhanced_tracing")
    
    # 投机采样参数
    parser.add_argument("--spec_algo", type=str, default=None,
                      help='''Speculative decoding config in JSON format''')

    parser.add_argument("--draft_model_path", type=str, default=None,
                      help='''Speculative decoding config in JSON format''')

    parser.add_argument("--max_draft_len", type=str, default=None,
                      help='''Speculative decoding config in JSON format''')

    # PD 分离参数
    parser.add_argument("--disaggregation_transfer_backend", type=str, default="ucx",
                      help="The communication backend type to use for the cache transceiver."
                           "Supported values are ucx, nixl, mpi")

    parser.add_argument("--disaggregation_transfer_buffer_size", type=int, default=32768,
                      help="The max number of tokens the transfer buffer can fit."
                           "Usually set to be slightly larger than max sequence length")

    # 其他参数
    parser.add_argument("--allreduce_strategy", type=str, default="AUTO",
                      help="Allreduce strategy (default: AUTO)")
        
    parser.add_argument("--reasoning_prefix", type=str, default=None)

    parser.add_argument("--torch_cuda_arch_list", type=str, default="8.0;8.6;8.9;9.0",
                      help="CUDA architectures for torch (default: 8.0;8.6;8.9;9.0)")                    
    
    return parser.parse_args()


def mv_cuda_compat(src_dir,dst_dir):
    try:
        if os.path.exists(src_dir) and os.path.isdir(src_dir):
            if os.path.exists(dst_dir):
                if os.path.isdir(dst_dir):
                    shutil.rmtree(dst_dir)
                else:
                    os.remove(dst_dir)
            shutil.move(src_dir, dst_dir)
            print_colored(f"Moved '{src_dir}' to '{dst_dir}'", COLOR['YELLOW'])
        else:
            print_colored(f"Source directory '{src_dir}' does not exist or is not a directory. Nothing to move.", COLOR['YELLOW'])
    except PermissionError:
        print_colored(
            f"Error: Permission denied. You might need to run this script with 'sudo' to move '{src_dir}'.",
            COLOR['YELLOW']
        )
    except Exception as e:
        print_colored(
            f"Warning: Failed to move '{src_dir}' to '{dst_dir}' - {str(e)}",
            COLOR['YELLOW']
        )


def is_vl_model(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if 'architectures' not in config or not config['architectures']:
            return False

        first_architecture = config['architectures'][0].lower()
        return 'vl' in first_architecture
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error while checking VL model: {str(e)}")
        return False


def setup_environment(args) -> None:
    """设置运行环境"""
    os.environ["TORCH_CUDA_ARCH_LIST"] = args.torch_cuda_arch_list
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
    os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"]="http/protobuf"
    
    # AI云当前 NVIDIA H20-3e集群需要删除/usr/local/cuda/compat
    assert torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name == "NVIDIA H20-3e":
        src_dir = "/usr/local/cuda/compat"
        dst_dir = "/tmp/compat_bk"
        mv_cuda_compat(src_dir,dst_dir)

    # 1.0.0rc4 版本 VL 模型 bug
    if args.backend == "pytorch" and is_vl_model(os.path.join(args.model_path,"config.json")):
        print_colored(f"\nIt's VL Model!", COLOR['YELLOW'])
        args.cuda_graph_batch_sizes = list(range(1, args.cuda_graph_max_batch_size + 1))
        args.disable_block_reuse = True

def main():
    """主函数"""
    try:
        args = parse_arguments()
        trtllm_version = get_trtllm_version()
        print_colored(f"\nINFO: TensorRT-LLM Version: {trtllm_version or 'Not available'}", COLOR['GREEN'])
        
        setup_environment(args)
        extra_config = create_extra_llm_config(args,trtllm_version)
        
        try:
            with open("/tmp/extra_llm_api_options.yaml", "w") as f:
                yaml.dump(extra_config, f, default_flow_style=False)
        except IOError as e:
            print_colored(f"\nError writing config file: {str(e)}", COLOR['RED'])
            sys.exit(1)

        if args.backend == "trt":
            command = build_trt_command(args.model_path, args)
        else:
            command = build_torch_command(args.model_path, args)

        log_file = setup_logging(args.model_path, args)
        print_config_table(args)
        
        print_colored(f"\nextra_llm_api_options:", COLOR['YELLOW'])
        print_dict(extra_config, COLOR['YELLOW'])
        
        print_colored(f"\nStarting inference service at {datetime.now()}", COLOR['YELLOW'])
        print_colored(f"\nCommand: {' '.join(command)}", COLOR['YELLOW'])
        print_colored(f"\nLog file: {log_file}\n", COLOR['GREEN'])
        
        return_code = run_command(command, log_file)
        if return_code != 0:
            print_colored(f"\nError: Process exited with code {return_code}", COLOR['RED'])
            sys.exit(return_code)
        
        print_colored("\nInference service completed successfully", COLOR['GREEN'])

    except KeyboardInterrupt:
        print_colored("\nProcess terminated by user", COLOR['YELLOW'])
        kill_trtllm_serve()
        sys.exit(0)
    except Exception as e:
        print_colored(f"\nFatal error: {str(e)}", COLOR['RED'])
        kill_trtllm_serve()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

# vLLM
 
# SGLang

# Huggingface

- Tokenizer

```
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json

model_name = "/mnt/modelops/models/QwQ-32B/"

message=[{"role":"user","content":"你好"}]

tokenizer = AutoTokenizer.from_pretrained(model_name)

print(tokenizer)

text = tokenizer.apply_chat_template(
    message,
    tokenize=False,
    add_generation_prompt = True,
    # continue_final_message = True
)

print(repr(text))
```

- OpenAI chat

```
from openai import OpenAI
import time
import threading
import queue
import statistics
from concurrent.futures import ThreadPoolExecutor
import argparse

class PerformanceTester:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI(
            # api_key="EMPTY",
            # base_url="http://127.0.0.1:9122/v1",
            api_key="3lRflT2tO2EIgrdWS5djWXQavdrYOwlA",

            # AI 云
            # api_key="b2oSmBnWi0oSOlWl7ognzu0XdPXhm0uf",
            base_url="https://antchat.alipay.com/v1",
        )
        self.request_queue = queue.Queue()
        self.latencies = []
        self.success_count = 0
        self.failure_count = 0
        self.total_tokens = 0
        self.start_time = None
        self.lock = threading.Lock()

    def worker(self):
        while True:
            try:
                # Get task from queue with timeout
                task_id = self.request_queue.get(timeout=1)
            except queue.Empty:
                break

            try:
                start_time = time.time()
                
                # Make API call
                completion = self.client.chat.completions.create(
                    model="QWEN3_32B_EAGLE3_TP1",
                    messages=args.text, #[
                    #     {'role': 'user', 'content': self.args.text}
                    # ],
                    temperature=0.0,
                    # stop="<|im_end|>",
                    # max_tokens=32,
                    stream=False,
                    logprobs=False,
                    max_tokens=2048,
                    frequency_penalty=0.0,
                    n=1,
                    
                )

                # Process response
                end_time = time.time()
                latency = end_time - start_time

                content = ""
                message = completion.choices[0].message
                if hasattr(message,"reasoning_content") and message.reasoning_content:
                    content += message.reasoning_content
                if  hasattr(message,"content") and message.content:
                    content += message.content
                
                print(f"{content}")

                tokens = len(content.split())  # Approximate token count

                with self.lock:
                    self.latencies.append(latency)
                    self.success_count += 1
                    self.total_tokens += tokens

            except Exception as e:
                with self.lock:
                    self.failure_count += 1
                print(f"Request failed: {str(e)}")
            finally:
                self.request_queue.task_done()

    def run_test(self):
        print(f"Starting performance test with {self.args.concurrency} concurrent workers...")
        self.start_time = time.time()

        # Fill the queue with tasks
        for i in range(self.args.requests):
            self.request_queue.put(i)

        # Create worker threads
        with ThreadPoolExecutor(max_workers=self.args.concurrency) as executor:
            for _ in range(self.args.concurrency):
                executor.submit(self.worker)

        # Wait for all requests to complete
        self.request_queue.join()
        test_duration = time.time() - self.start_time

        # Calculate metrics
        qps = self.success_count / test_duration
        throughput_tokens = self.total_tokens / test_duration
        
        # Calculate latency statistics
        if self.latencies:
            avg_latency = statistics.mean(self.latencies)
            min_latency = min(self.latencies)
            max_latency = max(self.latencies)
            p95 = statistics.quantiles(self.latencies, n=20)[-1]  # 95th percentile
        else:
            avg_latency = min_latency = max_latency = p95 = 0

        # Print results
        print("\n=== Test Results ===")
        print(f"Total Requests: {self.args.requests}")
        print(f"Successful Requests: {self.success_count}")
        print(f"Failed Requests: {self.failure_count}")
        print(f"Test Duration: {test_duration:.2f} seconds")
        print(f"QPS: {qps:.2f} requests/second")
        print(f"Throughput: {throughput_tokens:.2f} tokens/second")
        print("\nLatency Statistics:")
        print(f"Average: {avg_latency:.4f}s")
        print(f"Minimum: {min_latency:.4f}s")
        print(f"Maximum: {max_latency:.4f}s")
        print(f"95th Percentile: {p95:.4f}s")

def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI API Performance Tester")
    parser.add_argument("--model", type=str, default="qwen3_32b_aibaby",
                       help="Model to test")
    parser.add_argument("--text", type=str, default="",
                       help="Input text for testing")
    parser.add_argument("--requests", type=int, default=32,
                       help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=32,
                       help="Number of concurrent workers")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    while 1:
        print("**************************** 开始测试 ****************************")
        args.text = [{"role":"system","content":""}]
        tester = PerformanceTester(args)
        tester.run_test()
```

# 其他

- curl

```
# 流式

curl http://127.0.0.1:9122/health/

curl http://127.0.0.1:9122/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages":[{"role": "user", "content": "荆轲刺秦"}],
        "stream":false
    }'


curl http://127.0.0.1:9122/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "prompt":"<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "stream":true,
        "max_tokens":20
    }'

# 非流式
curl http://127.0.0.1:9122/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages":[{"role": "user", "content": "你好"}],
        "stream":false,
        "top_k":-1
    }'

$curl http://127.0.0.1:9122/v1/chat/completions     -H "Content-Type: application/json"     -d '{
        "model": "test",
        "messages":[{"role": "user", "content": "你好"}],
        "stream":true,
        "stream_options":{"include_usage":true},
        "max_tokens":10
    }'

```

- 分布式测试

```
import os
import torch
import os
import torch
import torch.distributed as dist
from datetime import timedelta
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim import SGD
from datetime import timedelta
import time

def setup(rank, world_size):

    dist.init_process_group(
        backend="nccl",
        init_method="env://", # 如果使用 torchrun，可以用 env://
        world_size=world_size,
        rank=rank,
        timeout=timedelta(hours=2)  # 设置超长超时，避免因心跳超时被误杀
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def create_large_model():
    """
    创建一个巨大的模型以产生高通信量。
    这里使用一个超大的全连接层来模拟大模型的参数量。
    注意：模型大小需要根据你的GPU显存调整，避免OOM。
    """
    # 尝试一个非常大的模型。如果显存不够，请调小 hidden_dim。
    hidden_dim = 20480  # 20k 维度，参数量约为 20480 * 20480 * 4 bytes ≈ 1.6GB
    model = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
    )
    return model

def train_step(model, optimizer, data):
    """执行一个训练步骤"""
    optimizer.zero_grad()
    output = model(data)
    loss = output.sum()  # 一个简单的损失函数
    loss.backward()
    optimizer.step()
    return loss.item()

def run_test(rank, world_size):
    print(f"[Rank {rank}] 进程启动，开始设置分布式环境...")
    setup(rank, world_size)
    
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    print(f"[Rank {rank}] 分布式环境设置完成。GPU: {local_rank}, World Size: {world_size}")
    
    try:
        # 步骤1: 初始化一个巨大的模型
        print(f"[Rank {rank}] 步骤1: 创建大型模型...")
        model = create_large_model().cuda(local_rank)
        
        # 步骤2: 用 FSDP 包装模型 (这是通信压力最大的地方)
        print(f"[Rank {rank}] 步骤2: 使用 FSDP 包装模型... (将触发大规模 all-gather)")
        model = FSDP(
            model,
            # cpu_offload=CPUOffload(offload_params=True), # 可选：开启CPU卸载以测试更复杂的场景
            device_id=local_rank,
            sync_module_states=True,  # 强制在初始化时同步状态，产生通信
        )
        print(f"[Rank {rank}] FSDP 包装完成。模型已分片。")

        # 步骤3: 创建优化器
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # 步骤4: 准备数据
        dummy_data = torch.randn(1024, 20480, device=local_rank)  # 大批量数据
        
        # 步骤5: 执行多轮训练，持续施加压力
        num_epochs = 5
        steps_per_epoch = 20
        print(f"[Rank {rank}] 步骤3: 开始 {num_epochs} 轮训练，每轮 {steps_per_epoch} 步...")
        
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for step in range(steps_per_epoch):
                loss = train_step(model, optimizer, dummy_data)
                epoch_loss += loss
                
                if step % 5 == 0:
                    print(f"[Rank {rank}] Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
            
            avg_loss = epoch_loss / steps_per_epoch
            print(f"[Rank {rank}] Epoch {epoch} 完成。平均 Loss: {avg_loss:.4f}")
        
        total_time = time.time() - start_time
        print(f"[Rank {rank}] 所有训练步骤完成。总耗时: {total_time:.2f} 秒。")
        
        # 步骤6: 最后的同步，确保所有进程都成功
        dist.barrier()
        print(f"[Rank {rank}] 最终 barrier 通过。压力测试成功完成！🎉")
        
    except Exception as e:
        print(f"[Rank {rank}] 训练过程中发生异常: {e}")
        raise
    finally:
        cleanup()

def main():
    # 从环境变量获取配置 (通常由 torchrun 设置)
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"主函数启动。World Size: {world_size}, Rank: {rank}, Local Rank: {local_rank}")
    
    # 直接运行，适用于单进程启动脚本
    run_test(rank, world_size)

if __name__ == "__main__":
    main()
```
