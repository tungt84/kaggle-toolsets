
def check_dataset(tag,dataset_name,default_path=None):
    import os
    dataset_path = f"/kaggle/input/{dataset_name}"
    dataset_path_with_tag = f"/kaggle/input/datasetst/{tag}/{dataset_name}"
    if os.path.exists(dataset_path_with_tag):
        return dataset_path_with_tag
    elif os.path.exists(dataset_path):
        return dataset_path
    else:
        return default_path
    

def check_notebook(tag,notebook_name,default_path=None):
    import os
    path_with_tag = f'/kaggle/input/notebooks/{tag}/{notebook_name}'
    path = f'/kaggle/input/{notebook_name}'
    if os.path.exists(path_with_tag):
        return path_with_tag
    elif os.path.exists(path):
       return path
    return default_path

def run_command(command):
    import subprocess
    """Hàm hỗ trợ chạy lệnh shell"""
    subprocess.run(command, shell=True, check=True)


def install_vllm(is_tpu = False):
    import os
    if is_tpu == False:
        print("Install vllm for GPU/CPU")
        run_command("pip install uv -q")
        run_command("uv pip install https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl --system  -q")
    else:
        print("Install vllm")
        
        # Cài đặt uv và gỡ các thư viện cũ
        run_command("pip install uv -q")
        run_command("pip uninstall jax jaxlib libtpu torch torchvision torchaudio -y -q")
        
        # Clone repository
        run_command("git clone --branch releases/v0.19.0 https://github.com/vllm-project/vllm.git")
        
        # Cài đặt qua uv trong thư mục vllm
        run_command("cd vllm && uv pip install -r requirements/tpu.txt --system -q")
        run_command('cd vllm && VLLM_TARGET_DEVICE="tpu" uv pip install -e . --system -q')
        run_command("cd vllm && uv pip install git+https://github.com/jiejiezi0v0/tpu-inference.git --system -U --pre -q")

        run_command('uv pip install --upgrade "datasets>=2.20.0" --system -q')

        run_command('uv pip uninstall -y huggingface-hub --system -q')
        run_command('uv pip install "huggingface-hub>=0.34.0,<1.0" --system -q')
        run_command('uv pip uninstall -y numpy --system -q')
        run_command('uv pip install "numpy<=2.3" --system -q')

        run_command('pip uninstall -y prometheus-fastapi-instrumentator -q')
        run_command('pip install -U prometheus-fastapi-instrumentator -q')

        run_command('cp /usr/local/lib/python3.12/site-packages/prometheus_fastapi_instrumentator/routing.py /usr/local/lib/python3.12/site-packages/prometheus_fastapi_instrumentator/routing.py.bak')

    
        from pathlib import Path
        p = Path("/usr/local/lib/python3.12/site-packages/prometheus_fastapi_instrumentator/routing.py")
        bak = p.with_suffix(".py.bak")
        if not bak.exists():
            p.replace(bak)  # move original to backup
            bak = p.with_suffix(".py.bak")
        else:
            # nếu đã có backup, giữ nguyên và load file gốc từ backup
            pass
        s = bak.read_text()
        
        old = "route_name = route.path"
        new = (
            "route_name = getattr(route, 'path', None)\n"
            "            if route_name is None:\n"
            "                route_name = getattr(route, 'prefix', None)\n"
            "            if route_name is None:\n"
            "                continue"
        )
        if old not in s:
            print('Không tìm thấy chuỗi cần thay thế; kiểm tra file thủ công.')
        else:
            s = s.replace(old, new)
            p.write_text(s)
            print('Đã patch', p)

def suggest_vllm_gpu_config(prefer_all_gpus=True, max_tp=None):
    import os
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n == 0:
        return {
            "device": "cpu",
            "env": {
                "CUDA_VISIBLE_DEVICES": "",
            },
            "start_args": {
                "tensor_parallel_size": 1,
                "distributed_executor_backend": "mp",
                "dtype": "float32",
                "kv_cache_dtype": "auto",
            },
            "note": "Không có GPU. vLLM trên CPU chạy được nhưng chậm."
        }

    use_n = n if prefer_all_gpus else 1
    if max_tp is not None:
        use_n = min(use_n, max_tp)

    # T4 nên dùng fp16 thay vì bf16
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(use_n)),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "NCCL_DEBUG": "WARN",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    start_args = {
        "tensor_parallel_size": use_n,
        "distributed_executor_backend": "mp",
        "dtype": "float16",
        "kv_cache_dtype": "float16",
    }

    return {
        "device": "cuda",
        "gpu_count": n,
        "env": env,
        "start_args": start_args,
        "note": "Nếu model nhỏ, TP=1 có thể latency tốt hơn; model lớn/throughput thì TP=so_gpu thường tốt hơn."
    }

def start_vllm_server(model_path,model_name,is_tpu= False, tensor_parallel_size=1,max_model_len=3072, max_num_seqs=32, kv_cache_dtype='bfloat16', dtype='bfloat16', host='0.0.0.0', port=8000, distributed_executor_backend='ray', data_parallel_size=1, timeout=1800,max_num_batched_tokens=16384):
    import subprocess
    import threading
    import os
    import time

    effective_tensor_parallel_size = tensor_parallel_size
    effective_distributed_executor_backend = distributed_executor_backend
    effective_dtype = dtype
    effective_kv_cache_dtype = kv_cache_dtype

    if is_tpu:
        print("Use TPU to validate")
        os.environ['MODEL_IMPL_TYPE'] = 'vllm'
        os.environ['USE_BATCHED_RPA_KERNEL'] = '1'
        os.environ['SKIP_JAX_PRECOMPILE'] = '0'
        os.environ['TPU_MULTIHOST_BACKEND'] = 'ray'
        os.environ['RAY_memory_monitor_refresh_ms'] = '0'
        os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'
        os.environ['JAX_PLATFORMS'] = 'tpu,cpu'
        os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    else:
        print("Use GPU/CPU to validate")
        cfg = suggest_vllm_gpu_config(prefer_all_gpus=True)
        for k, v in cfg["env"].items():
            os.environ[k] = v
        effective_tensor_parallel_size = cfg["start_args"]["tensor_parallel_size"]
        effective_distributed_executor_backend = cfg["start_args"]["distributed_executor_backend"]
        effective_dtype = cfg["start_args"]["dtype"]
        effective_kv_cache_dtype = cfg["start_args"]["kv_cache_dtype"]

    

    
    

    def start_server(is_tpu = False,tensor_parallel_size=1,max_model_len=3072, max_num_seqs=32, kv_cache_dtype='bfloat16', dtype='bfloat16', port=8000, served_model_name=model_name, distributed_executor_backend='ray', data_parallel_size=1,max_num_batched_tokens=16384):
        cmd = [
            'vllm', 'serve',
            model_path,
            '--tensor-parallel-size', str(tensor_parallel_size),
            '--max-model-len', str(max_model_len),
            '--max-num-seqs', str(max_num_seqs),
            '--dtype', dtype,
            '--kv-cache-dtype', kv_cache_dtype,
            '--enable-chunked-prefill',
            '--no-enable-prefix-caching',
            '--trust-remote-code',
            '--host', '0.0.0.0',
            '--port', str(port),
            '--served-model-name', served_model_name,
            '--distributed-executor-backend', distributed_executor_backend,
        ]
        #Nguyên nhân sâu xa là do các GPU T4 trên Kaggle kết nối với nhau qua giao thức PCIe (không có kết nối tốc độ cao NVLink), điều này khiến việc chụp đồ thị CUDA (CUDA Graph Capture) kết hợp với các phép toán chia sẻ bộ nhớ của NCCL/vLLM bị lỗi invalid argument
        
        if is_tpu == False:
            cmd.append("--enforce-eager")
        
        if data_parallel_size and data_parallel_size > 1:
            cmd.extend(['--data-parallel-size', str(data_parallel_size)])

        if max_num_batched_tokens and max_num_batched_tokens > 0:
            cmd.extend(['--max-num-batched-tokens', str(max_num_batched_tokens)])

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        def stream_output():
            for line in iter(process.stdout.readline, ''):
                print(f"[SERVER] {line}", end='')

        threading.Thread(target=stream_output, daemon=True).start()
        return process

    def wait_for_server(host="localhost", port=8000, timeout=1800):
        """Poll health endpoint until server is ready."""
        import requests
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"http://{host}:{port}/health", timeout=5)
                if r.status_code == 200:
                    print(f"\n✓ Server ready after {int(time.time()-start)}s")
                    return True
            except Exception:
                pass
            time.sleep(5)
            if int(time.time()-start) % 60 == 0:
                print(f"Waiting for server... {int(time.time()-start)}s elapsed")
        raise TimeoutError("Server failed to start")

    server_proc = start_server(
        tensor_parallel_size=effective_tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        kv_cache_dtype=effective_kv_cache_dtype,
        dtype=effective_dtype,
        port=port,
        served_model_name=model_name,
        distributed_executor_backend=effective_distributed_executor_backend,
        data_parallel_size=data_parallel_size,
        is_tpu=is_tpu,
        max_num_batched_tokens=max_num_batched_tokens
    )
    print("Starting vLLM server and waiting for /health ...")
    wait_for_server(host="localhost", port=port, timeout=timeout)
    return server_proc
