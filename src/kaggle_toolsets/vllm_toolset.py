

from kaggle_toolsets import run_command, check_tpu


def install_vllm():
    import os
    if check_tpu() == False:
        print("Install vllm for GPU/CPU")
        run_command("pip install uv -q")
        run_command("uv pip install https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl --system  -q")
    else:
        print("Install vllm")
        
        # Cài đặt uv và gỡ các thư viện cũ
        run_command("pip install uv -q")
        run_command("pip uninstall jax jaxlib libtpu torch torchvision torchaudio -y -q")
        run_command("rm -rf vllm")
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




def suggest_vllm_tpu_config(model_path,model_name):

    env = {
        "MODEL_IMPL_TYPE": "vllm",
        "USE_BATCHED_RPA_KERNEL": "1",
        "SKIP_JAX_PRECOMPILE": "0",
        "TPU_MULTIHOST_BACKEND": "ray",
        "RAY_memory_monitor_refresh_ms": "0",
        "VLLM_XLA_CHECK_RECOMPILATION": "0",
        "JAX_PLATFORMS": "tpu,cpu",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"
    }

    start_args = [
        'vllm', 'serve',
        model_path,
        '--tensor-parallel-size', '4',
        '--max-model-len', '3072',
        '--max-num-seqs', '16',
        # 2. Khai báo cứng block-size là 16 (phù hợp với cấp số nhân của TPU)
        '--block-size', '16', 
        # 3. Ép chạy chế độ eager để tránh lỗi biên dịch đồ thị (Graph Compilation) của Ray trên TPU
        '--enforce-eager', 
        '--dtype', 'bfloat16',
        '--kv-cache-dtype', 'bfloat16',
        '--no-enable-prefix-caching',
        '--trust-remote-code',
        '--host', '0.0.0.0',
        '--port', '8000',
        '--served-model-name', model_name,
        '--distributed-executor-backend', 'ray',
        '--max-num-batched-tokens', '16384'
    ]

    return {
        "device": "tpu",
        "env": env,
        "start_args": start_args,
        "note": "TPU default config"
    }
    
def start_vllm_server_score(cmd):
    
    import subprocess
    import threading
    import time

    def start_server():
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def stream_output():
            for line in iter(process.stdout.readline, ""):
                print(f"[SERVER] {line}", end="")

        threading.Thread(target=stream_output, daemon=True).start()
        return process

    def wait_for_server(timeout=1800):
        import requests

        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"http://localhost:8000/health", timeout=5)
                if r.status_code == 200:
                    print(f"\nServer ready after {int(time.time() - start)}s")
                    return True
            except Exception:
                pass

            time.sleep(5)
            elapsed = int(time.time() - start)
            if elapsed > 0 and elapsed % 60 == 0:
                print(f"Waiting for server... {elapsed}s elapsed")

        raise TimeoutError("Server failed to start")

    server_proc = start_server()
    print("Starting vLLM server and waiting for /health ...")
    wait_for_server()

def start_vllm_with_tpu(model_path,model_name,cfg=None):
    import os
    if cfg is None:
        cfg = suggest_vllm_tpu_config(model_path,model_name)
    for k, v in cfg["env"].items():
        os.environ[k] = v
    cmd = cfg["start_args"]
    start_vllm_server_score(cmd)

def suggest_vllm_gpu_config(model_path,model_name,prefer_all_gpus=True, max_tp=None):
    import os
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n == 0:
        return {
            "device": "cpu",
            "env": {
                "CUDA_VISIBLE_DEVICES": "",
            },
            "start_args": [
                "vllm", "serve",
                model_path,
                "--tensor-parallel-size", 1,
                "--max-model-len",'3072',
                "--max-num-seqs", '16',
                "--dtype", "float32",
                "--kv-cache-dtype", "auto",
                "--no-enable-prefix-caching",
                "--trust-remote-code",
                "--host","0.0.0.0",
                "--port","8000",
                "--served-model-name",model_name,
                "--distributed-executor-backend","mp"
            ],
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

    start_args = [
            "vllm", "serve",
            model_path,
            "--tensor-parallel-size", str(use_n),
            "--max-model-len",'3072',
            "--max-num-seqs", '16',
            "--dtype", "float16",
            "--kv-cache-dtype", "float16",
            "--no-enable-prefix-caching",
            "--trust-remote-code",
            "--host","0.0.0.0",
            "--port","8000",
            "--served-model-name",model_name,
            "--distributed-executor-backend","mp",
            "--gpu-memory-utilization", "0.8",
            "--block-size", "16"
        ]

    return {
        "device": "cuda",
        "gpu_count": n,
        "env": env,
        "start_args": start_args,
        "note": "Nếu model nhỏ, TP=1 có thể latency tốt hơn; model lớn/throughput thì TP=so_gpu thường tốt hơn."
    }

def start_vllm_with_gpu(model_path,model_name,cfg=None):
    import os
    if cfg is None:
        cfg = suggest_vllm_gpu_config(model_path,model_name)
    for k, v in cfg["env"].items():
        os.environ[k] = v
    cmd = cfg["start_args"]
    start_vllm_server_score(cmd)

def start_vllm_server(model_path,model_name):
    import os
    if check_tpu():
        start_vllm_with_tpu(model_path,model_name)
    else:
        start_vllm_with_gpu(model_path,model_name)