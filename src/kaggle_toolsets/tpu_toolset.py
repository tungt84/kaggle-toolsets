def suggest_vllm_tpu_config(model_path,model_name):

    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(use_n)),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "NCCL_DEBUG": "WARN",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
    

def start_vllm_with_tpu(model_path,model_name,cfg=None):
    import os
    if cfg is None:
        cfg = suggest_vllm_tpu_config(model_path,model_name)
    for k, v in cfg["env"].items():
        os.environ[k] = v
    cmd = cfg["start_args"]

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