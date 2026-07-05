import os


def check_tpu():
    import os
    try:
        if ('TPU_ACCELERATOR_TYPE' in os.environ) or ('TPU_WORKER_HOSTNAMES' in os.environ) or ('TPU_PROCESS_ADDRESSES' in os.environ):
            return True
        else:
            return False
    except Exception as e:
        return False
    return False

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


