import subprocess
import re

def get_gpu_memory_info():
    try:
        # 执行nvidia-smi命令并获取输出
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,nounits,noheader'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 检查是否有错误输出
        if result.stderr:
            raise Exception(f"Error in running nvidia-smi: {result.stderr}")
        
        # 解析输出
        lines = result.stdout.strip().split('\n')
        memory_info = []
        for line in lines:
            total_memory, used_memory = map(int, line.split(', '))
            free_memory = total_memory - used_memory
            memory_info.append({'total_memory': total_memory, 'used_memory': used_memory, 'free_memory': free_memory})
        
        return memory_info

    except FileNotFoundError:
        raise Exception("nvidia-smi not found. Ensure that the NVIDIA drivers are installed and nvidia-smi is in your PATH.")

# 获取显卡显存信息
gpu_memory_info = get_gpu_memory_info()

# 打印每张显卡的显存信息
for idx, info in enumerate(gpu_memory_info):
    print(f"GPU {idx}: Total Memory: {info['total_memory']} MB, Used Memory: {info['used_memory']} MB, Free Memory: {info['free_memory']} MB")
