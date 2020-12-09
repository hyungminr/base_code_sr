import os
import time
import subprocess

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    # print(memory_free_values)
    return memory_free_values

def sec2time(sec, n_msec=0):
    if hasattr(sec,'__len__'): return [sec2time(s) for s in sec]    
    m, s = divmod(sec, 60)    
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if n_msec > 0: pattern = '%%02d:%%02d:%%0%d.%df' % (n_msec+3, n_msec)
    else: pattern = r'%02d:%02d:%02d'
    if d == 0: return pattern % (h, m, s)
    return ('%d days, ' + pattern) % (d, h, m, s)
