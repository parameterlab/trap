import os
#import torch

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_llama2_base'

    config.tokenizer_paths=["/mnt/hdd-nfs/mgubri/models_hf/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"]
    config.model_paths=["/mnt/hdd-nfs/mgubri/models_hf/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"]
    config.conversation_templates=['llama-2']
    #config.model_kwargs = [{"low_cpu_mem_usage": True, "use_cache": False, 'torch_dtype': torch.float16}]

    return config