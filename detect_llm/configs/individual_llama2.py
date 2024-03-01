import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_llama2'

    config.tokenizer_paths=["/mnt/hdd-nfs/mgubri/models_hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235/"]
    config.model_paths=["/mnt/hdd-nfs/mgubri/models_hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235/"]
    #config.model_kwargs = [{"low_cpu_mem_usage": True, "use_cache": False, 'torch_dtype': torch.float16}] # float16 added later by us
    config.conversation_templates=['llama-2']

    return config