import os

os.sys.path.append("..")
from configs.template import get_config as default_config


def get_config():
    config = default_config()

    config.result_prefix = 'results/individual_guanaco'

    config.tokenizer_paths = [
        "/mnt/hdd-nfs/mgubri/models_hf/models--TheBloke--guanaco-7B-HF/snapshots/293c24105fa15afa127a2ec3905fdc2a0a3a6dac/"]
    config.model_paths = [
        "/mnt/hdd-nfs/mgubri/models_hf/models--TheBloke--guanaco-7B-HF/snapshots/293c24105fa15afa127a2ec3905fdc2a0a3a6dac/"]
    # config.model_kwargs = [{"low_cpu_mem_usage": True, "use_cache": False, 'torch_dtype': torch.float16}] # float16 added later by us
    config.conversation_templates = ['guanaco']

    return config