import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    # config.transfer = True  # we do not use transfer: only 1 prompt and we do not need processive_models
    config.logfile = ""

    #config.progressive_goals = False
    config.stop_on_success = False
    config.num_train_models = 2  # use the first 2 models as train, the rest (0) as test
    config.tokenizer_paths = [
        "/mnt/hdd-nfs/mgubri/models_hf/models--TheBloke--guanaco-7B-HF/snapshots/293c24105fa15afa127a2ec3905fdc2a0a3a6dac/", # "TheBloke/guanaco-7B-HF",
        #"TheBloke/guanaco-13B-HF",
        "/mnt/hdd-nfs/mgubri/models_hf/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6/", # "/DIR/vicuna/vicuna-7b-v1.3",
        #"/DIR/vicuna/vicuna-13b-v1.3"
    ]
    #config.tokenizer_kwargs = [{"use_fast": False}, {"use_fast": False}, {"use_fast": False}, {"use_fast": False}]
    config.tokenizer_kwargs = [{"use_fast": False}, {"use_fast": False}]
    config.model_paths = [
        "/mnt/hdd-nfs/mgubri/models_hf/models--TheBloke--guanaco-7B-HF/snapshots/293c24105fa15afa127a2ec3905fdc2a0a3a6dac/",
        #"TheBloke/guanaco-13B-HF",
        "/mnt/hdd-nfs/mgubri/models_hf/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6/",
        #"/DIR/vicuna/vicuna-13b-v1.3"
    ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False},
        {"low_cpu_mem_usage": True, "use_cache": False},
        #{"low_cpu_mem_usage": True, "use_cache": False},
        #{"low_cpu_mem_usage": True, "use_cache": False}
    ]
    #config.conversation_templates = ["guanaco", "guanaco", "vicuna", "vicuna"]
    config.conversation_templates = ["guanaco", "vicuna"]
    #config.devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    config.devices = ["cuda:0", "cuda:1"]

    return config
