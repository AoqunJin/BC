def get_BCConfig():
    return {
        "batch_size":                  8,
        "gradient_accumulation_steps": 1,
        "num_workers":                 16,
        "lr":                          2e-5,
        "total_epoch":                 1000,
        "total_step":                  100000,
        "steps_per_eval":              5000,
        "num_to_keep":                 1,
        "checkpoint_score_attribute":  "val_acc",
        "checkpoint_score_order":      "max",
        "allow_tf32":                  False,
        "deepspeed":                   True,
        "ds_config":                   get_DSZeRO2Config(),
        "tune":                        True,
        "tune_config":                 {"metric": "val_acc", "mode": "max"},
        "data_path":                   "/path/to/hdf5",
        "zero_tasks":                  None,
        # "zero_tasks":                  [
        #     '_button-press-topdown-wall-v2-goal-observable_', 
        #     '_button-press-v2-goal-observable_', 
        #     '_reach-wall-v2-goal-observable_', 
        #     '_push-v2-goal-observable_', 
        #     '_pick-place-wall-v2-goal-observable_',
        #     '_disassemble-v2-goal-observable_',
        #     '_door-open-v2-goal-observable_',
        #     '_door-unlock-v2-goal-observable_',
        #     '_drawer-close-v2-goal-observable_',
        #     '_faucet-close-v2-goal-observable_',
        #     '_plate-slide-back-v2-goal-observable_',
        #     '_plate-slide-side-v2-goal-observable_', 
        #     '_window-close-v2-goal-observable_'
        # ],
        # "zero_tasks":                  [
        #     '_button-press-topdown-v2-goal-observable_',
        #     '_button-press-wall-v2-goal-observable_', 
        #     '_reach-v2-goal-observable_', 
        #     '_push-wall-v2-goal-observable_',
        #     '_pick-place-v2-goal-observable_',
        #     '_assembly-v2-goal-observable_', 
        #     '_door-close-v2-goal-observable_', 
        #     '_door-lock-v2-goal-observable_', 
        #     '_drawer-open-v2-goal-observable_',
        #     '_faucet-open-v2-goal-observable_', 
        #     '_plate-slide-v2-goal-observable_', 
        #     '_plate-slide-back-side-v2-goal-observable_',
        #     '_window-open-v2-goal-observable_', 
        # ],
        "seq_len":                     12,
        "frame_skip":                  0,
        "use_language":                False,
        "storage_path":                "/path/to/output",
        "processer":                   "basic",
        "checkpoint":                  None,
        "restore":                     False,
    }
    
    
def get_LCBCConfig():
    config = get_BCConfig()
    config.update({
        "method":                      "lcbc",
    })
    # tune_config(config)
    return config


def get_RT1Config():
    config = get_BCConfig()
    config.update({
        "method":                      "rt1",
        "batch_size":                  2,
    })
    # tune_config(config)
    return config


def get_DiffusionConfig():
    config = get_BCConfig()
    config.update({
        "method":                      "diffusion",    
    })
    # tune_config(config)
    return config


def get_InverseConfig():
    config = get_BCConfig()
    config.update({
        "method":                      "inverse",
        "processer":                   "canny",
        "vision_model":                "vit",  # [vit | resnet]
        "batch_size":                  8,
        "seq_len":                     12,
        "num_workers":                 32,
        "lr":                          2e-5,
    })
    # tune_config(config)
    return config


def get_DSConfig():
    deepspeed_config = {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},  # Turn this on if using AMPERE GPUs.
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 16,        
        "gradient_clipping": 1.0,
        "steps_per_print": 2000000
    }
    return deepspeed_config


def get_DSZeRO2Config():
    deepspeed_config = {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},  # Turn this on if using AMPERE GPUs.
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 16,        
        "gradient_clipping": 1.0,
        "steps_per_print": 2000000
    }
    return deepspeed_config


def ds_config(config):
    if config["deepspeed"] and len(config["ds_config"]):
        config["ds_config"]["train_micro_batch_size_per_gpu"] = config["batch_size"]
        config["ds_config"]["gradient_accumulation_steps"] = config["gradient_accumulation_steps"]

    
def tune_config(config):
    if config["tune"]:
        from ray import tune
        config["lr"] = tune.loguniform(1e-5, 2e-4)
        # config["batch_size"] = tune.choice([4, 8, 16])
        # config["seq_len"] = tune.choice([8, 12, 16])
