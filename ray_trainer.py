import os
from typing import Dict, Optional

import tempfile
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from deepspeed.runtime.lr_schedules import WarmupLR
import ray
from ray import train, tune
from ray.train import Checkpoint, RunConfig, FailureConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from ray.tune.schedulers import ASHAScheduler
import accelerate

from tools import data_map
from config import ds_config

def train_val_worker(config):
    # TODO 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if config["allow_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True        
    
    if config["deepspeed"] and len(config["ds_config"]):
        ds_config(config)
    
    # TODO Test move_to_device in deepspeed
    move_to_device = not (config["deepspeed"] and len(config["ds_config"]))
    
    train_loader, val_loader, zero_loader = config["get_dataloader"](**config)
    train_loader = train.torch.prepare_data_loader(train_loader, move_to_device=move_to_device)
    val_loader = train.torch.prepare_data_loader(val_loader, move_to_device=move_to_device)
    
    if zero_loader:
        zero_loader = train.torch.prepare_data_loader(zero_loader, move_to_device=move_to_device)
        
    model = config["get_model"](**config)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    lr_scheduler = WarmupLR(optimizer, warmup_max_lr=config["lr"])
    forward_fn = config["forward_fn"]
  
    if config["deepspeed"] and len(config["ds_config"]):
        import deepspeed
        from deepspeed.accelerator import get_accelerator
        # Initialize DeepSpeed Engine
        model, optimizer, _dataloader, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=model.parameters(),
            lr_scheduler=lr_scheduler,
            config=config["ds_config"],
        )
        device = get_accelerator().device_name(model.local_rank)
    else:
        model = train.torch.prepare_model(model)
        device = None

    target_dtype = torch.float32
    if config["deepspeed"] and len(config["ds_config"]):
        if model.bfloat16_enabled(): target_dtype = torch.bfloat16
        elif model.fp16_enabled(): target_dtype = torch.half

    # Recover
    start_epoch, total_step, total_loss, total_acc, step = 0, 0, 0, 0, 0
    checkpoint: Optional[Checkpoint] = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            if config["deepspeed"] and len(config["ds_config"]):
                model.load_checkpoint(checkpoint_dir)
            else:
                model_path = os.path.join(checkpoint_dir, "model.pt")
                if os.path.exists(model_path): model.load_state_dict(torch.load(model_path))
                optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
                
                if os.path.exists(optimizer_path): optimizer.load_state_dict(torch.load(optimizer_path))
                lr_scheduler_path = os.path.join(checkpoint_dir, "lr_scheduler.pt")
                lr_scheduler.load_state_dict(torch.load(lr_scheduler_path))
            
            client_state_dict = None
            info_path = os.path.join(checkpoint_dir, "info.pt")
            if os.path.exists(info_path): client_state_dict = torch.load(info_path)
            
            if client_state_dict:
                start_epoch = client_state_dict["epoch"] + 1
                total_step = client_state_dict["step"]
                
            print("Loading checkpoint from", checkpoint_dir, "Epoch", start_epoch, "Step", total_step)
            # TODO Resume from stop step
            # accelerate.data_loader.skip_first_batches(
            #     dataloader=train_loader, num_batches=(total_step-1) % len(train_loader))
        
    for epoch in range(start_epoch, config["total_epoch"]):
        model.train()
        for data in train_loader:
            if device != None: data = data_map(data, device)
            if target_dtype != None: data = data_map(data, target_dtype)

            outputs = forward_fn(model, **data)
            loss = outputs["loss"]
            acc = outputs["acc"]
            
            if config["deepspeed"] and len(config["ds_config"]):
                model.backward(loss)
            else:
                loss.backward()
                
            optimizer.step()
            optimizer.zero_grad()
            
            if lr_scheduler: lr_scheduler.step()

            model_tmp = model.module if hasattr(model, 'module') else model
            if hasattr(model_tmp, "on_train_batch_end"):
                model_tmp.on_train_batch_end()
                
            total_acc += acc.item()
            total_loss += loss.item()
            total_step += 1
            step += 1
        
            if total_step % config["steps_per_eval"] == 0:
                metrics = {"loss_train": total_loss / step, "acc_train": total_acc / step, 
                           "epoch": epoch, "step": total_step, "lr": lr_scheduler.get_last_lr()[0]}
                
                if val_loader: 
                    val_metrics = validation(val_loader, target_dtype, device, forward_fn, model, epoch, type="val")
                    metrics.update(val_metrics)
                    
                if zero_loader: 
                    zero_metrics = validation(zero_loader, target_dtype, device, forward_fn, model, epoch, type="zero")
                    metrics.update(zero_metrics)
 
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    client_state_dict = {"step": total_step, "epoch": epoch}
                    torch.save(client_state_dict, os.path.join(temp_checkpoint_dir, f"info.pt"))
                    
                    if config["deepspeed"] and len(config["ds_config"]):
                        model.save_checkpoint(os.path.join(temp_checkpoint_dir))
                    else:
                        rank = train.get_context().get_world_rank()
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(temp_checkpoint_dir, f"model.pt"))
                        torch.save(optimizer.state_dict(), os.path.join(temp_checkpoint_dir, f"optimizer.pt"))
                        torch.save(lr_scheduler.state_dict(), os.path.join(temp_checkpoint_dir, f"lr_scheduler.pt"))

                    torch.distributed.barrier()
                    train.report(metrics, checkpoint=train.Checkpoint.from_directory(temp_checkpoint_dir),)
                step, total_loss, total_acc = 0, 0, 0
            if total_step >= config["total_step"]:
                break


def validation(val_loader, target_dtype, device, forward_fn, model, epoch, type):
    model.eval()
    with torch.no_grad():
        val_step, val_loss, val_acc = 0, 0, 0
        for data in val_loader:
            if device != None: data = data_map(data, device)
            if target_dtype != None: data = data_map(data, target_dtype)

            outputs = forward_fn(model, **data)
            val_loss += outputs["loss"].item()
            val_acc += outputs["acc"].item()
            val_step += 1
        metrics = {f"{type}_loss": val_loss / val_step, f"{type}_acc": val_acc / val_step}  
    model.train()
    return metrics
    

def train_ray(config):
    context = ray.init(address='auto')
    print(context.dashboard_url)
    
    if config["restore"]:
        experiment_path = os.path.join(config["storage_path"], f"{config['method']}_results")
        if TorchTrainer.can_restore(experiment_path):
            trainer = TorchTrainer.restore(experiment_path)
    else:
        checkpoint = None
        if config["checkpoint"]: 
            checkpoint = Checkpoint(config["checkpoint"])
            
        trainer = TorchTrainer(
            train_val_worker,
            train_loop_config=config,
            scaling_config=ScalingConfig(
                num_workers=1, use_gpu=True,
                resources_per_worker={"GPU": 1}
            ),
            run_config=RunConfig(
                name=f"{config['method']}_results", 
                storage_path=os.path.expanduser(config["storage_path"]),
                failure_config=FailureConfig(max_failures=10),  # -1 will always
                stop=None,  # {"training_iteration": 10, "mean_accuracy": 0.8}
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=config["num_to_keep"],
                    checkpoint_score_attribute=config["checkpoint_score_attribute"],
                    checkpoint_score_order=config["checkpoint_score_order"],
                ),
            ),
            resume_from_checkpoint=checkpoint,
        )
    result = trainer.fit()


def tune_dist_ray(config):
    context = ray.init(address='auto')
    print(context.dashboard_url)
    
    if config["restore"]:
        experiment_path = os.path.join(config["storage_path"], f"{config['method']}_results")
        if tune.Tuner.can_restore(experiment_path):
            tuner = tune.Tuner.restore(experiment_path, trainable=train_val_worker, resume_errored=True)
    else:
        trainer = TorchTrainer(
            train_val_worker,
            scaling_config=ScalingConfig(
                num_workers=1, use_gpu=True,
                resources_per_worker={"GPU": 1}
            ),
            run_config=RunConfig(
                name=f"{config['method']}_results", 
                storage_path=os.path.expanduser(config["storage_path"]),
                failure_config=FailureConfig(max_failures=10),  # -1 will always
                stop=None,  # {"training_iteration": 10, "mean_accuracy": 0.8}
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=config["num_to_keep"],
                    checkpoint_score_attribute=config["checkpoint_score_attribute"],
                    checkpoint_score_order=config["checkpoint_score_order"],
                ),
            ),
        )
        
        scheduler = ASHAScheduler(
            time_attr='step',
            metric=config["tune_config"]["metric"],
            mode=config["tune_config"]["mode"],
            max_t=config["total_step"],
            grace_period=5,
            reduction_factor=4,
        )
        tuner = tune.Tuner(
            trainer,
            param_space={"train_loop_config": config},
            tune_config=tune.TuneConfig(
                scheduler=scheduler,
                num_samples=20,
                max_concurrent_trials=2
            ),
        )
    
    results = tuner.fit()
    