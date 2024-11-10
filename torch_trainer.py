import os
from typing import Dict, Optional

import tempfile
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from deepspeed.runtime.lr_schedules import WarmupLR
import accelerate
from torch.utils.tensorboard import SummaryWriter


from tools import data_map
from config import ds_config

def train_val_worker(config):
    temp_checkpoint_dir = os.path.join(config["storage_path"], f"{config['method']}_results")
    writer = SummaryWriter(log_dir=temp_checkpoint_dir)

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
        device = None
        raise ValueError("No support w/o deepspeed now")

    target_dtype = torch.float32
    if config["deepspeed"] and len(config["ds_config"]):
        if model.bfloat16_enabled(): target_dtype = torch.bfloat16
        elif model.fp16_enabled(): target_dtype = torch.half

    # Recover
    start_epoch, total_step, total_loss, total_acc, step = 0, 0, 0, 0, 0
    checkpoint = config["checkpoint"]
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
 
                
                client_state_dict = {"step": total_step, "epoch": epoch}
                torch.save(client_state_dict, os.path.join(temp_checkpoint_dir, f"info.pt"))
                
                if config["deepspeed"] and len(config["ds_config"]):
                    model.save_checkpoint(os.path.join(temp_checkpoint_dir))
                else:
                    ...

                torch.distributed.barrier()
                # train.report(metrics, checkpoint=train.Checkpoint.from_directory(temp_checkpoint_dir),)
                for key, value in metrics.items():
                    writer.add_scalar(key, value, epoch)

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
    

def train_torch(config):
    train_val_worker(config)
