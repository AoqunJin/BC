import math
from collections.abc import Mapping
from typing import Union, Callable
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer, T5EncoderModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

PRED_HORIZON = 8
OBS_HORIZON = 2
ACTION_HORIZON = 4


def data_to(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, Mapping):
        return {k: data_to(v, device) for k, v in data.items()}
    else:
        return data


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
    
    
def get_resnet(name:str, weights=None, **kwargs):
    func = getattr(models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, obs_horizon, action_dim):
        super().__init__()
        vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(vision_encoder)
        self.text_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small")
        self.tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon,
        )
        self.ema = None
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    
    # Lazy initialization (For Parallel Training)
    def init_ema(self):
        if self.ema is None:
            self.ema = EMAModel(
                parameters=self.noise_pred_net.parameters(),
                power=0.75
            )
    
    def on_train_batch_end(self):
        self.ema.step(self.noise_pred_net.parameters())
    
    def on_inference_start(self):
        self.ema.copy_to(self.noise_pred_net.parameters())

    def forward(self, image, action, agent_pos=None, language=None):
        self.init_ema()
        with torch.no_grad():
            B = image.shape[0]
            image_features = self.vision_encoder(image.flatten(end_dim=1))
            obs_features = image_features.reshape(*image.shape[:2], -1)

            if agent_pos:
                obs_features = torch.cat([obs_features, agent_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)

            text_inputs = self.tokenizer(language, padding=True, return_tensors="pt")
            text_inputs = data_to(text_inputs, obs_features.device)
            
            text_outputs = self.text_model(**text_inputs)
            text_features = torch.sum(text_outputs.last_hidden_state, dim=1)
            text_features = text_features.flatten(start_dim=1)

            obs_cond = torch.cat([obs_cond, text_features], dim=1)
            
            noise = torch.randn_like(action)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=action.device).long()
            noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)

        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

        return noise_pred, noise
    
    def inference(self, image, language):
        B = 1
        image_features = self.vision_encoder(image.flatten(end_dim=1))
        obs_features = image_features.reshape(*image.shape[:2], -1)
        obs_cond = obs_features.flatten(start_dim=1)
        
        text_inputs = self.tokenizer(language, padding=True, return_tensors="pt")
        text_inputs = data_to(text_inputs, obs_features.device)
        
        text_outputs = self.text_model(**text_inputs)        
        text_features = torch.sum(text_outputs.last_hidden_state, dim=1)
        text_features = text_features.flatten(start_dim=1)
        
        obs_cond = torch.cat([obs_cond, text_features], dim=1)
        
        noisy_action = torch.randn((B, PRED_HORIZON, ACTION_HORIZON), device=obs_cond.device)
        self.noise_scheduler.set_timesteps(100)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.noise_pred_net(
                noisy_action, k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action
            ).prev_sample
        
        return noisy_action

    def load_state_dict(self, state_dict):
        self.init_ema()
        self.noise_pred_net.load_state_dict(state_dict["noise_pred_net"])
        self.ema.load_state_dict(state_dict["ema"])
        
    def state_dict(self):
        return {
            "noise_pred_net": self.noise_pred_net.state_dict(),
            "ema": self.ema.state_dict()
        }
    