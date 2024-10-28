import numpy as np
import copy
from PIL import Image
import gymnasium
from gymnasium.core import Env
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from tqdm import tqdm
import cv2
import torch
import os
import json
from tools import processer_basic, processer_canny, discrete_to_continuous
import os
os.environ["MUJOCO_GL"] = "egl"

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.25,
    "azimuth": 145,  # rotates the camera around the up vector
    "elevation": -25.0,  # rotates the camera around the right vector
    "lookat": np.array([0.0, 0.65, 0.0]),
    }

DEFAULT_SIZE=512

class CameraWrapper(gymnasium.Wrapper):
    def __init__(self, env: Env, seed:int):
        super().__init__(env)

        self.unwrapped.model.vis.global_.offwidth = DEFAULT_SIZE
        self.unwrapped.model.vis.global_.offheight = DEFAULT_SIZE
        self.unwrapped.mujoco_renderer = MujocoRenderer(env.model, env.data, DEFAULT_CAMERA_CONFIG, DEFAULT_SIZE, DEFAULT_SIZE)

        # Hack: enable random reset
        self.unwrapped._freeze_rand_vec = False
        self.unwrapped.seed(seed)

    def reset(self):
        obs, info = super().reset()
        
        return obs, info

    def step(self, action):
        next_obs, reward, done, truncate, info = self.env.step(action) 
        
        return next_obs, reward, done, truncate, info

def setup_metaworld_env(task_name:str, seed:int):
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
    
    env = CameraWrapper(env_cls(render_mode="rgb_array"), seed)
    
    return env

env_names = {
    # 'button-press-topdown-v2-goal-observable': "",     
    # 'button-press-wall-v2-goal-observable': "", 
    # 'reach-v2-goal-observable': "", 
    # 'push-wall-v2-goal-observable': "",
    # 'pick-place-v2-goal-observable': "",
    # 'assembly-v2-goal-observable': "", 
    # 'door-close-v2-goal-observable': "", 
    # 'door-lock-v2-goal-observable': "", 
    # 'drawer-open-v2-goal-observable': "",
    # 'faucet-open-v2-goal-observable': "", 
    # 'plate-slide-v2-goal-observable': "", 
    # 'plate-slide-back-side-v2-goal-observable': "",
    # 'window-open-v2-goal-observable': "", 
    # # Zero-Shot
    # 'button-press-topdown-wall-v2-goal-observable': "", 
    # 'button-press-v2-goal-observable': "", 
    'reach-wall-v2-goal-observable': "", 
    # 'push-v2-goal-observable': "", 
    # 'pick-place-wall-v2-goal-observable': "",
    # 'disassemble-v2-goal-observable': "",
    # 'door-open-v2-goal-observable': "",
    # 'door-unlock-v2-goal-observable': "",
    # 'drawer-close-v2-goal-observable': "",
    # 'faucet-close-v2-goal-observable': "",
    # 'plate-slide-back-v2-goal-observable': "",
    # 'plate-slide-side-v2-goal-observable': "", 
    # 'window-close-v2-goal-observable': ""
}


def val(model, inference, device, processor, write_video=False, 
        video_path="video", results_path="results", 
        max_path_length=500, test_n=5):
    os.makedirs(video_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    model.to(device)
    
    s_acc = {}
    t_reward = {}
    for env_n in tqdm(env_names.keys()):
        env = setup_metaworld_env(env_n, 10)
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.max_path_length = max_path_length

        s_time = 0; reward = 0
        for i in range(test_n):
            obs = env.reset()
            if write_video: video_list = []
            for j in range(max_path_length):
                obs_rgb = copy.deepcopy(env.render())
                if write_video: video_list.append(obs_rgb)
                # print("Obs shape:", obs_rgb.shape)
                obs_rgb = processor(obs_rgb).to(device).to(torch.half)
            
                with torch.no_grad():
                    clear_buffer = (j == 0)
                    act_e = inference(model, obs_rgb, [env_names[env_n]], clear_buffer=clear_buffer)
                    act = discrete_to_continuous(act_e)
                    # print(act, act_e)
                
                next_obs, rew, done, truncate, info = env.step(act)
                reward += rew
                if info['success'] or done:
                    s_time += 1
                    break
                # break
            if write_video: 
                writer = video_writer(f"{env_n}-{i}-{bool(info['success'] or done)}", 30, (512, 512), video_path)
                for o in video_list:
                    writer.write(o)
                writer.release()
        s_acc[env_n] = s_time / test_n if s_time else s_time
        t_reward[env_n] = reward
    
    # 保存为txt文件
    with open(os.path.join(results_path, "s_acc.txt"), 'w') as f:
        json.dump(s_acc, f, indent=4)
    # with open(os.path.join(results_path, "t_reward.txt"), 'w') as f:
    #     json.dump(t_reward, f, indent=4)

    return s_acc, t_reward

def video_writer(tag, fps, res, video_path):
    return cv2.VideoWriter(
        os.path.join(video_path, f"{tag}.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, res
    )

def load_pytorch_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()


def main(val_model = 'lcbc', checkpoint = ''):
    if val_model == 'lcbc':
        from models.lcbc import (
            get_model, inference
        )
    elif val_model == 'rt1':
        from models.rt1 import (
            get_model, inference
        )    
    elif val_model == 'diffusion':
        from models.diffusion import (
            get_model, inference
        )
    
    processer = processer_basic()
    model = get_model()
    load_pytorch_model(model, checkpoint)
    model.to(torch.half)
    
    if hasattr(model, "on_inference_start"):
        model.on_inference_start()
        
    device = torch.device("cuda")
    val(model, inference, device=device, processor=processer, 
        write_video=True, video_path=f"./validation-gen/{val_model}/video", results_path=f"./validation/{val_model}/results", test_n=20)
    
    
if __name__ == "__main__":
    main("rt1", "/home/ao/workspace/BC/outputs/rt1_results/TorchTrainer_c0984_00001_1_lr=0.0001_2024-10-17_01-04-49/checkpoint_000017/model.pt")
    
    # /home/ao/workspace/BC/outputs/rt1_results_base/TorchTrainer_776ff_00000_0_batch_size=2,lr=0.0001,seq_len=12_2024-10-12_02-48-55/checkpoint_000006/model.pt
    # /home/ao/workspace/BC/outputs/rt1_results/TorchTrainer_c0984_00001_1_lr=0.0001_2024-10-17_01-04-49/checkpoint_000017/model.pt
    
    # CUDA_VISIBLE_DEVICES="0" MUJOCO_GL="egl" python eval.py