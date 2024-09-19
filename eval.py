import metaworld.envs.mujoco.env_dict as _env_dict
from tqdm import tqdm
import cv2
import torch

import os
import json
from tools import processer_basic, processer_canny


env_names = {
    # Zero-Shot
    "button-press-topdown-wall-v2": "Environment Description: This is a computer-generated simulated environment featuring a robotic arm with a gripper attached to its end, a table, and several objects in the environment. The robotic arm is red, and the two fingers of the gripper are white and blue respectively. The table is made of brown wood, with gray vertical barriers added to its edges. There are shadows for all objects in the environment. The simulated environment looks very realistic and adheres to physical laws. In front of the arm, there is a button vertically connected to a device, which is placed vertically on the table surface. The bottom of the button is black, and the top is red. The bottom of the device is black, and the top is yellow.  Task Description: The arm moves upward, reaching a position higher than the button. Next, it moves forward, positioning itself above the button. The gripper then closes. And finally, the arm moves downward to press the button.",
    "coffee-button-v2": "Environment Description: This is a computer-generated simulated environment featuring a robotic arm with a gripper attached to its end, a table, and several objects in the environment. The robotic arm is red, and the two fingers of the gripper are white and blue respectively. The table is made of brown wood, with gray vertical barriers added to its edges. There are shadows for all objects in the environment. The simulated environment looks very realistic and adheres to physical laws. In front of the arm, there is a coffee machine placed vertically on the table surface. It has a button directly in front, facing the arm. Between the coffee machine and the arm, there is a cup. The coffee machine is red, with a black base and handle. The button on it is white, and the cup is white.  Task Description: The arm moves upward to a horizontal position directly facing the button, then the arm moves forward, crossing over the cup, and press the button.",
    "plate-slide-v2": "Environment Description: This is a computer-generated simulated environment featuring a robotic arm with a gripper attached to its end, a table, and several objects in the environment. The robotic arm is red, and the two fingers of the gripper are white and blue respectively. The table is made of brown wood, with gray vertical barriers added to its edges. There are shadows for all objects in the environment. The simulated environment looks very realistic and adheres to physical laws. There is a soccer goal vertically placed on the table in front of the arm, directly facing it. There is a disc under the arm. The goal is red with a white net, and the disc is black.  Task Description: The arm moves down to press the disc, and finally moves forward to push the disc to the red point.",
    "faucet-open-v2": "Environment Description: This is a computer-generated simulated environment featuring a robotic arm with a gripper attached to its end, a table, and several objects in the environment. The robotic arm is red, and the two fingers of the gripper are white and blue respectively. The table is made of brown wood, with gray vertical barriers added to its edges. There are shadows for all objects in the environment. The simulated environment looks very realistic and adheres to physical laws. In front of the arm, there is a faucet placed vertically on the table surface. The upper half of the faucet is red, and the lower half is gray.  Task Description: The arm moves downward to a horizontal position directly facing the faucet. The arm moves forward and to the right, reaching the right side of the faucet. The gripper then closes. Next, it moves to the left, turning the faucet towards the left side.",
    "push-wall-v2": "Environment Description: This is a computer-generated simulated environment featuring a robotic arm with a gripper attached to its end, a table, and several objects in the environment. The robotic arm is red, and the two fingers of the gripper are white and blue respectively. The table is made of brown wood, with gray vertical barriers added to its edges. There are shadows for all objects in the environment. The simulated environment looks very realistic and adheres to physical laws. There is a red cylinder on the table. Near the cylinder, there is a wall, which is reddish-brown in color.  Task Description: 1. Grip Red Cylinder. 2. Go green point.",
    # Training
    # "button-press-v2": "Environment Description: This is a computer-generated simulated environment featuring a robotic arm with a gripper attached to its end, a table, and several objects in the environment. The robotic arm is red, and the two fingers of the gripper are white and blue respectively. The table is made of brown wood, with gray vertical barriers added to its edges. There are shadows for all objects in the environment. The simulated environment looks very realistic and adheres to physical laws. In front of the arm, there is a button horizontally connected to a device, which is also placed lying down on the table surface, facing the arm directly. The bottom of the button is black, and the top is red. The bottom of the device is black, and the top is yellow.  Task Description: The arm moves downward to a horizontal position directly facing the button. Next, it moves forward and to the up, then moves forward and to the down to press the button.",
    # "faucet-close-v2": "The video begins with a computer-generated simulated environment featuring a red robotic arm with a gripper attached to its end. The arm is positioned above a table made of brown wood, with gray vertical barriers added to its edges. On the table, there are several objects, including a faucet placed vertically. The upper half of the faucet is red, and the lower half is gray. The robotic arm moves downward to a horizontal position directly facing the faucet. It then moves forward and to the left, reaching the left side of the faucet. Next, it moves to the right, turning the faucet towards the right side. The arm continues to move in this manner, demonstrating a series of precise and controlled movements as it interacts with the faucet.",
    # "coffee-push-v2": "The video begins with a computer-generated simulated environment featuring a robotic arm with a gripper attached to its end. The arm is positioned above a table that has a brown wooden surface and gray vertical barriers added to its edges. On the table, there are several objects, including a red coffee machine with a white button in front, a white cup, and some small colored objects (green, red, and blue). The robotic arm moves forward to a position above the cup, then moves downward and grasps the cup. Finally, it moves forward, pushing the cup towards a red dot on the table. The video continues with the robotic arm still positioned above the table, with the red coffee machine and the white cup visible. The arm moves forward again, maintaining its position above the cup. It then moves downward and grasps the cup once more. Afterward, it moves forward, pushing the cup towards the red dot on the table. The video concludes with the arm moving forward, still holding the cup, and the cup being pushed towards the red dot."
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
        env = _env_dict.MT50_V2[env_n]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.max_path_length = max_path_length

        s_time = 0; reward = 0
        for i in range(test_n):
            obs = env.reset()
            if write_video: video_list = []
            for j in range(max_path_length):
                obs_rgb = env.sim.render(*(224, 224), mode="offscreen", camera_name="corner")[:,:,::-1]
                if write_video: video_list.append(obs_rgb)
                obs_rgb = processor(obs_rgb).to(device)

                with torch.no_grad():
                    clear_buffer = (j == 0)
                    act = inference(model, obs_rgb, [env_names[env_n]], clear_buffer=clear_buffer)
                
                obs, rew, done, info = env.step(act)
                reward += rew
                if info['success'] or done:
                    s_time += 1
                    break
            if write_video: 
                writer = video_writer(f"{env_n}-{i}-{bool(info['success'] or done)}", 
                                      env.metadata["video.frames_per_second"], (224, 224), video_path)
                for o in video_list:
                    writer.write(o)
                writer.release()
        s_acc[env_n] = s_time / test_n if s_time else s_time
        t_reward[env_n] = reward
    
    # 保存为txt文件
    with open(os.path.join(results_path, "s_acc.txt"), 'w') as f:
        json.dump(s_acc, f, indent=4)
    with open(os.path.join(results_path, "t_reward.txt"), 'w') as f:
        json.dump(t_reward, f, indent=4)

    return s_acc, t_reward

def video_writer(tag, fps, res, video_path):
    return cv2.VideoWriter(
        os.path.join(video_path, f"{tag}.avi"),
        cv2.VideoWriter_fourcc("M","J","P","G"),
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
    device = torch.device("cuda:0")
    processer = processer_basic()
    model = get_model()
    load_pytorch_model(model, checkpoint)
    
    if hasattr(model, "on_inference_start"):
        model.on_inference_start()
    
    val(model, inference, device=device, processor=processer, 
        write_video=True, video_path=f"./validation/{val_model}/video", results_path=f"./validation/{val_model}/results")
    
    
if __name__ == "__main__":
    main("rt1", "/home/casia/workspace/BC/output/rt1_results/TorchTrainer_23d15_00000_0_2024-09-10_20-56-47/checkpoint_000003/model.pt")
    