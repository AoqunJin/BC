import random
import h5py
import numpy as np
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from tools import processer_basic, processer_canny, continuous_to_discrete


class HDF5VideoDataset(Dataset):
    def __init__(self, data_path, processer=None, seq_len=16, 
                 required_tasks=None, avoid_tasks=None, 
                 use_language=False, **kwargs):
        self.data_path = data_path
        self.processer = get_processer(processer)()
        self.processer_name = processer
        self.seq_len = seq_len
        self.use_language = use_language
        with h5py.File(self.data_path, 'r') as f:
            self.env_names = list(f.keys())
            self.demos = []
            for env_name in self.env_names:
                for task_name in f[env_name].keys():
                    if required_tasks and task_name not in required_tasks:
                        continue
                    elif avoid_tasks and task_name in avoid_tasks:
                        continue
                    for timestamp in f[env_name][task_name].keys():
                        self.demos.append((env_name, task_name, timestamp))

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, idx):
        env_name, task_name, timestamp = self.demos[idx]
        with h5py.File(self.data_path, 'r') as f:
            demo_group = f[f"{env_name}/{task_name}/{timestamp}"]
            
            max_end = len(demo_group['action'][:])
            max_sta = max(0, max_end - self.seq_len)
            
            rand_sta = random.randint(0, max_sta)
            rand_end = min(max_end, rand_sta + self.seq_len)
            
            # observation = torch.from_numpy(demo_group['observation'][rand_sta:rand_end])
            # reward = torch.from_numpy(demo_group['reward'][rand_sta:rand_end])
            # done = torch.from_numpy(demo_group['done'][rand_sta:rand_end])
            frame = demo_group['frames'][rand_sta:rand_end]
            if not self.use_language:
                instruction = ""
            elif random.random() < 0.05:
                instruction = ""
            else:
                instruction = demo_group['instruction'][()].decode()  # 读取字符串数据集
            
            # -1, 0, 1 => [3]
            action = demo_group['action'][rand_sta:rand_end]
            for i in range(len(action)):
                action[i] = continuous_to_discrete(action[i], 3)
            action = torch.from_numpy(action)
            
            if self.processer:
                if self.processer_name == "canny":
                    noise = random.randint(1, 3)
                    new_frame = np.stack([self.processer(frame[i], noise) for i in range(len(frame))])
                else:
                    new_frame = np.stack([self.processer(frame[i]) for i in range(len(frame))])
                frame = new_frame

            frame = torch.from_numpy(frame)
            
            return {
                # 'observation': observation,
                # 'reward': reward,
                # 'done': done,
                'action': action,
                'frame': frame,
                'instruction': instruction,
            }


def collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        # action padding
        if key == 'action': padding_value = 1
        else: padding_value = 0
        # instruction padding
        if key == 'instruction': collated[key] = [item[key] for item in batch]
        else: collated[key] = pad_sequence([item[key] for item in batch], batch_first=True, padding_value=padding_value)
    
    return collated


def get_processer(processer="basic"):
    if processer == "basic": processer = processer_basic
    elif processer == "canny": processer = processer_canny
    return processer


def get_dataset(**config):
    return HDF5VideoDataset(**config)
    

def get_dataloader(**config):
    return DataLoader(
        get_dataset(**config), config["batch_size"], 
        num_workers=config["num_workers"], collate_fn=collate_fn
    )

    
def get_dataloader_split(**config):
    train_ds = get_dataset(**config, avoid_tasks=['ButtonPressTopdownWall', 'CoffeeButton', 'PlateSlide', 'FaucetOpen', 'PushWall'])
    
    zero_ds = get_dataset(**config, required_tasks=['ButtonPressTopdownWall', 'CoffeeButton', 'PlateSlide', 'FaucetOpen', 'PushWall'])    
    
    train_size = int(0.95 * len(train_ds))  # 95% for traing
    val_size = len(train_ds) - train_size  # 5% for validation

    train_dataset, val_dataset = random_split(train_ds, [train_size, val_size])
    
    train_dl = DataLoader(train_dataset, config["batch_size"], 
                          num_workers=config["num_workers"], collate_fn=collate_fn)
    val_dl = DataLoader(val_dataset, config["batch_size"], 
                        num_workers=config["num_workers"], collate_fn=collate_fn)
    zero_dl = DataLoader(zero_ds, config["batch_size"], 
                         num_workers=config["num_workers"], collate_fn=collate_fn)
    
    return train_dl, val_dl, zero_dl
    

if __name__ == "__main__":
    data_path = "/home/ao/workspace/BC/data/metaworld.hdf5"
    dataset = HDF5VideoDataset(data_path, 'canny', seq_len=8)
    dataloader = DataLoader(dataset, batch_size=8,
                      num_workers=4, collate_fn=collate_fn)
    
    for batch in dataloader:
        print("Batch keys:", batch.keys())  # dict_keys(['observation', 'action', 'reward', 'done', 'frames', 'instruction'])
        # print("Observation shape:", batch['observation'].shape)  # torch.Size([4, 227, 39])
        print("Action shape:", batch['action'].shape)  # torch.Size([4, 227, 4])
        print("Frames shape:", batch['frame'].shape)  # torch.Size([4, 227, 3, 224, 224])
        print("Instructions:", len(batch['instruction']))  # List (4)
        
        save_image(batch['frame'][:, 0] * 255, 'test.png')
        break
    