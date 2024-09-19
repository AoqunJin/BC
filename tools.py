from collections.abc import Mapping
import random
import cv2
import torch
import numpy as np
from numba import njit
import torchvision.transforms as T


def data_map(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, Mapping):
        return {k: data_map(v, device) for k, v in data.items()}
    else:
        return data
    

def processer_basic():
    # 对 RGB 图像进行处理，例如调整大小、标准化等
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    return transform


def processer_canny():
    basic_processer = processer_basic()
    
    def process_frame_canny(frame, fill: int = -1, 
                            point_noise_add: float = 0.02, point_noise_remove: float = 0.1, 
                            line_noise_add: int = 10, line_noise_remove: int = 10):
        if fill == -1: fill = random.randint(0, 3)
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Canny边缘检测
        edges = cv2.Canny(gray, 150, 200)

        # 去除离散值
        # noise_add = np.zeros_like(edges)
        noise_remove_mask = edges > 150
        edges[noise_remove_mask] = 225
        edges[~noise_remove_mask] = 0
        
        # 添加随机点噪声
        noise_add = np.random.random(edges.shape)
        noise_add_mask = noise_add < point_noise_add
        edges[noise_add_mask] = 255

        # 添加随机线噪声
        for _ in range(line_noise_add):
            pt1 = (np.random.randint(0, edges.shape[1]), np.random.randint(0, edges.shape[0]))
            pt2 = (np.random.randint(0, edges.shape[1]), np.random.randint(0, edges.shape[0]))
            cv2.line(edges, pt1, pt2, 255, 1)
        
        # 移除随机点噪声
        noise_remove = np.random.random(edges.shape)
        noise_remove_mask = noise_remove < point_noise_remove
        edges[noise_remove_mask] = 0

        # 移除随机线段
        for _ in range(line_noise_remove):
            pt1 = (np.random.randint(0, edges.shape[1]), np.random.randint(0, edges.shape[0]))
            pt2 = (np.random.randint(0, edges.shape[1]), np.random.randint(0, edges.shape[0]))
            cv2.line(edges, pt1, pt2, 0, 5)  # 使用较粗的线条来确保移除效果

        # # 添加边缘填充
        if fill:
            kernel = np.ones((fill, fill), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        obs = __add_noise(edges_color)
        
        obs = basic_processer(obs)
        return obs

    @njit
    def __add_noise(obs):
        obs = obs.astype(np.float64)  # 转换为浮点数以防止溢出
        for i in range(2):
            noise = np.random.normal(0, 50, obs.shape)
            obs += noise
        obs = np.clip(obs, 0, 255).astype(np.uint8)  # 裁剪到 [0, 255] 范围并转回 uint8
        return obs
    
    return process_frame_canny


@njit
def __discretize_action(action, num_bins=256):
    """
    Discretize a continuous action from [-1, 1] into one of 256 bins.
    
    Args:
    action (float): Continuous action value between -1 and 1.
    num_bins (int): Number of discrete bins (default is 256).
    
    Returns:
    int: Discretized action as an integer between 0 and 255.
    """
    # Ensure the action is within the valid range
    if action < -1: action = -1
    elif action > 1: action = 1
    
    # Map from [-1, 1] to [0, 1]
    normalized_action = (action + 1) / 2
    
    # Map from [0, 1] to [0, 255]
    discretized_action = int(normalized_action * (num_bins - 1))
    
    return discretized_action


@njit
def continuous_to_discrete(actions, num_bins=256):
    """
    Convert an array of continuous actions to discrete actions.
    
    Args:
    actions (numpy.ndarray): Array of continuous actions between -1 and 1.
    
    Returns:
    numpy.ndarray: Array of discretized actions between 0 and 255.
    """
    return np.array([__discretize_action(a, num_bins) for a in actions])