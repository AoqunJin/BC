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
    # Perform processing on RGB images, such as resizing, normalization, etc.
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
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Canny edge detection
        edges = cv2.Canny(gray, 150, 200)

        # Remove discrete values
        # noise_add = np.zeros_like(edges)
        noise_remove_mask = edges > 150
        edges[noise_remove_mask] = 225
        edges[~noise_remove_mask] = 0
        
        # Add random noise
        noise_add = np.random.random(edges.shape)
        noise_add_mask = noise_add < point_noise_add
        edges[noise_add_mask] = 255

        # Add random line noise
        for _ in range(line_noise_add):
            pt1 = (np.random.randint(0, edges.shape[1]), np.random.randint(0, edges.shape[0]))
            pt2 = (np.random.randint(0, edges.shape[1]), np.random.randint(0, edges.shape[0]))
            cv2.line(edges, pt1, pt2, 255, 1)
        
        # Remove random point noise
        noise_remove = np.random.random(edges.shape)
        noise_remove_mask = noise_remove < point_noise_remove
        edges[noise_remove_mask] = 0

        # Remove random line segments
        for _ in range(line_noise_remove):
            pt1 = (np.random.randint(0, edges.shape[1]), np.random.randint(0, edges.shape[0]))
            pt2 = (np.random.randint(0, edges.shape[1]), np.random.randint(0, edges.shape[0]))
            cv2.line(edges, pt1, pt2, 0, 5)  # Use thicker lines to ensure removal

        # Add edge padding
        if fill:
            kernel = np.ones((fill, fill), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        obs = __add_noise(edges_color)
        
        obs = basic_processer(obs)
        return obs

    @njit
    def __add_noise(obs):
        obs = obs.astype(np.float64)  # Convert to floating point to prevent overflow
        for i in range(2):
            noise = np.random.normal(0, 50, obs.shape)
            obs += noise
        obs = np.clip(obs, 0, 255).astype(np.uint8)  # Clip to [0, 255] range and convert back to uint8
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


@njit
def __discrete_to_continuous(discrete_action, num_bins=256):
    """
    Convert a discretized action (integer) back to a continuous action in the range [-1, 1].
    
    Args:
    discrete_action (int): Discretized action as an integer between 0 and (num_bins - 1).
    num_bins (int): Number of discrete bins (default is 256).
    
    Returns:
    float: Continuous action in the range [-1, 1].
    """
    # Ensure the action is within the valid discrete range
    if discrete_action < 0: discrete_action = 0
    elif discrete_action >= num_bins: discrete_action = num_bins - 1
    
    # Map from [0, num_bins - 1] to [0, 1]
    normalized_action = discrete_action / (num_bins - 1)
    
    # Map from [0, 1] to [-1, 1]
    continuous_action = normalized_action * 2 - 1
    
    return continuous_action


@njit
def discrete_to_continuous(discrete_actions, num_bins=256):
    """
    Convert an array of discretized actions to continuous actions.
    
    Args:
    discrete_actions (numpy.ndarray): Array of discretized actions between 0 and (num_bins - 1).
    
    Returns:
    numpy.ndarray: Array of continuous actions in the range [-1, 1].
    """
    return np.array([__discrete_to_continuous(a, num_bins) for a in discrete_actions])
