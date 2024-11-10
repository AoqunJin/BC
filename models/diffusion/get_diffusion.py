from functools import wraps
from collections import deque
import torch
import torch.nn as nn

from .diffusion import (
    DiffusionPolicy,
    OBS_HORIZON,
    ACTION_HORIZON,
    PRED_HORIZON
)

def calculate_accuracy(preds, labels, tolerance=0.1):
    if preds.shape != labels.shape:
        raise ValueError

    diff = torch.abs(preds - labels)
    correct = diff < tolerance
    correct_percentage = torch.mean(correct.float())
    return correct_percentage


def get_model(num_actions = 4, **kwargs):
    vision_feature_dim = 512
    lowdim_obs_dim = 0  # No low dim
    obs_dim = vision_feature_dim + lowdim_obs_dim
    obs_horizon = 2 + 1  # 2V + 1L
    model = DiffusionPolicy(obs_dim, obs_horizon, num_actions)
    return model


def pad_sequence(batch, pred_horizon, obs_horizon=0, padd_action=0):
    obs = batch['frame']
    action = batch['action']

    if obs_horizon != 0:
        # Pad observations at the beginning
        pad_obs = obs[:, :1].repeat(1, obs_horizon - 1, 1, 1, 1)
        obs = torch.cat([pad_obs, obs], dim=1)
    
    if padd_action != 0:
        # Pad actions at the end
        pad_action = action[:, -1:].repeat(1, padd_action, 1)
        action = torch.cat([action, pad_action], dim=1)
    
    return {'frames': obs, 'action': action}


def split_and_iterate_sequence(batch, pred_horizon, obs_horizon, action_horizon):
    padded_batch = pad_sequence(batch, pred_horizon, obs_horizon, 0)
    obs = padded_batch['frames']
    action = padded_batch['action']

    seq_length = action.shape[1] - pred_horizon + 1
    
    for i in range(seq_length):
        obs_slice = obs[:, i:i+obs_horizon]
        action_slice = action[:, i:i+action_horizon]
        pred_action_slice = action[:, i:i+pred_horizon]

        batch_slice = {
            'image': obs_slice,
            'action': pred_action_slice,
            'language': batch['instruction']
        }

        yield batch_slice


def forward_fn(model, **batch):
    pred_horizon = PRED_HORIZON
    obs_horizon = OBS_HORIZON
    action_horizon = ACTION_HORIZON
    # No norm
    total_loss = []
    total_acc = []
    total_iter = 0
    for batch_slice in split_and_iterate_sequence(batch, pred_horizon, obs_horizon, action_horizon):
        noise_pred, noise = model(**batch_slice)
        total_loss.append(nn.functional.mse_loss(noise_pred, noise))
        with torch.no_grad():
            total_acc.append(calculate_accuracy(noise_pred, noise))
        total_iter += 1
    return {"loss": sum(total_loss) / total_iter, "acc": sum(total_acc) / total_iter}


def video_inference_decorator(max_frames=2):
    frame_buffer = deque(maxlen=max_frames)
    action_buffer = deque()
    
    def decorator(func):
        @wraps(func)
        def wrapper(model, image, instructions, clear_buffer=False, *args, **kwargs):
            if clear_buffer:
                frame_buffer.clear()
                action_buffer.clear()
            if len(frame_buffer) == 0:
                frame_buffer.append(image)
            frame_buffer.append(image)

            video = torch.stack(list(frame_buffer), dim=0)  # [L, 3, 224, 224]
            video = video.unsqueeze(0)  # [1, L, 3, 224, 224]
            
            if len(action_buffer) == 0:
                action = func(model, video, instructions, *args, **kwargs)
                for a in action:
                    action_buffer.append(a)

            return action_buffer.popleft()
        return wrapper
    return decorator


@video_inference_decorator(max_frames=2)
def inference(model, frame, instructions, **kwargs):
    noisy_action = model.inference(frame, instructions)

    # unnormalize action
    naction = noisy_action.detach().to('cpu').numpy()
    # (B, pred_horizon, action_dim)
    naction = naction.squeeze()

    return naction[:ACTION_HORIZON]


if __name__ == "__main__":
    model = get_model()

    print("number of parameters: {:e}".format(
        sum(p.numel() for p in model.parameters()))
    )  # number of parameters: 7.994727e+07
    
    loss = forward_fn(
        model,
        frame=torch.randn(2, 8, 3, 224, 224),
        instruction=["a photo of a cat", "a photo of a cat"],
        action=torch.randn(2, 8, 4)
    )
    print(loss)
    
    image = torch.randn(3, 224, 224)
    instructions = ["Your instructions here Your instructions here"]
    result = inference(model, image, instructions).squeeze()
    print(result.shape)
    