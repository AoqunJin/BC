from collections import deque
from functools import wraps
import torch
import torch.nn.functional as F

from .robotic_transformer_pytorch import MaxViT, RT1


def calculate_accuracy(preds, labels):
    predicted_labels = torch.argmax(preds, dim=-1)
    correct_predictions = (predicted_labels == labels).sum()
    total_predictions = labels.numel()
    accuracy = correct_predictions / total_predictions
    
    return accuracy


def get_model(num_actions = 4, action_bins = 256, **kwargs):
    vit = MaxViT(
        num_classes = 1000,
        dim_conv_stem = 64,
        dim = 96,
        dim_head = 32,
        depth = (2, 2, 5, 2),
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1
    )

    model = RT1(
        vit = vit,
        num_actions = num_actions,
        action_bins = action_bins,
        depth = 6,
        heads = 8,
        dim_head = 64,
        cond_drop_prob = 0.2
    )
    return model


def forward_fn(model, **batch):
    # [4, 20, 3, 224, 224] -> [4, 3, 20, 224, 224]
    video = batch['frame'].permute(0, 2, 1, 3, 4)
    instructions = batch['instruction']
    label = batch['action'].to(torch.long)
    
    train_logits = model(video, instructions)  # (batch, frames, actions, bins)
    loss = F.cross_entropy(train_logits.permute(0, 3, 1, 2), label)
    with torch.no_grad():
        acc = calculate_accuracy(train_logits, label)
    return {"loss": loss, "acc": acc}


def video_inference_decorator(max_frames=32):
    frame_buffer = deque(maxlen=max_frames)
    
    def decorator(func):
        @wraps(func)
        def wrapper(model, image, instructions, clear_buffer=False, *args, **kwargs):
            if clear_buffer:
                frame_buffer.clear()
                
            # 添加新帧到缓冲区
            frame_buffer.append(image)
            
            # 创建视频张量
            video = torch.stack(list(frame_buffer), dim=0)  # [L, 3, 224, 224]
            video = video.unsqueeze(0)  # 添加批次维度 [1, L, 3, 224, 224]
            video = video.permute(0, 2, 1, 3, 4)  # 调整为 [1, 3, L, 224, 224]
            
            # 调用原始函数
            return func(model, video, instructions, *args, **kwargs)
        
        return wrapper
    return decorator


@video_inference_decorator(max_frames=8)
def inference(model, video, instructions, **kwargs):
    model.eval()
    with torch.no_grad():
        eval_logits = model(video, instructions, cond_scale=3.)[-1, -1]  # classifier free guidance with conditional scale of 3
    return (torch.argmax(eval_logits, dim=-1) - 1).squeeze().cpu().detach().numpy()


if __name__ == '__main__':
    # 使用示例
    model = get_model()
    print("number of parameters: {:e}".format(
        sum(p.numel() for p in model.parameters()))
    )  # number of parameters: 2.694905e+08
    
    loss = forward_fn(
        model,
        frame=torch.randn(2, 20, 3, 224, 224),
        instruction=["a photo of a cat", "a photo of a cat"],
        action=torch.randint(0, 2, size=(2, 20, 4))
    )
    print(loss)
    
    # 测试
    image = torch.randn(3, 224, 224)  # 假设这是您的输入图像
    instructions = ["Your instructions here"]
    result = inference(model, image, instructions)[0, -1].squeeze()
    print(result.size())
    
    image = torch.randn(3, 224, 224)  # 假设这是您的输入图像
    result = inference(model, image, instructions)[0, -1].squeeze()
    print(result.size())
