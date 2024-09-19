import torch
import torch.nn.functional as F

from .inverse_model import InverseModel

def calculate_accuracy(preds, labels):
    predicted_labels = torch.argmax(preds, dim=-1)
    correct_predictions = (predicted_labels == labels).sum()
    total_predictions = labels.numel()
    accuracy = correct_predictions / total_predictions
    
    return accuracy


def get_model(num_actions = 4, action_bins = 3, **kwargs):
    return InverseModel(num_actions, action_bins, **kwargs)


def forward_fn(model, **batch):
    # [4, 20, 3, 224, 224]
    video = batch['frame']
    instruction = batch['instruction']
    label = batch['action'].to(torch.long)
    
    train_logits = model(instruction, video).squeeze(3)
    loss = F.cross_entropy(train_logits.permute(0, 3, 1, 2), label)

    with torch.no_grad():
        acc = calculate_accuracy(train_logits, label)
    return {"loss": loss, "acc": acc}


def inference(model, frame, instruction, **kwargs):
    model.eval()
    with torch.no_grad():
        eval_logits = model(instruction, frame.unsqueeze(0))
    return (torch.argmax(eval_logits, dim=-1) - 1).squeeze().cpu().detach().numpy()


if __name__ == "__main__":
    # 使用示例
    model = get_model(vision_model="vit")
    # print("number of parameters: {:e}".format(
    #     sum(p.numel() for p in model.parameters()))
    # )  # number of parameters: 1.613975e+08
    
    loss = forward_fn(
        model,
        frame=torch.randn(2, 20, 3, 224, 224),
        instruction=["a photo of a cat", "a photo of a cat"],
        action=torch.randint(0, 2, size=(2, 20, 4))
    )
    print(loss)
    
    # 测试
    image = torch.randn(20, 3, 224, 224)  # 假设这是您的输入图像
    instruction = ["Your instructions here"]
    result = inference(model, image, instruction).squeeze()
    print(result.shape)
    