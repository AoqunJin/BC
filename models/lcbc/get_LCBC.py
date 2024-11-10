import torch
import torch.nn.functional as F

from .LCBC import LCBC


def calculate_accuracy(preds, labels):
    predicted_labels = torch.argmax(preds, dim=-1)
    correct_predictions = (predicted_labels == labels).sum()
    total_predictions = labels.numel()
    accuracy = correct_predictions / total_predictions
    
    return accuracy


def get_model(num_actions = 4, action_bins = 256, **kwargs):
    return LCBC(num_actions, action_bins, **kwargs)


def forward_fn(model, **batch):
    # [4, 20, 3, 224, 224]
    video = batch['frame']
    instruction = batch['instruction']
    label = batch['action'].to(torch.long)
    total_loss = []
    total_acc = []
    total_iter = 0
    for i in range(video.shape[1]):
        train_logits = model(instruction, video[:, i])  # .squeeze()
        
        total_loss.append(F.cross_entropy(train_logits.permute(0, 2, 1), label[:, i]))
        with torch.no_grad():
            total_acc.append(calculate_accuracy(train_logits, label[:, i]))
        total_iter += 1

    return {"loss": sum(total_loss) / total_iter, "acc": sum(total_acc) / total_iter}


def inference(model, frame, instruction, **kwargs):
    model.eval()
    with torch.no_grad():
        eval_logits = model(instruction, frame.unsqueeze(0))
    return torch.argmax(eval_logits, dim=-1).squeeze().cpu().detach().numpy()


if __name__ == "__main__":
    model = get_model()
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
    
    image = torch.randn(3, 224, 224)
    instruction = ["Your instructions here"]
    result = inference(model, image, instruction).squeeze()
    print(result)
    