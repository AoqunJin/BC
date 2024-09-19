from collections.abc import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, AutoTokenizer, T5EncoderModel
from einops.layers.torch import Rearrange


def data_to(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, Mapping):
        return {k: data_to(v, device) for k, v in data.items()}
    else:
        return data


class LCBC(nn.Module):
    def __init__(self, num_actions = 4, action_bins = 3, **kwargs):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small")
        self.tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")

        self.fusion_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(self.fusion_layer, num_layers=6)
        
        self.to_logits = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, num_actions * action_bins),
            Rearrange('... (a b) -> ... a b', b = action_bins)
        )
        
    def forward(self, text_inputs, vision_inputs):
        with torch.no_grad():
            # Features
            vision_outputs = self.vision_model(pixel_values=vision_inputs)
            vision_features = vision_outputs.last_hidden_state
            
            text_inputs = self.tokenizer(text_inputs, padding=True, return_tensors="pt")
            text_inputs = data_to(text_inputs, vision_features.device)
            
            text_outputs = self.text_model(**text_inputs)
            text_features = text_outputs.last_hidden_state
            
            # 在最后一维的末尾填充256个单位，前面0个，填充值为0
            text_features = F.pad(text_features, pad=(0, 256), mode='constant', value=0)

            # Combine features
            combined_features = torch.cat([vision_features, text_features], dim=1)
            
        # Apply fusion
        fused_features = self.fusion_encoder(combined_features)
        # print(fused_features)
        fused_output = torch.mean(fused_features, dim=1)
        
        logits = self.to_logits(fused_output)
        return logits

    def load_state_dict(self, state_dict):
        self.fusion_encoder.load_state_dict(state_dict["fusion_encoder"])
        self.to_logits.load_state_dict(state_dict["to_logits"])
        
    def state_dict(self):
        return {
            "fusion_encoder": self.fusion_encoder.state_dict(),
            "to_logits": self.to_logits.state_dict()
        }