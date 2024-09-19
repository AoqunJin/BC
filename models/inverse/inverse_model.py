from collections.abc import Mapping
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from transformers import CLIPVisionModel, AutoTokenizer, T5EncoderModel
from einops.layers.torch import Rearrange


def data_to(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, Mapping):
        return {k: data_to(v, device) for k, v in data.items()}
    else:
        return data


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x


class InverseModel(nn.Module):
    def __init__(self, num_actions = 4, action_bins = 3, vision_model = "vit", use_language = False, **kwargs):
        super().__init__()
        self.vision_model_name = vision_model
        if vision_model == "resnet":
            self.vision_model = resnet18()
        elif vision_model == "vit":
            self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        self.use_language = use_language  # Only fp32 bf16 TF32
        self.tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")
        self.text_model = T5EncoderModel.from_pretrained("google/t5-v1_1-small")

        self.fusion_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(self.fusion_layer, num_layers=6)
        
        self.to_logits = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, num_actions * action_bins),
            Rearrange('... (a b) -> ... a b', b = action_bins)
        )
        self.positional_encoding = PositionalEncoding(768)
        
    def forward(self, text_inputs, vision_inputs):
        # Features
        # with torch.no_grad():
        batch_size, num_images = vision_inputs.shape[:2]
        vision_inputs_reshaped = vision_inputs.view(-1, *vision_inputs.shape[2:])
        # Res
        if self.vision_model_name == "resnet":
            vision_outputs = self.vision_model(vision_inputs_reshaped)
            features = vision_outputs[:, :768]
        # Vit
        elif self.vision_model_name == "vit":
            vision_outputs = self.vision_model(pixel_values=vision_inputs_reshaped)
            features = vision_outputs.pooler_output
        
        features = features.view(batch_size, num_images, -1)
        
        if self.use_language:
            text_inputs = self.tokenizer(text_inputs, padding=True, return_tensors="pt")
            text_inputs = data_to(text_inputs, features.device)
            text_outputs = self.text_model(**text_inputs)
            text_features = text_outputs.last_hidden_state
            
            # 在最后一维的末尾填充256个单位，前面0个，填充值为0 | 512->768
            text_features = F.pad(text_features, pad=(0, 256), mode='constant', value=0)

            # Combine features
            features = torch.cat([text_features, features], dim=1)
        
        position_features = self.positional_encoding(features)
            
        # Apply fusion
        fused_features = self.fusion_encoder(position_features)
        fused_output = fused_features[:, -num_images:]
        
        logits = self.to_logits(fused_output)
        return logits

    # def load_state_dict(self, state_dict):
    #     self.fusion_encoder.load_state_dict(state_dict["fusion_encoder"])
    #     self.to_logits.load_state_dict(state_dict["to_logits"])
        
    # def state_dict(self):
    #     return {
    #         "fusion_encoder": self.fusion_encoder.state_dict(),
    #         "to_logits": self.to_logits.state_dict()
    #     }
    