import torch
import torch.nn as nn

from core.models.pos import positional_encoding
from core.models.encoding import EncoderMLP
from core.models.block import Block, MLP


class Temporal_Encoder(nn.Module):
    def __init__(self, N, embed_dim, n_heads, forward_expansion, act_name, p, attn_p, device):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.device = device
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    forward_expansion=forward_expansion,
                    act_name=act_name,
                    p=p,
                    attn_p=attn_p,
                ) for _ in range(N)
            ]
        )
        
        self.demographics_encoder = EncoderMLP(
            input_dim=5,
            embed_dim=embed_dim,
            forward_expansion=forward_expansion,
            activation=act_name,
        )
    
        self.value_encoder = EncoderMLP(
            input_dim=1,
            embed_dim=embed_dim,
            forward_expansion=forward_expansion,
            activation=act_name,
        )
        
    def forward(self, x_static, x_temporal):
        encoding_static = self.demographics_encoder(x_static).unsqueeze(1)
        
        # compute encoding of time steps
        time_steps = x_temporal[:, :, 0]
        zeros = torch.zeros(time_steps.size(0), 1, device=time_steps.device)
        time_steps = torch.cat((zeros, time_steps), dim=1) + 1
        encoding_time_steps = positional_encoding(time_steps, self.embed_dim, self.device)
        
        # computing embedding of sensor values
        encoding_temporal = self.value_encoder(x_temporal[:, :, 1].unsqueeze(2))

        x = torch.cat([encoding_static, encoding_temporal], dim=1)
        x += encoding_time_steps
        x = self.norm(x)
        
        for block in self.blocks:
            x = block(x)
            
        cls_token = x[:, 0, :].unsqueeze(1)
        return cls_token
    
    
class PhysioFormer(nn.Module):
    def __init__(self, N, embed_dim, n_heads, forward_expansion, act_name, p, attn_p, device):
        super().__init__()
        
        self.temporal_vars = [
            'Temp', 'pH', 'FiO2', 'TroponinT', 'Creatinine', 'PaCO2', 'HCT', 'TroponinI',\
            'AST', 'Mg', 'SysABP', 'RespRate', 'NIDiasABP', 'Platelets', 'Cholesterol', 'Albumin', 'MechVent',\
            'NISysABP', 'Glucose', 'MAP', 'ALT', 'Lactate', 'Na', 'K', 'WBC', 'SaO2', 'HCO3', 'Bilirubin',\
            'BUN', 'ALP', 'Weight', 'DiasABP', 'PaO2', 'Urine', 'HR', 'GCS', 'NIMAP'
        ]
                
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self.demographics_encoder = EncoderMLP(
            input_dim=5,
            embed_dim=embed_dim,
            forward_expansion=forward_expansion,
            activation=act_name,
            p=p,
        )
        
        self.positional_embeddings = nn.Parameter(torch.randn(1, 1+len(self.temporal_vars), embed_dim))
        
        self.temporal_encoders = nn.ModuleDict(
            {
                var: Temporal_Encoder(
                    N=N, 
                    embed_dim=embed_dim, 
                    n_heads=n_heads, 
                    forward_expansion=forward_expansion, 
                    act_name=act_name, 
                    p=p, 
                    attn_p=attn_p,
                    device=device,
                )
                for var in self.temporal_vars
            }
        )
        
        self.blocks = nn.ModuleList([Block(
            embed_dim=embed_dim,
            n_heads=n_heads,
            forward_expansion=forward_expansion,
            act_name=act_name,
            p=p,
            attn_p=attn_p,
        ) for _ in range(N)])
        
        self.classif_head = MLP(
            in_features=embed_dim,
            hidden_features=embed_dim,
            act_name=act_name,
            out_features=1,
            p=p,
        )
        
    def forward(self, data):
        demographics_encoding = self.demographics_encoder(data['Static']).unsqueeze(1)
        temporal_encodings = {
            var: self.temporal_encoders[var](data['Static'], data['Temporal'][var])
            for var in self.temporal_vars
        }
        
        # Sort the dictionary by keys and then concatenate the values
        sorted_keys = sorted(temporal_encodings.keys())
        concatenated_encodings = torch.cat([temporal_encodings[key] for key in sorted_keys], dim=1)
        
        x = torch.cat([demographics_encoding, concatenated_encodings], dim=1)
        x = x + self.positional_embeddings
        x = self.norm(x)
        
        for block in self.blocks:
            x = block(x)
        
        cls_token = x[:, 0, :].squeeze(1)
        
        logits = self.classif_head(cls_token).squeeze(1)
        
        preds = torch.sigmoid(logits)

        return preds
