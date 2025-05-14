import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, sequence_num, d_model=150):
        super(SelfAttention, self).__init__()

        #self.multihead_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=3, batch_first=True)

        self.input_proj = nn.Linear(d_model, d_model)  # mutation_input에 적용
        self.k_proj =  nn.Linear(d_model, d_model) # DxD
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.attentionNorm = nn.LayerNorm(d_model)


        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 26)
        )


    def forward(self, q_input, kv_input, key_padding_mask=None):
        Q = self.input_proj(q_input)           # importance 반영된 Q
        K = self.k_proj(kv_input)              # 원본 K
        V = self.v_proj(kv_input)              # 원본 V

        attention_output, attention_weights = self.multihead_attention(
            Q, K, V, key_padding_mask=key_padding_mask
        )

        mean_pool = attention_output.mean(dim=1)

        mean_pool = self.attentionNorm(mean_pool)

        output = self.output_layer(mean_pool)
        return output




class SelfAttentionMultiHead(nn.Module):
    def __init__(self, sequence_num, d_model=150, ff_hidden_dim=256, num_classes=26):
        super(SelfAttentionMultiHead, self).__init__()
        self.d_model = d_model

        # Q, K, V projection
        self.input_proj = nn.Linear(d_model, d_model)  # mutation_input에 적용
        self.k_proj =  nn.Linear(d_model, d_model) # DxD
        self.v_proj = nn.Linear(d_model, d_model)

        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=3, batch_first=True)
        # LayerNorm for attention output
        self.attn_norm = nn.LayerNorm(d_model)

        # FeedForward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # Output layer
        self.output_layer = nn.Linear(d_model, num_classes)


    def forward(self, q_input, kv_input, key_padding_mask=None):
        Q = self.input_proj(q_input)   # [B, T, d_model]
        K = self.k_proj(kv_input)
        V = self.v_proj(kv_input)

        attn_output, _ = self.attention(Q, K, V, key_padding_mask=key_padding_mask)  # shape: [B, T, d_model]

        # Residual + LayerNorm
        attn_output = self.attn_norm(Q + attn_output)

        # FFN → Residual + LayerNorm
        ffn_output = self.ffn(attn_output)
        ffn_output = self.ffn_norm(attn_output + ffn_output)

        # Mean Pooling
        pooled = ffn_output.mean(dim=1)

        return self.output_layer(pooled)


class SelfAttentionOneHeadWithDropOut(nn.Module):
    def __init__(self, sequence_num, d_model=150, ff_hidden_dim=256, num_classes=26):
        super(SelfAttentionOneHeadWithDropOut, self).__init__()
        self.d_model = d_model

        # Q, K, V projection
        self.input_proj = nn.Linear(d_model, d_model)  # mutation_input에 적용
        self.k_proj =  nn.Linear(d_model, d_model) # DxD
        self.v_proj = nn.Linear(d_model, d_model)

        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)
        # LayerNorm for attention output
        self.attn_norm = nn.LayerNorm(d_model)

        # FeedForward network
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),        # 추가
            nn.Linear(ff_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),        # 추가
            nn.Linear(64, 26)
        )

        self.ffn_norm = nn.LayerNorm(d_model)

        # Output layer
        self.output_layer = nn.Linear(d_model, num_classes)


        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)

    def forward(self, q_input, kv_input, key_padding_mask=None):
        Q = self.input_proj(q_input)   # [B, T, d_model]
        K = self.k_proj(kv_input)
        V = self.v_proj(kv_input)

        attn_output, _ = self.attention(Q, K, V, key_padding_mask=key_padding_mask)  # shape: [B, T, d_model]

        # Residual + LayerNorm
        attn_output = self.attn_norm(Q + attn_output)

        # FFN → Residual + LayerNorm
        ffn_output = self.ffn(attn_output)
        ffn_output = self.ffn_norm(attn_output + ffn_output)

        # Mean Pooling
        pooled = ffn_output.mean(dim=1)

        return self.output_layer(pooled)


    




