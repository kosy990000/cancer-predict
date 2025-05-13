import torch
import torch.nn as nn
import torch.nn.functional as F

class MutationAttentionWithAdv(nn.Module):
    def __init__(self, num_genes=4384, d_model=128):
        super(MutationAttentionWithAdv, self).__init__()

        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=3, batch_first=True)

        self.input_proj = nn.Linear(4, d_model)  # mutation_input에 적용
        self.k_proj =  nn.Linear(d_model, d_model) # DxD
        self.v_proj = nn.Linear(d_model, d_model)
        
    

        self.attentionNorm = nn.LayerNorm(d_model)

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 26)
        )

    def forward(self, mutation_input, external_V):
        """
        mutation_input: FloatTensor [batch_size, 4384, 5]
        external_V: FloatTensor [4384, d_model] — 
        """
        batch_size = mutation_input.size(0)
        
        wt_mask = (mutation_input == torch.tensor([1,0,0,0,0], device=mutation_input.device)).all(dim=-1)  # [B, G]
        masked_input = mutation_input[..., 1:]
        valid_mask = ~wt_mask  # [B, G]
        # Q: mutation 정보 기반
        Q = self.input_proj(masked_input)  # [batch, 4384, d_model]


        # K: 학습 파라미터
        K = self.k_proj(external_V).unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 4384, d_model]

        # V: 외부 입력 (co-occurrence vector), 배치 차원으로 broadcast
        V = self.v_proj(external_V).unsqueeze(0).expand(batch_size, -1, -1) # [batch, 4384, d_model]


        attention_output, attention_weights = self.multihead_attention(Q, K, V, key_padding_mask=wt_mask)
       

        # 5. 마스킹된 된 위치 유지
        masked_output = attention_output * valid_mask.unsqueeze(-1)
        
        mean_pool = (masked_output).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True)


        output = self.output_layer(mean_pool)
        mean_pool = self.attentionNorm(mean_pool)

        return output

class MutationAttention(nn.Module):
    def __init__(self, num_genes=4384, d_model=128):
        super(MutationAttention, self).__init__()

        self.input_proj = nn.Linear(4, d_model)  # mutation_input에 적용

        # Key는 학습 파라미터로 유지
        self.key = nn.Parameter(torch.randn(num_genes, d_model))

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 26)
        )

    def forward(self, mutation_input, external_V):
        """
        mutation_input: FloatTensor [batch_size, 4384, 5]
        external_V: FloatTensor [4384, d_model] — 
        """
        batch_size = mutation_input.size(0)
        
        wt_mask = (mutation_input == torch.tensor([1,0,0,0,0], device=mutation_input.device)).all(dim=-1)  # [B, G]
        masked_input = mutation_input[..., 1:]
        # Q: mutation 정보 기반
        Q = self.input_proj(masked_input)  # [batch, 4384, d_model]

        # K: 학습 파라미터
        K = self.key.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 4384, d_model]

        # V: 외부 입력 (co-occurrence vector), 배치 차원으로 broadcast
        V = external_V.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 4384, d_model]

        # Attention
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)

        # 6. Attention 마스크 적용 
        wt_mask_q = wt_mask.unsqueeze(2)  # [B, G, 1]
        wt_mask_k = wt_mask.unsqueeze(1)  # [B, 1, G]
        attn_mask = wt_mask_q | wt_mask_k  # [B, G, G]
        attention_scores = attention_scores.masked_fill(attn_mask, -1e9)


        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.bmm(attention_weights, V)  # [batch, 4384, d_model]

        weight_sum = attention_output.mean(dim=1)  # [batch, d_model]
        output = self.output_layer(weight_sum)

        return output.squeeze(-1)
    


class MutationAttentionAdv(nn.Module):
    def __init__(self, num_genes=4384, d_model=128):
        super(MutationAttentionAdv, self).__init__()

        self.input_proj = nn.Linear(4, d_model)  # mutation_input에 적용

        # Key는 학습 파라미터로 유지
        self.key = nn.Parameter(torch.randn(num_genes, d_model))

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 26)
        )

    def forward(self, mutation_input, external_V):
        """
        mutation_input: FloatTensor [batch_size, 4384, 5]
        external_V: FloatTensor [4384, d_model] — 
        """
        batch_size = mutation_input.size(0)
        
        wt_mask = (mutation_input == torch.tensor([1,0,0,0,0], device=mutation_input.device)).all(dim=-1)  # [B, G]
        masked_input = mutation_input[..., 1:]
        # Q: mutation 정보 기반
        Q = self.input_proj(masked_input)  # [batch, 4384, d_model]

        # K: 학습 파라미터
        K = self.key.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 4384, d_model]

        # V: 외부 입력 (co-occurrence vector), 배치 차원으로 broadcast
        V = external_V.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 4384, d_model]

        # Attention
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)

        # 6. Attention 마스크 적용 
        wt_mask_q = wt_mask.unsqueeze(2)  # [B, G, 1]
        wt_mask_k = wt_mask.unsqueeze(1)  # [B, 1, G]
        attn_mask = wt_mask_q | wt_mask_k  # [B, G, G]
        attention_scores = attention_scores.masked_fill(attn_mask, -1e9)


        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.bmm(attention_weights, V)  # [batch, 4384, d_model]

        # gene-level attention weights의 가중 평균
        valid_mask = ~wt_mask  # [B, G]
        valid_mask = valid_mask.unsqueeze(-1)  # [B, G, 1]
        attention_output = attention_output * valid_mask  # zero out wt
        weight_sum = attention_output.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

        

        output = self.output_layer(weight_sum)

        return output.squeeze(-1)


