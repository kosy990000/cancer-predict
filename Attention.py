import torch
import torch.nn as nn
import torch.nn.functional as F

class MutationAttention(nn.Module):
    def __init__(self, num_genes=4384, d_model=128):
        super(MutationAttention, self).__init__()

        self.input_proj = nn.Linear(5, d_model)  # mutation_input에 적용

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

        # Q: mutation 정보 기반
        Q = self.input_proj(mutation_input)  # [batch, 4384, d_model]

        # K: 학습 파라미터
        K = self.key.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 4384, d_model]

        # V: 외부 입력 (co-occurrence vector), 배치 차원으로 broadcast
        V = external_V.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 4384, d_model]

        # Attention
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.bmm(attention_weights, V)  # [batch, 4384, d_model]

        weight_sum = attention_output.mean(dim=1)  # [batch, d_model]
        output = self.output_layer(weight_sum)

        return output.squeeze(-1)


class MutationAttentionExternalVTransformed(nn.Module):
    def __init__(self, num_genes=4384, d_model=128):
        super(MutationAttentionExternalVTransformed, self).__init__()

        self.input_proj = nn.Linear(4, d_model)  # mutation_input → Q

        self.v_proj = nn.Linear(d_model, d_model)  # 외부 V 선형 변환

        self.key = nn.Parameter(torch.randn(num_genes, d_model))  # K는 학습 파라미터

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 26)
        )

    def forward(self, mutation_input, external_V):
        """
        mutation_input: FloatTensor [batch_size, 4384, 4]
        external_V: FloatTensor [4384, d_model]
        """
        batch_size = mutation_input.size(0)

        # Q: mutation 정보에서 생성
        Q = self.input_proj(mutation_input)  # [batch, 4384, d_model]

        # K: 고정 학습 파라미터
        K = self.key.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 4384, d_model]

        # V: 외부 임베딩 → 선형변환
        V = self.v_proj(external_V)  # [4384, d_model]
        V = V.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 4384, d_model]

        # Attention
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.bmm(attention_weights, V)  # [batch, 4384, d_model]

        # Aggregate
        mean_output = attention_output.mean(dim=1)
        output = self.output_layer(mean_output)

        return output.squeeze(-1)
