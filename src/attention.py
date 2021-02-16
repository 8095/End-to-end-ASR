import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, stacked_encoder=False, use_attn=True, attn_len=0, dropout_p=0):
        # attn_len 0 for full attention(Causal Dynamic Attention),
        # Else x_t-w, ..., x_t only used(Causeal Local Attention).
        super(AttentionModel, self).__init__()
        self.stacked_encoder = stacked_encoder if use_attn else True
        self.use_attn = use_attn
        self.attn_len = attn_len

        # Encoder
        self.feat = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.k_enc = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.q_enc = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Attention
        self.score = nn.Linear(hidden_size, hidden_size, bias=False)

        # Generator
        enhance_in = hidden_size * (2 if use_attn else 1)
        self.enhance = nn.Linear(enhance_in, hidden_size)
        self.mask = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x dim (B, T, F)
        input_x = x

        # Encoder
        x = self.feat(x).tanh()
        # TODO - Not sure it is good place to dropout
        x = self.dropout(x)
        self.k_enc.flatten_parameters()
        self.q_enc.flatten_parameters()
        k, _ = self.k_enc(x)
        q, _ = self.q_enc(k if self.stacked_encoder else x)

        # Attention
        out = q
        attn_weights = None
        if self.use_attn:
            # attn_score dim (B x T x T'(k))
            attn_score = torch.bmm(self.score(q), k.transpose(1, 2))
            attn_max, _ = torch.max(attn_score, dim=-1, keepdim=True)  # For numerial stability
            exp_score = torch.exp(attn_score - attn_max)

            # Causal contraints(score <= t)
            attn_weights = torch.tril(exp_score)
            if self.attn_len > 0:
                # Static constraints(t - w <= score)
                attn_weights = torch.triu(attn_weights, diagonal=-self.attn_len)
            weights_denom = torch.sum(attn_weights, dim=-1, keepdim=True)
            # attn_weights = attn_weights / (weights_denom + 1e-10)
            attn_weights = attn_weights / (weights_denom + 1e-30)

            c = torch.bmm(attn_weights, k)

            # concat query and context
            out = torch.cat((c, q), -1)

        # Generator
        out = self.enhance(out).tanh()
        out = self.mask(out).sigmoid()

        return input_x * out, attn_weights

# #  # attn_score dim (B x T x T'(k))
#             # 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3.
#             attn_score = torch.bmm(self.score(q), k.transpose(1, 2))
#
#             # output = torch.max(input, dim)：
#             # 输入：
#             # input是softmax函数输出的一个tensor
#             # dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
#             # 输出：函数会返回两个tensor，第一个tensor是每行的最大值，softmax的输出中最大的是1，所以第一个tensor是全1的tensor；第二个tensor是每行最大值的索引。
#             attn_max, _ = torch.max(attn_score, dim=-1, keepdim=True)  # For numerial stability为了数值稳定
#             exp_score = torch.exp(attn_score - attn_max)
#
#             # Causal contraints(score <= t)
#             # 返回一个张量，包含输入张量(2D张量)的下三角部分，其余部分设为0，参数diagonal控制对角线。
#             attn_weights = torch.tril(exp_score)
#             if self.attn_len > 0:
#                 # Static constraints(t - w <= score)
#                 # 返回一个张量，包含输入矩阵的上三角部分，其余被置为0。
#                 attn_weights = torch.triu(attn_weights, diagonal=-self.attn_len)
#             # 返回输入疑是给定维度上每行的和
#             weights_denom = torch.sum(attn_weights, dim=-1, keepdim=True)
#             # attn_weights = attn_weights / (weights_denom + 1e-10)
#             attn_weights = attn_weights / (weights_denom + 1e-30)
