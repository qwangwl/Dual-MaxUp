import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self, ):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, Q, K, V, mask=None):
        K_T = K.transpose(-1, -2) # 计算矩阵 K 的转置
        d_k = Q.size(-1)
        # 1, 计算 Q, K^T 矩阵的点积，再除以 sqrt(d_k) 得到注意力分数矩阵
        scores = torch.matmul(Q, K_T) / math.sqrt(d_k)
        # 2, 如果有掩码，则将注意力分数矩阵中对应掩码位置的值设为负无穷大
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 3, 对注意力分数矩阵按照最后一个维度进行 softmax 操作，得到注意力权重矩阵，值范围为 [0, 1]
        attn_weights = self.softmax(scores)
        # 4, 将注意力权重矩阵乘以 V，得到最终的输出矩阵
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer
    Args:
        d_model: Dimensions of the input embedding vector, equal to input and output dimensions of each head
        n_head: number of heads, which is also the number of parallel attention layers
    """
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model//8)  # Q 线性变换层
        self.w_k = nn.Linear(d_model, d_model//8)  # K 线性变换层
        self.w_v = nn.Linear(d_model, d_model)  # V 线性变换层
        #self.fc = nn.Linear(d_model, d_model)   # 输出线性变换层

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # size is [batch_size, seq_len, d_model]
        # 2, split by number of heads(n_head) # size is [batch_size, n_head, seq_len, d_model//n_head]
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3, compute attention
        sa_output, attn_weights = self.attention(q, k, v, mask)
        # 4, concat attention and linear transformation
        concat_tensor = self.concat(sa_output)
        #mha_output = self.fc(concat_tensor)

        return concat_tensor

    def split(self, tensor):
        """
        split tensor by number of head(n_head)

        :param tensor: [batch_size, seq_len, d_model]
        :return: [batch_size, n_head, seq_len, d_model//n_head], 输出矩阵是四维的，第二个维度是 head 维度

        # 将 Q、K、V 通过 reshape 函数拆分为 n_head 个头
        batch_size, seq_len, _ = q.shape
        q = q.reshape(batch_size, seq_len, n_head, d_model // n_head)
        k = k.reshape(batch_size, seq_len, n_head, d_model // n_head)
        v = v.reshape(batch_size, seq_len, n_head, d_model // n_head)
        """

        batch_size, seq_len, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        split_tensor = tensor.view(batch_size, seq_len, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return split_tensor

    def concat(self, sa_output):
        """ merge multiple heads back together

        Args:
            sa_output: [batch_size, n_head, seq_len, d_tensor]
            return: [batch_size, seq_len, d_model]
        """
        batch_size, n_head, seq_len, d_tensor = sa_output.size()
        d_model = n_head * d_tensor
        concat_tensor = sa_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return concat_tensor

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # '-1' means last dimension.
        var = x.var(-1, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta

        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_diff, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_diff)
        self.fc2 = nn.Linear(d_diff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

#这里的参数ratio是n_head
class Transformer_encoder(nn.Module):
    def __init__(self, n_channels, d_model, ratio, drop_prob=0.1):
        super(Transformer_encoder, self).__init__()
        self.mha = MultiHeadAttention(d_model, ratio)
        self.ffn = PositionwiseFeedForward(d_model, d_model//ratio)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)

        self.out_features = n_channels * d_model

    def forward(self, x, mask=None):
        x_residual1 = x

        # 1, compute multi-head attention
        x = self.mha(q=x, k=x, v=x, mask=mask)

        # 2, add residual connection and apply layer norm
        x = self.ln1(x_residual1 + self.dropout1(x))
        x_residual2 = x

        # 3, compute position-wise feed forward
        x = self.ffn(x)

        # 4, add residual connection and apply layer norm
        x = self.ln2(x_residual2 + self.dropout2(x))

        return x

if __name__ == "__main__":

    x = torch.randn(size = (10, 32, 128))

    net = Transformer_encoder(n_channels=32, d_model=128, ratio=4)

    print(net(x).shape)
    print(sum(param.numel() for param in net.parameters()))