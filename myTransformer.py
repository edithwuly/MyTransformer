import torch


class AttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q_linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.k_linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.v_linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # q.shape = k.shape = v.shape = (batch_size, seq_len, dim)
        q, k, v = self.q_linear(hidden_states), self.k_linear(hidden_states), self.v_linear(hidden_states)
        weight = torch.matmul(q, k.transpose(1, 2))
        weight = torch.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        score = torch.matmul(weight, v)
        return score


class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.m_head = config.num_attention_heads

        self.query = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.key = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.value = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def _split_head(self, x):
        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size, seq_len, self.m_head, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        q, k, v = self.query(hidden_states), self.key(hidden_states), self.value(hidden_states)
        q, k, v = self._split_head(q), self._split_head(k), self._split_head(v)
        batch_size, m_head, seq_len, head_dim = v.shape

        weight = torch.matmul(q, k.transpose(2, 3)) / torch.sqrt(torch.tensor(head_dim))
        if attention_mask:
            weight = weight + attention_mask
        weight = torch.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        if head_mask:
            weight = weight * head_mask
        score = torch.matmul(weight, v)

        score = score.transpose(1, 2).contiguous().view(batch_size, seq_len, m_head * head_dim)
        return score


class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape, self.eps = normalized_shape, eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))

    def _mean(self, x):
        _shape = list(x.shape[:-len(self.normalized_shape)]) + [-1]
        _x = x.view(*_shape)
        mean = torch.sum(_x, dim=-1) / _x.shape[-1]
        for i in range(len(x.shape) - len(mean.shape)):
            mean = mean.unsqueeze(-1)
        return mean

    def forward(self, x):
        mean = self._mean(x)
        std = torch.sqrt(self._mean((x - mean).pow(2)))
        return self.weight * ((x - mean) / (std + self.eps)) + self.bias


class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.config = config
        self.attention = MultiHeadAttentionLayer(config)

        self.ffn = torch.nn.Sequential(torch.nn.Linear(config.hidden_size, config.hidden_size * 4),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(config.hidden_size * 4, config.hidden_size))

        self.norm1 = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.norm2 = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        attention_out = self.norm1(self.dropout(self.attention(x)))
        ffn_out = self.norm2(self.dropout(self.ffn(attention_out)))
        return ffn_out


def layernorm_sample():
    torch.manual_seed(999)
    x = torch.rand((3, 4, 6))
    normalized_shape = [4, 6]
    norm1 = LayerNorm(normalized_shape)
    norm2 = torch.nn.LayerNorm(normalized_shape)
    print(norm1(x))
    print(norm2(x))


class ExampleConfig():
    def __init__(self):
        self.num_attention_heads = 3
        self.layer_norm_eps = 1e-5
        self.resid_pdrop = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.hidden_size = 12
        self.hidden_dropout_prob = 0.1


def t_TransformerBlock():
    torch.manual_seed(999)
    config = ExampleConfig()
    trans = TransformerBlock(config)
    q = torch.rand((3, 4, config.hidden_size))
    r = trans(q)
    print(q)
    print(r)


if __name__ == "__main__":
    # layernorm_sample()
    t_TransformerBlock()
