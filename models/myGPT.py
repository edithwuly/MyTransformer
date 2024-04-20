import math
from myTransformer import LayerNorm
import torch


class GPTConfig():
    def __init__(self, vocab_size=100, n_embd=100, n_positions=100, n_layer=3, n_head=2, n_ctx=2000,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1, layer_norm_epsilon=1e-5,
                 afn='gelu_new',
                 **kwargs):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.hidden_size = n_embd
        self.n_positions = n_positions
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.max_position_embeddings = n_ctx
        self.embd_pdrop, self.attn_pdrop, self.resid_pdrop = embd_pdrop, attn_pdrop, resid_pdrop
        self.attention_probs_dropout_prob = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.activation_function = afn
        self.scale_attn_weights = True
        for k, v in kwargs.items():
            setattr(self, k, v)


class Conv1D(torch.nn.Module):
    def __init__(self, out_dim, input_dim):
        super(Conv1D, self).__init__()

        w = torch.empty(input_dim, out_dim)
        torch.nn.init.normal_(w, std=0.02)
        self.weight = torch.nn.Parameter(w)
        self.bias = torch.nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.transpose(0, 1), self.bias)


class AttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scale_attn_weights = config.scale_attn_weights
        max_positions = config.max_position_embeddings
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = torch.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = torch.nn.Dropout(config.resid_pdrop)
        self.is_causal = True

    def _split_m_head(self, x):
        b, s, d = x.shape
        x = x.view(b, s, self.num_heads, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        hidden_states = self.c_attn(hidden_states)
        query, key, value = hidden_states.split(self.embed_dim, dim=-1)
        query, key, value = self._split_m_head(query), self._split_m_head(key), self._split_m_head(value)

        weight = torch.matmul(query, key.transpose(2, 3))
        if self.scale_attn_weights:
            weight = weight / math.sqrt(value.size(-1))

        if self.is_causal:
            # casual mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
            mask_value = torch.finfo(weight.dtype).min
            mask_value = torch.full([], mask_value, dtype=weight.dtype, device=weight.device)
            weight = torch.where(causal_mask, weight.to(weight.dtype), mask_value)

        if attention_mask is not None:
            weight = weight + attention_mask
        weight = torch.softmax(weight, dim=-1)
        weight = self.attn_dropout(weight)
        if head_mask:
            weight = weight * head_mask

        score = torch.matmul(weight, value)

        batch_size, m_head, seq_len, head_dim = value.shape
        score = score.transpose(1, 2).contiguous().view(batch_size, seq_len, m_head * head_dim)
        score = self.resid_dropout(self.c_proj(score))
        return score


class NewGELUActivation(torch.nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


ACT2FN = {"gelu": torch.nn.GELU, "relu": torch.nn.ReLU, 'gelu_new': NewGELUActivation}


class GPT2MLP(torch.nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]()
        self.dropout = torch.nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = 4 * hidden_size

        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = AttentionLayer(config=config)
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask=attention_mask, head_mask=head_mask)
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


class GPT2Model(torch.nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.embed_dim = config.hidden_size

        self.wte = torch.nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = torch.nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = torch.nn.Dropout(config.embd_pdrop)
        self.h = torch.nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, attention_mask=None):
        inputs_embeds = self.wte(input_ids)
        if position_ids is None:
            # absolute position
            position_ids = torch.arange(input_ids.size()[-1], dtype=torch.long)
            position_ids = position_ids.unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        return self.ln_f(hidden_states)


class GPTLMHeadModel(torch.nn.Module):
    def __init__(self, config):
        super(GPTLMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)
        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits, hidden_states)
        return outputs


def sample_block():
    config = GPTConfig()
    block = GPT2Block(GPTConfig())
    x = torch.randn(2, 3, config.n_embd)
    print(block(x))


def sample_gpt():
    config = GPTConfig()
    gpt = GPT2Model(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 3))
    token_type_ids = torch.randint(0, config.vocab_size, (2, 3))
    position_ids = torch.randint(0, config.n_positions, (2, 3))
    print(gpt(input_ids, position_ids=position_ids, token_type_ids=token_type_ids))


def sample_gpt_lmhead():
    config = GPTConfig()
    gpt = GPTLMHeadModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 3))
    token_type_ids = torch.randint(0, config.vocab_size, (2, 3))
    position_ids = torch.randint(0, config.n_positions, (2, 3))
    print(gpt(input_ids, position_ids=position_ids, token_type_ids=token_type_ids))


def sample_attention():
    config = GPTConfig()
    attention = AttentionLayer(config)
    x = torch.randn(2, 3, config.n_embd)
    print(attention(x))


if __name__ == '__main__':
    # sample_attention()
    # sample_block()
    # sample_gpt()
    sample_gpt_lmhead()
