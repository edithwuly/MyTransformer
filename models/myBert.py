import json
from collections import OrderedDict

import torch
from myTokenizers import BasicTokenizer, WordPieceTokenizer
from myTransformer import LayerNorm, MultiHeadAttentionLayer


class BertConfig():
    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 pad_token_id=0,
                 **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        for k, v in kwargs.items():
            setattr(self, k, v)


class BertTokenizer():
    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, tokenizer_chinese_chars=True):
        self.special_tokens = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        self.unk, self.sep, self.pad, self.cls, self.mask = self.special_tokens
        self.do_basic_tokenize = do_basic_tokenize
        self.vocab = self._load_vocab(vocab_file)
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case, self.special_tokens, tokenizer_chinese_chars)
        self.wordpiece_tokenizer = WordPieceTokenizer(vocab_size=len(self.vocab), lowercase=do_lower_case,
                                                      basic_tokenizer=lambda x: x.strip().split(),
                                                      unk=self.unk, sep=self.sep, pad=self.pad, cls=self.cls,
                                                      mask=self.mask)
        self.wordpiece_tokenizer.load(vocab=self.vocab)

    def _load_vocab(self, vocab_file):
        vocab = OrderedDict()
        for idx, token in enumerate(open(vocab_file, 'r').readlines()):
            vocab[token.rstrip('\n')] = idx
        return vocab

    def tokenize(self, text):
        tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.special_tokens):
                if token in self.special_tokens:
                    tokens.append(token)
                else:
                    tokens.extend(self.wordpiece_tokenizer.tokenize(token, add_post=None, add_mid="##", add_pre=None))
        else:
            tokens.extend(self.wordpiece_tokenizer.tokenize(text, add_post=None, add_mid="##", add_pre=None))

        return tokens

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens, ]
        return [self.vocab.get(token, self.vocab.get(self.unk)) for token in tokens]

    def encode_plus(self, text, text_pair=None, max_length=None, padding=True, truncation=True):
        text_ids = self.convert_tokens_to_ids(self.tokenize(text))
        text_pair_ids = self.convert_tokens_to_ids(self.tokenize(text_pair)) if text_pair else []

        ids_len = len(text_ids) + len(text_pair_ids) + 3 if text_pair_ids else len(text_ids) + 2
        if truncation and ids_len > max_length:
            # longest first
            for _ in range(ids_len - max_length):
                if len(text_ids) > len(text_pair_ids):
                    text_ids = text_ids[:-1]
                else:
                    text_pair_ids = text_pair_ids[:-1]

        # [cls] text1 [sep] text2 [sep]
        input_ids = self.convert_tokens_to_ids([self.cls]) + text_ids + self.convert_tokens_to_ids([self.sep])
        segment_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        if text_pair_ids:
            input_ids += text_pair_ids + self.convert_tokens_to_ids([self.sep])
            segment_ids += [1] * (len(text_pair_ids) + 1)
            attention_mask += [1] * (len(text_pair_ids) + 1)

        if padding:
            # max length
            while len(input_ids) < max_length:
                input_ids += self.convert_tokens_to_ids(self.pad)
                segment_ids += [0]
                attention_mask += [0]

        return {"input_ids": input_ids, "segment_ids": segment_ids, "attention_mask": attention_mask}


class BertSelfOutput(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertOutput(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttentionLayer(torch.nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = MultiHeadAttentionLayer(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states):
        attention_score = self.self(hidden_states)
        attention_output = self.output(attention_score, hidden_states)
        return attention_output


class BertIntermediaLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states



ACT2FN = {"gelu": torch.nn.GELU(), "relu": torch.nn.ReLU()}


class BertTransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super(BertTransformerBlock, self).__init__()
        self.config = config
        self.attention = BertAttentionLayer(config)
        self.intermediate = BertIntermediaLayer(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_out = self.attention(hidden_states)
        intermediate_out = self.intermediate(attention_out)
        output = self.output(intermediate_out, attention_out)
        return output


class BertEmbeddingLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        if position_ids == None:
            # absolute position
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long)
        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = self.dropout(self.LayerNorm(input_embeddings + position_embeddings + token_type_embeddings))
        return embeddings


class BertEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = torch.nn.ModuleList([BertTransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        for block in self.layer:
            hidden_states = block(hidden_states, attention_mask)

        return hidden_states


class BertPooler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(torch.nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.config = config
        self.embeddings = BertEmbeddingLayer(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        embedding_outputs = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        encoder_outputs = self.encoder(embedding_outputs, attention_mask)
        last_hidden_state = encoder_outputs[0]
        pooler_outputs = self.pooler(encoder_outputs)
        return (last_hidden_state, pooler_outputs)


class BertForSequenceClassification(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = BertModel(config)
        self.drop = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, segment_ids=None, position_ids=None):
        hidden_states, pooler_output = self.bert(input_ids, attention_mask, segment_ids, position_ids)
        pooled_output = self.drop(pooler_output)
        logits = self.classifier(pooled_output)
        return logits


def load_model(config, ckpt_path):
    model = BertForSequenceClassification(config)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model


def load_config(config_fn):
    _d = json.load(open(config_fn, 'r'))
    return BertConfig(**_d)


def sample_BertTokenizer():
    text = "“五一”小长假临近，30岁的武汉市民万昕在文旅博览会上获得了一些制定5天旅游计划的新思路。“‘壮美广西’‘安逸四川’，还有‘有一种叫云南的生活’这些展馆标识都很新颖，令人心向往之。”万昕说，感到身边越来越多的人走出家门去旅游。"
    # text = 'Say that thou didst forsake me for some fault, And I will comment upon that offence; Speak of my lameness, and I straight will halt, Against thy reasons making no defence.'

    tokenizer = BertTokenizer(vocab_file='../../models/bert-base-uncased/vocab.txt')
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    plus = tokenizer.encode_plus(text, max_length=100, padding=True, truncation=True)
    print("=" * 50, 'my')
    print(tokens)
    print(ids)
    print(plus)

    from transformers import BertTokenizer as OfficialBertTokenizer
    official_tokenizer = OfficialBertTokenizer(vocab_file='../../models/bert-base-uncased/vocab.txt')
    official_tokens = official_tokenizer.tokenize(text)
    official_ids = official_tokenizer.convert_tokens_to_ids(official_tokens)
    official_plus = official_tokenizer.encode_plus(text, max_length=100, padding='max_length', truncation='longest_first')
    print('=' * 50 + 'huggingface')
    print(official_tokens)
    print(official_ids)
    print(official_plus)

    assert tokens == official_tokens
    assert ids == official_ids
    assert plus['input_ids'] == official_plus['input_ids']
    assert plus['segment_ids'] == official_plus['token_type_ids']
    assert plus['attention_mask'] == official_plus['attention_mask']

    print('=' * 50 + 'special tokens')
    tokens = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
    print(tokenizer.convert_tokens_to_ids(tokens))
    print(official_tokenizer.convert_tokens_to_ids(tokens))


def sample_BertForSequenceClassification():
    config = load_config('../../models/bert-base-go-emotion/config.json')
    classes = config.id2label
    config.num_labels = len(config.id2label)
    print('classes: {}'.format(config.num_labels))

    model = load_model(config, '../../models/bert-base-go-emotion/pytorch_model.bin')
    tokenizer = BertTokenizer(vocab_file='../../models/bert-base-go-emotion/vocab.txt')
    query = 'I like you. I love you'
    tokens = tokenizer.encode_plus(query, padding=False, truncation=False)
    model.eval()
    output_ids = model(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0),
                       attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0),
                       segment_ids=torch.tensor(tokens['segment_ids']).unsqueeze(0))
    output_probs = torch.softmax(output_ids, dim=-1)
    pred_idx = torch.argmax(output_probs, dim=-1)
    print('max_pred: {}, max_prob: {}'.format(classes[str(pred_idx.item())], output_probs[0, pred_idx].item()))

    print('=' * 10, ' details ', '=' * 10)
    scores = [(i, prob) for i, prob in enumerate(output_probs.tolist()[0])]
    scores = sorted(scores, key=lambda x: -x[1])
    for i, prob in scores:
        print(classes[str(i)], prob)

if __name__ == "__main__":
    # sample_BertTokenizer()
    sample_BertForSequenceClassification()