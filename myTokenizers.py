import re
import unicodedata
from collections import Counter, OrderedDict


def wordpunct_tokenize(text):
    _pattern = r"\w+|[^\w\s]+"
    _regexp = re.compile(_pattern, flags=re.UNICODE | re.MULTILINE | re.DOTALL)
    return _regexp.findall(text)


class BPETokenizer():
    def __init__(self, vocab_size=2048, lowercase=True, basic_tokenizer=wordpunct_tokenize,
                unk='<UNK>', pad='<PAD>', end='<END>', mask='<MASK>', sep='<SEP>', cls='<CLS>', user_specials=None):
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.basic_tokenizer = basic_tokenizer
        self.unk, self.pad, self.end, self.mask, self.sep, self.cls = unk, pad, end, mask, sep, cls
        self.special = [unk, sep, pad, cls, mask]
        self.special.extend(user_specials if user_specials else [])

    def load(self, vocab_fn='vocab.txt', vocab=None):
        if vocab:
            self.vocab = vocab
        else:
            with open(vocab_fn, 'r') as fin:
                self.vocab = [w.strip() for w in fin.readlines()]
            self.vocab_size = len(self.vocab)

    def fit(self, corpus: list, max_steps=10000, out_fn='vocab.txt'):
        if self.lowercase:
            corpus = [s.lower() for s in corpus]

        token_list = []
        for data in corpus:
            data = wordpunct_tokenize(data)
            for token in data:
                token_list.append(tuple(token) + ("</w>",))

        word_corpus = Counter(token_list)
        vocab = self._count_vocab(word_corpus)

        for _ in range(max_steps):
            word_corpus, bi_cnt = self._fit_step(word_corpus)
            vocab = self._count_vocab(word_corpus)
            if bi_cnt < 0 or len(vocab) > self.vocab_size:
                break

        for s in self.special:
            if s not in vocab:
                vocab.insert(0, (s, 99999))

        with open(out_fn, 'w') as f:
            f.write('\n'.join([w for w, _ in vocab]))
        self.vocab = [token for token, _ in vocab]
        self.vocab_size = len(self.vocab)
        return vocab

    def _count_vocab(self, word_corpus):
        vocab_list = []
        for word, cnt in word_corpus.items():
            vocab_list.extend(word * cnt)

        vocab_list = Counter(vocab_list)
        vocab_list = sorted(vocab_list.items(), key=lambda x: -x[1])
        return vocab_list

    def _fit_step(self, word_corpus):
        ngram = 2
        bigram_counter = Counter()

        for tokens, count in word_corpus.items():
            if len(tokens) < 2:
                continue
            for i in range(len(tokens) - ngram + 1):
                bigram = tokens[i:i + ngram]
                bigram_counter[bigram] += count

        if len(bigram_counter) > 0:
            max_bigram = max(bigram_counter, key=bigram_counter.get)
        else:
            return word_corpus, -1

        bi_cnt = bigram_counter.get(max_bigram)

        words_tokens = list(word_corpus.keys())
        for tokens in words_tokens:
            _new_tokens = tuple(' '.join(tokens).replace(' '.join(max_bigram), ''.join(max_bigram)).split(' '))
            if _new_tokens != tokens:
                word_corpus[_new_tokens] = word_corpus[tokens]
                word_corpus.pop(tokens)

        return word_corpus, bi_cnt

    def tokenize(self, text: str, add_pre=None, add_mid=None, add_post='</w>'):
        all_tokens = []
        if self.lowercase:
            text = text.lower()

        for token in self.basic_tokenizer(text):
            if add_pre:
                token = add_pre + token
            if add_post:
                token = token + add_post

            start, end = 0, len(token)
            while start < end:
                sub_token = token[start:end]
                if start > 0 and add_mid:
                    sub_token = add_mid + sub_token
                if sub_token in self.vocab:
                    all_tokens.append(sub_token)
                    start = end
                    end = len(token)
                elif end - start == 1:
                    all_tokens.append(self.unk)
                    start = end
                    end = len(token)
                else:
                    end -= 1
        return all_tokens

    def _token2id(self, token):
        if token in self.vocab:
            return self.vocab.index(token)
        return self.vocab.index(self.unk)

    def _id2token(self, id):
        return self.vocab[id]

    def encode(self, sentences):
        token_ids = []
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            token_ids.append([self._token2id(token) for token in tokens])
        return token_ids

    def decode(self, token_ids):
        sentences = []
        for token_id in token_ids:
            tokens = [self._id2token(id) for id in token_id]
            sentence = ''.join(tokens).replace("</w>", ' ')
            sentences.append(sentence)
        return sentences


class WordPieceTokenizer(BPETokenizer):
    def _fit_step(self, word_corpus):
        ngram = 2
        bigram_counter = Counter()
        unbigram_counter = Counter()

        for tokens, count in word_corpus.items():
            for token in tokens:
                unbigram_counter[token] += count
            if len(tokens) < 2:
                continue
            for i in range(len(tokens) - ngram + 1):
                bigram = tokens[i:i + ngram]
                bigram_counter[bigram] += count

        if len(bigram_counter) > 0:
            max_bigram = max(bigram_counter, key=lambda x: bigram_counter.get(x) / (unbigram_counter.get(x[0]) * unbigram_counter.get(x[1])))
        else:
            return word_corpus, -1

        bi_cnt = bigram_counter.get(max_bigram)

        words_tokens = list(word_corpus.keys())
        for tokens in words_tokens:
            _new_tokens = tuple(' '.join(tokens).replace(' '.join(max_bigram), ''.join(max_bigram)).split(' '))
            if _new_tokens != tokens:
                word_corpus[_new_tokens] = word_corpus[tokens]
                word_corpus.pop(tokens)

        return word_corpus, bi_cnt


class BasicTokenizer():
    def __init__(self, do_lower_case=False, never_split=None, tokenize_chinese_chars=True, strip_accents=True):
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split) if never_split else set()
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)

        original_tokens = text.strip().split()
        split_tokens = []
        for token in original_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                if self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))
        output_tokens = " ".join(split_tokens).strip().split()
        return output_tokens

    def _is_control(self, char):
        if char == "\t" or char == "\n" or char == "\r":  # treat them as whitespace
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def _is_whitespace(self, char):
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or self._is_control(char):
                continue
            output.append(" " if self._is_whitespace(char) else char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        if (cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF) or \
                (cp >= 0x20000 and cp <= 0x2A6DF) or (cp >= 0x2A700 and cp <= 0x2B73F) or \
                (cp >= 0x2B740 and cp <= 0x2B81F) or (cp >= 0x2B820 and cp <= 0x2CEAF) or \
                (cp >= 0xF900 and cp <= 0xFAFF) or (cp >= 0x2F800 and cp <= 0x2FA1F):
            return True
        return False

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            output.append(" {} ".format(char) if self._is_chinese_char(cp) else char)
        return "".join(output)

    def _run_strip_accents(self, token):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", token)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _is_punctuation(self, char):
        # We treat all non-letter/number ASCII as punctuation.
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _run_split_on_punc(self, token, never_split):
        """Splits punctuation on a piece of text."""
        if token in never_split:
            return [token]
        output = [[]]
        for char in list(token):
            if self._is_punctuation(char):
                output.append([char])
                output.append([])
            else:
                output[-1].append(char)
        return ["".join(x) for x in output]


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


def bpe_sample():
    corpus = '''
            Object raspberrypi functools dict kwargs. Gevent raspberrypi functools. Dunder raspberrypi decorator dict didn't lambda zip import pyramid, she lambda iterate?
            Kwargs raspberrypi diversity unit object gevent. Import fall integration decorator unit django yield functools twisted. Dunder integration decorator he she future. Python raspberrypi community pypy. Kwargs integration beautiful test reduce gil python closure. Gevent he integration generator fall test kwargs raise didn't visor he itertools...
            Reduce integration coroutine bdfl he python. Cython didn't integration while beautiful list python didn't nit!
            Object fall diversity 2to3 dunder script. Python fall for: integration exception dict kwargs dunder pycon. Import raspberrypi beautiful test import six web. Future integration mercurial self script web. Return raspberrypi community test she stable.
            Django raspberrypi mercurial unit import yield raspberrypi visual rocksdahouse. Dunder raspberrypi mercurial list reduce class test scipy helmet zip?
        '''
    corpus = [s.strip() for s in corpus.strip().split('\n')]
    bpe = BPETokenizer(vocab_size=60)
    bpe.fit(corpus)

    bpe.load()
    print(bpe.vocab)
    # ['<MASK>', '<CLS>', '<PAD>', '<SEP>', '<UNK>', '</w>', 'i', 'd', 'c', 'e', 'a', 't</w>', 't', 'l', 'o', 's', 'e</w>', 'm', 'h', 'f', '.</w>', 'u', 'un', 'l</w>', 'b', 'g', 'r', 'or', 'p', 'er', 's</w>', 'raspberrypi</w>', 'w', 'di', 'py', 'integration</w>', 'n', 'y', 'v', 'on</w>', 'st</w>', 'ur', 'k', 'te', 'j', 'n</w>', "'", 'ti', 'ora', 'z', '?', 'x', 'ra', '.', ',', '!', '2', '3', ':', 'tera', 'te</w>', 'era', 'ter', 'tion</w>', 'pypy', 'tin']
    print(bpe.tokenize("Vizzini: He didn't fall? INCONCEIVABLE!"))
    # ['v', 'i', 'z', 'z', 'i', 'n', 'i', '</w>', ':', '</w>', 'h', 'e</w>', 'di', 'd', 'n</w>', "'", '</w>', 't</w>', 'f', 'a', 'l', 'l</w>', '?', '</w>', 'i', 'n', 'c', 'o', 'n', 'c', 'e', 'i', 'v', 'a', 'b', 'l', 'e</w>', '!', '</w>']
    print(bpe.encode(["Vizzini: He didn't fall? INCONCEIVABLE!"]))
    # [[38, 6, 49, 49, 6, 36, 6, 5, 58, 5, 18, 16, 33, 7, 45, 46, 5, 11, 19, 10, 13, 23, 50, 5, 6, 36, 8, 14, 36, 8, 9, 6, 38, 10, 24, 13, 16, 55, 5]]
    print(bpe.decode(bpe.encode(["Vizzini: He didn't fall? INCONCEIVABLE!"])))
    # ["vizzini : he didn ' t fall ? inconceivable ! "]


def wp_sample():
    corpus = '''
            Object raspberrypi functools dict kwargs. Gevent raspberrypi functools. Dunder raspberrypi decorator dict didn't lambda zip import pyramid, she lambda iterate?
            Kwargs raspberrypi diversity unit object gevent. Import fall integration decorator unit django yield functools twisted. Dunder integration decorator he she future. Python raspberrypi community pypy. Kwargs integration beautiful test reduce gil python closure. Gevent he integration generator fall test kwargs raise didn't visor he itertools...
            Reduce integration coroutine bdfl he python. Cython didn't integration while beautiful list python didn't nit!
            Object fall diversity 2to3 dunder script. Python fall for: integration exception dict kwargs dunder pycon. Import raspberrypi beautiful test import six web. Future integration mercurial self script web. Return raspberrypi community test she stable.
            Django raspberrypi mercurial unit import yield raspberrypi visual rocksdahouse. Dunder raspberrypi mercurial list reduce class test scipy helmet zip?
        '''
    corpus = [s.strip() for s in corpus.strip().split('\n')]
    wp = WordPieceTokenizer(vocab_size=60)
    wp.fit(corpus)

    print(wp.vocab)
    # ['<MASK>', '<CLS>', '<PAD>', '<SEP>', '<UNK>', '</w>', 'r', 't', 'e', 'i', 'n', 'd', 'o', 's', 'u', 'a', 'c', '.', 'g', 'l', 'aspbe', 'ypi', 'y', 'py', 'hon', 'kwa', 'impo', "'", 'fall', 'he', 'b', 'iful', 'func', 'objec', 'me', 'she', 'geven', '?', 'django', 'fu', 'lambda', 'zip', 'communi', 'ip', 'dive', 'web', ',', '!', ':', '2', 'o3', 'bdfl', 'fo', 'pypy', 'six', 'scipy', 'ocksdahouse', 'self', 'while', 'excep', 'amid', 'helme', 'visual', 'viso', 'wis', 'bl']
    print(wp.tokenize("Vizzini: He didn't fall? INCONCEIVABLE!"))
    # ['<UNK>', 'i', '<UNK>', '<UNK>', 'i', 'n', 'i', '</w>', ':', '</w>', 'he', '</w>', 'd', 'i', 'd', 'n', '</w>', "'", '</w>', 't', '</w>', 'fall', '</w>', '?', '</w>', 'i', 'n', 'c', 'o', 'n', 'c', 'e', 'i', '<UNK>', 'a', 'bl', 'e', '</w>', '!', '</w>']
    print(wp.encode(["Vizzini: He didn't fall? INCONCEIVABLE!"]))
    # [[4, 9, 4, 4, 9, 10, 9, 5, 48, 5, 29, 5, 11, 9, 11, 10, 5, 27, 5, 7, 5, 28, 5, 37, 5, 9, 10, 16, 12, 10, 16, 8, 9, 4, 15, 65, 8, 5, 47, 5]]
    print(wp.decode(wp.encode(["Vizzini: He didn't fall? INCONCEIVABLE!"])))
    # ["<UNK>i<UNK><UNK>ini : he didn ' t fall ? inconcei<UNK>able ! "]


def sample_BertTokenizer():
    text = "“五一”小长假临近，30岁的武汉市民万昕在文旅博览会上获得了一些制定5天旅游计划的新思路。“‘壮美广西’‘安逸四川’，还有‘有一种叫云南的生活’这些展馆标识都很新颖，令人心向往之。”万昕说，感到身边越来越多的人走出家门去旅游。"
    # text = 'Say that thou didst forsake me for some fault, And I will comment upon that offence; Speak of my lameness, and I straight will halt, Against thy reasons making no defence.'

    tokenizer = BertTokenizer(vocab_file='../models/bert-base-uncased/vocab.txt')
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    plus = tokenizer.encode_plus(text, max_length=100, padding=True, truncation=True)
    print("=" * 50, 'my')
    print(tokens)
    print(ids)
    print(plus)

    from transformers import BertTokenizer as OfficialBertTokenizer
    official_tokenizer = OfficialBertTokenizer(vocab_file='../models/bert-base-uncased/vocab.txt')
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


if __name__ == "__main__":
    # bpe_sample()
    # wp_sample()
    sample_BertTokenizer()
