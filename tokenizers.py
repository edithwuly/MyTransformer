import re
from collections import Counter


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


if __name__ == "__main__":
    wp_sample()
