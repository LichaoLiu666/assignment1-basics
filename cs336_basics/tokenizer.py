# from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
import json
from functools import lru_cache

VOCAB_PATH = "/Users/lichaoliu/Documents/cs336/assignment1-basics/tests/fixtures/gpt2_vocab.json"
MERGES_PATH = "/Users/lichaoliu/Documents/cs336/assignment1-basics/tests/fixtures/gpt2_merges.txt"
GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # append special token to vocab
        vocab_len = len(vocab)
        # print(vocab)
        if special_tokens:
            for special_token in special_tokens:
                if special_token.encode("utf-8") not in vocab.values():
                    print(vocab_len - 1, special_token, special_token.encode("utf-8"))
                    vocab[vocab_len - 1] = special_token.encode("utf-8")
                    vocab_len += 1
        # init
        self.vocab = vocab
        self.reversed_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_rank = {merges[i]: i for i in range(len(merges))}
        self.merges_set = set(self.merges)
        self.special_tokens = special_tokens
        if special_tokens:
            self.special_tokens_pat = (
                "(" + "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            )
        else:
            self.special_tokens_pat = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        import pickle

        vocab = pickle.load(vocab_filepath)
        merges = pickle.load(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text):
        if self.special_tokens_pat:
            splited_text = re.split(self.special_tokens_pat, text)
        else:
            splited_text = [text]
        cleaned_txt = []
        for text in splited_text:
            if self.special_tokens and text in self.special_tokens:
                cleaned_txt.append(text)
            else:
                pretokenized = re.findall(GPT2_PRETOKENIZE_PATTERN, text)
                cleaned_txt += pretokenized

        out = []
        for element in cleaned_txt:
            byte_list = []
            if self.special_tokens and element in self.special_tokens:
                out.append(self.reversed_vocab[element.encode("utf-8")])
                continue
            else:
                byte_list += [bytes([c]) for c in element.encode("utf-8")]
            mergable_set = set([(byte_list[i], byte_list[i + 1]) for i in range(len(byte_list) - 1)])
            while mergable_set.intersection(self.merges_set):
                rank_list = []
                i = 0
                while i < len(byte_list) - 1:
                    if (byte_list[i], byte_list[i + 1]) in self.merges_rank:
                        rank = self.merges_rank[(byte_list[i], byte_list[i + 1])]
                        rank_list.append((rank, i))
                    i += 1
                sorted_rank_list = sorted(rank_list, key=lambda x: x[0])
                merge_i = sorted_rank_list[0][1]
                merged_bytes = b"".join([byte_list[merge_i], byte_list[merge_i + 1]])
                byte_list = byte_list[:merge_i] + [merged_bytes] + byte_list[merge_i + 2 :]
                mergable_set = set([(byte_list[i], byte_list[i + 1]) for i in range(len(byte_list) - 1)])

            for byte in byte_list:
                out.append(self.reversed_vocab[byte])
        return out

    def encode_iterable(self, iterable):
        results = []
        for line in iterable:
            result = self.encode(line)
            results += result
        return results

    def decode(self, ids):
        out = []
        for id in ids:
            byte = self.vocab[id]
            out.append(byte)
        text = b"".join(out).decode("utf-8", errors="replace")
        return text


# vocab_path = "/Users/lichaoliu/Documents/cs336/assignment1-basics/tests/fixtures/gpt2_vocab.json"
# merges_path = "/Users/lichaoliu/Documents/cs336/assignment1-basics/tests/fixtures/gpt2_merges.txt"


# @lru_cache
# def gpt2_bytes_to_unicode() -> dict[int, str]:
#     bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
#     cs = bs[:]
#     n = 0
#     for b in range(2**8):
#         if b not in bs:
#             # If this integer isn't in our list of visually-representable
#             # charcters, then map it to the next nice character (offset by 256)
#             bs.append(b)
#             cs.append(2**8 + n)
#             n += 1
#     characters = [chr(n) for n in cs]
#     d = dict(zip(bs, characters))
#     return d


# gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
# with open(vocab_path) as vocab_f:
#     gpt2_vocab = json.load(vocab_f)
# gpt2_bpe_merges = []
# with open(merges_path) as f:
#     for line in f:
#         cleaned_line = line.rstrip()
#         if cleaned_line and len(cleaned_line.split(" ")) == 2:
#             gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
# vocab = {
#     gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
#     for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
# }
# merges = [
#     (
#         bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
#         bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
#     )
#     for merge_token_1, merge_token_2 in gpt2_bpe_merges
# ]
# print(merges)
# t = tokenizer(vocab, merges)
# text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
# encoded_ids = t.encode(text)
# # tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
# print(encoded_ids)
# decoded = [tokenizer.decode([x]) for x in encoded_ids]
# print(decoded)


# test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
# encoded_ids = t.encode(test_string)
# tokenized_string = [t.decode([x]) for x in encoded_ids]
# assert tokenized_string.count("<|endoftext|>") == 3
