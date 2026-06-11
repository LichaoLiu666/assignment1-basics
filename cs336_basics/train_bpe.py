from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
import regex as re
from collections import Counter, defaultdict
import time
import heapq


def process_chunk(chunk_cleaned: str) -> dict[tuple[str], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # for special_token in special_tokens:
    #     chunk_cleaned = re.sub(special_token,"",chunk)
    pre_tokenized_chunk_cleaned = re.findall(PAT, chunk_cleaned)
    return Counter(pre_tokenized_chunk_cleaned)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges = []
    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode(("utf-8"))
    merged_element = None
    chunk_list = []
    num_processes = 32
    word_tuple_dict = defaultdict(int)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pat = "|".join(re.escape(t) for t in special_tokens)
            chunk = re.split(pat, chunk)
            chunk_list += chunk
    start = time.time()
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunk_list)
    # print("preprocessing time:" + str(time.time() - start))
    for result in results:
        for key, value in result.items():
            # new_key = tuple(bytes([byte]) for element in key for byte in element.encode("utf-8"))
            new_key_list = []
            for element in key:
                bytes_element = element.encode("utf-8")
                # print(bytes_element[0])
                if len(bytes_element) == 1:
                    new_key_list.append(bytes_element)
                else:
                    new_key_list += [bytes([x]) for x in bytes_element]
            new_key = tuple(new_key_list)
            word_tuple_dict[new_key] = word_tuple_dict.get(new_key, 0) + value
    # print(word_tuple_dict)
    merge_dict = defaultdict(int)
    pair_word_dict = defaultdict(set)
    start = time.time()
    while len(vocab) < vocab_size:
        tmpDict = {}
        # first iteration
        if merged_element is None:
            for key, value in word_tuple_dict.items():
                for i in range(len(key) - 1):
                    merge_dict[(key[i], key[i + 1])] += value
                    pair_word_dict[(key[i], key[i + 1])].add(key)
        # highest freq
        max_freq = max(merge_dict.values())
        sorted_d_top_k = [k for k, v in merge_dict.items() if v == max_freq]
        sorted_d_top_k_lexicographically = sorted(sorted_d_top_k, reverse=True)
        merged_element = sorted_d_top_k_lexicographically[0]
        merges.append(merged_element)

        # add to vocab
        vocab[len(vocab)] = b"".join(merged_element)

        for key in pair_word_dict[merged_element].copy():
            value = word_tuple_dict[key]
            tmp = []
            i = 0
            while i < len(key):
                # print(key)
                if i + 1 < len(key) and merged_element is not None and (key[i], key[i + 1]) == merged_element:
                    if i > 0:
                        old_pair = (key[i - 1], merged_element[0])
                        new_pair = (key[i - 1], b"".join(merged_element))
                        merge_dict[old_pair] -= value
                        merge_dict[new_pair] += value

                    if i < len(key) - 2:
                        old_pair = (merged_element[1], key[i + 2])
                        new_pair = (b"".join(merged_element), key[i + 2])
                        merge_dict[old_pair] -= value
                        merge_dict[new_pair] += value
                    tmp.append(b"".join(merged_element))
                    i += 2
                    for j in range(len(key) - 1):
                        pair_word_dict[(key[j], key[j + 1])].discard(key)
                else:
                    tmp.append(key[i])
                    i += 1
            newKey = tuple(tmp)
            for k in range(len(newKey) - 1):
                pair_word_dict[(newKey[k], newKey[k + 1])].add(newKey)
            tmpDict[key] = newKey
            if merged_element in merge_dict.keys():
                del merge_dict[merged_element]
            if merged_element in pair_word_dict.keys():
                del pair_word_dict[merged_element]
        for oldKey, newKey in tmpDict.items():
            word_tuple_dict[newKey] = word_tuple_dict.pop(oldKey)

    return vocab, merges


if __name__ == "__main__":
    print(
        train_bpe(
            # input_path="/Users/lichaoliu/Documents/cs336/assignment1-basics/tests/fixtures/corpus.en",
            input_path="/Users/lichaoliu/Documents/cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt",
            vocab_size=300,
            special_tokens=["<|endoftext|>"],
        )
    )
