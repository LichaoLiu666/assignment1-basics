from cs336_basics.tokenizer import tokenizer
import pickle
import random
import numpy as np
import time

if __name__ == "__main__":
    tasks = {"a1": False, "a2": False, "b": False, "c": False, "d1": True, "d2": False}
    # a1) owt tokenizer and text
    if tasks["a1"]:
        owt_path = "/Users/lichaoliu/Documents/cs336/assignment1-basics/data/owt_valid.txt"
        with open(owt_path, "rb") as f:
            text = f.read()
        owt_text_list = text.decode("utf-8").split("<|endoftext|>")
        owt_sampled_text = random.sample(owt_text_list, 10)
        with open("/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/owt_vocabs.pkl", "rb") as f1:
            owt_vocab = pickle.load(f1)
        with open("/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/owt_merges.pkl", "rb") as f2:
            owt_merges = pickle.load(f2)

        owt_tokenizer = tokenizer(owt_vocab, owt_merges)
        owt_compression_ratio = len("".join(owt_sampled_text).encode("utf-8")) / len(
            owt_tokenizer.encode_iterable(owt_sampled_text)
        )
        print(f"-------owt tokenizer compression ratio:{owt_compression_ratio}---------")

    # a2) tinystories tokenizer and text
    if tasks["a2"]:
        tinystories_path = "/Users/lichaoliu/Documents/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
        with open(tinystories_path, "rb") as f:
            text = f.read()
        tinystories_text_list = text.decode("utf-8").split("<|endoftext|>")
        tinystories_sampled_text = random.sample(tinystories_text_list, 10)
        with open(
            "/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/tinystories_vocabs.pkl", "rb"
        ) as f1:
            tinystories_vocab = pickle.load(f1)
        with open(
            "/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/tinystories_merges.pkl", "rb"
        ) as f2:
            tinystories_merges = pickle.load(f2)

        tinystories_tokenizer = tokenizer(tinystories_vocab, tinystories_merges)
        tinystories_compression_ratio = len("".join(tinystories_sampled_text).encode("utf-8")) / len(
            tinystories_tokenizer.encode_iterable(tinystories_sampled_text)
        )
        print(f"-------tinystories tokenizer compression ratio:{tinystories_compression_ratio}---------")
    # b) tinystories tokenizer and owt text
    if tasks["b"]:
        owt_path = "/Users/lichaoliu/Documents/cs336/assignment1-basics/data/owt_valid.txt"
        with open(owt_path, "rb") as f:
            text = f.read()
        owt_text_list = text.decode("utf-8").split("<|endoftext|>")
        owt_sampled_text = random.sample(owt_text_list, 10)
        with open(
            "/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/tinystories_vocabs.pkl", "rb"
        ) as f1:
            tinystories_vocab = pickle.load(f1)
        with open(
            "/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/tinystories_merges.pkl", "rb"
        ) as f2:
            tinystories_merges = pickle.load(f2)

        tinystories_tokenizer = tokenizer(tinystories_vocab, tinystories_merges)
        tinystories_owt_compression_ratio = len("".join(owt_sampled_text).encode("utf-8")) / len(
            tinystories_tokenizer.encode_iterable(owt_sampled_text)
        )
        print(
            f"-------tinystories tokenizer on owt sample compression ratio:{tinystories_owt_compression_ratio}---------"
        )
    # c) owt tokenizer and owt full text for throughput estimation
    if tasks["c"]:
        owt_path = "/Users/lichaoliu/Documents/cs336/assignment1-basics/data/owt_valid.txt"
        with open(owt_path, "rb") as f:
            text = f.read()
        owt_text_list = text.decode("utf-8").split("<|endoftext|>")
        owt_text_list = random.sample(owt_text_list, 1000)
        with open("/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/owt_vocabs.pkl", "rb") as f1:
            owt_vocab = pickle.load(f1)
        with open("/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/owt_merges.pkl", "rb") as f2:
            owt_merges = pickle.load(f2)

        owt_tokenizer = tokenizer(owt_vocab, owt_merges)
        start_ts = time.time()
        owt_tokenizer.encode_iterable(owt_text_list)
        owt_tokenizer_throughput = len("".join(owt_text_list).encode("utf-8")) / (time.time() - start_ts)
        print(f"-------owt tokenizer on full text - 12GB throughput:{owt_tokenizer_throughput} bytes/second---------")
    # d1) owt tokenizer and owt full text
    if tasks["d1"]:
        for data in ["valid", "train"]:
            input_path = f"/Users/lichaoliu/Documents/cs336/assignment1-basics/data/owt_{data}.txt"
            output_path = (
                f"/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/owt_token_ids_np_{data}.pkl"
            )
            with open(
                "/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/owt_vocabs.pkl", "rb"
            ) as f1:
                vocab = pickle.load(f1)
            with open(
                "/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/owt_merges.pkl", "rb"
            ) as f2:
                merges = pickle.load(f2)

            tok = tokenizer(vocab, merges)

            with open(input_path, "r", encoding="utf-8") as in_f, open(output_path, "wb") as out_f:
                buffer = []
                for line in in_f:
                    while "<|endoftext|>" in line:
                        before, line = line.split("<|endoftext|>", 1)
                        buffer.append(before)
                        doc = "".join(buffer)
                        if doc:
                            token_ids = tok.encode(doc)
                            token_ids_np = np.array(token_ids, dtype=np.uint16)
                            token_ids_np.tofile(out_f)
                        buffer = []
                    buffer.append(line)
                doc = "".join(buffer)
                if doc:
                    token_ids = tok.encode(doc)
                    token_ids_np = np.array(token_ids, dtype=np.uint16)
                    token_ids_np.tofile(out_f)
                print(f"owt {data} saved")
    if tasks["d2"]:
        for data in ["train"]:
            print(f"TinyStories {data}")
            input_path = f"/Users/lichaoliu/Documents/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-{data}.txt"
            output_path = f"/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/tinystories_token_ids_np_{data}.pkl"
            with open(
                "/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/tinystories_vocabs.pkl", "rb"
            ) as f1:
                vocab = pickle.load(f1)
            with open(
                "/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/tinystories_merges.pkl", "rb"
            ) as f2:
                merges = pickle.load(f2)

            tok = tokenizer(vocab, merges)
            with open(input_path, "r", encoding="utf-8") as in_f, open(output_path, "wb") as out_f:
                buffer = []
                for line in in_f:
                    while "<|endoftext|>" in line:
                        before, line = line.split("<|endoftext|>", 1)
                        buffer.append(before)
                        doc = "".join(buffer)
                        if doc:
                            token_ids = tok.encode(doc)
                            token_ids_np = np.array(token_ids, dtype=np.uint16)
                            token_ids_np.tofile(out_f)
                        buffer = []
                    buffer.append(line)
                doc = "".join(buffer)
                if doc:
                    token_ids = tok.encode(doc)
                    token_ids_np = np.array(token_ids, dtype=np.uint16)
                    token_ids_np.tofile(out_f)
                print(f"tinystories {data} saved")
