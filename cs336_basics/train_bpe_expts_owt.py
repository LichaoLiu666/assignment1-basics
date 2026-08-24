from cs336_basics.train_bpe import train_bpe
import pickle

if __name__ == "__main__":
    input_path = "/Users/lichaoliu/Documents/cs336/assignment1-basics/data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    vocabs, merges = train_bpe(input_path, vocab_size, special_tokens)

    print("max_len_vocab:" + str(max([len(i) for i in vocabs.values()])))
    with open("/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/owt_vocabs.pkl", "wb") as f1:
        pickle.dump(vocabs, f1)
    print("vocabs saved")
    with open("/Users/lichaoliu/Documents/cs336/assignment1-basics/cs336_basics/output/owt_merges.pkl", "wb") as f2:
        pickle.dump(merges, f2)
    print("merges saved")
