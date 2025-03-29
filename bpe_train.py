import pandas as pd
from tqdm import tqdm

from Model.CustomBPE import BPEModel
from Model.Tokenizer import AutoTokenizer







if __name__ == "__main__":
    data = pd.read_csv("Data/BookCorpus3.csv")
    bpe_model = BPEModel(is_train = True)

    for i in tqdm(range(data.shape[0])):
        text = data.iloc[i]["text"]
        bpe_model.train(text, vocab_size = 50000, special_tokens = ["[CLS]", "[SEP]", "[MASK]", "[PAD]"])

        bpe_model.load_inverse_vocab()
        bpe_model.load_merge_rules()
        bpe_model.load_vocab()









