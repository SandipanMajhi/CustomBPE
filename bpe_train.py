import pandas as pd
from tqdm import tqdm

from Model.CustomBPE import BPEModel
from Model.Tokenizer import AutoTokenizer



if __name__ == "__main__":
    data = pd.read_csv("Data/BookCorpus3.csv")
    bpe_model = BPEModel(is_train = True)
    batch_size = 1000

    for i in tqdm(range(0, data.shape[0], batch_size)):
        text = [data.iloc[j,0] for j in range(i, i+batch_size)]
        text = " ".join(text)
        bpe_model.train(text, special_tokens = ["[CLS]", "[SEP]", "[MASK]", "[PAD]"])

        bpe_model.load_inverse_vocab()
        bpe_model.load_merge_rules()
        bpe_model.load_vocab()









