import pandas as pd
from tqdm import tqdm

from Model.BPE import BPEModel
from Model.Tokenizer import AutoTokenizer



if __name__ == "__main__":
    data = pd.read_csv("Data/BookCorpus3.csv")
    bpe_model = BPEModel(is_train = True, max_vocab_size=35000)
    batch_size = 32

    for i in tqdm(range(0, data.shape[0], batch_size)):
        
        if len(bpe_model.vocab) >= bpe_model.max_vocab_size:
            break

        text = [data.iloc[j,0] for j in range(i, i+batch_size)]
        text = " ".join(text)

        bpe_model.train(text, special_tokens = ["<CLS>", "<SEP>", "<MASK>", "<PAD>", "<EOS>"])








