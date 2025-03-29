import pandas as pd
from tqdm import tqdm

from Model.CustomBPE import BPEModel
from Model.Tokenizer import AutoTokenizer







if __name__ == "__main__":
    data = pd.read_csv("Data/BookCorpus3.csv")
    bpe_model = BPEModel(is_train = True)









