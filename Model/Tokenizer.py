from Model.CustomBPE import BPEModel


class AutoTokenizer:
    def __init__(self, truncate, return_tensors = True, max_tokens = 150):
        self.max_tokens = max_tokens
        self.truncate = truncate
        self.return_tensors = return_tensors
        self.bpe_model = BPEModel()

    def __call__(self, text):
        pass


    def decode(self):
        pass

