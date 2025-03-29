from Model.CustomBPE import BPEModel


class AutoTokenizer:
    def __init__(self, truncation_side = "left", from_pretrained = True, return_tensors = True, max_tokens = 150):
        """
            input: truncate, from_pretrained, return_tensors, max_tokens
            output: tokenizer object
        """
        self.max_tokens = max_tokens
        self.truncate = truncation_side
        self.return_tensors = return_tensors
        self.from_pretrained = from_pretrained

        self.bpe_model = BPEModel(is_train= not from_pretrained)

    def __call__(self, text):
        """
            input: text
            output: tokenized tensor and attention masks
        """
        pass


    def encode(self, text):
        pass


    def decode(self):
        pass

