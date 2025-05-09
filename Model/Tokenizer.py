import torch


import re
from Model.BPE import BPEModel
# from CustomBPE.Model.BPE import BPEModel


class AutoTokenizer:
    def __init__(self, truncation_side = "right", from_pretrained = True, return_tensors = True, max_tokens = 150):
        """
            input: truncate, from_pretrained, return_tensors, max_tokens
            output: tokenizer object
            truncation side = "left" or "right"
            from_pretrained = True or False
            return_tensors = True or False
        """
        self.max_tokens = max_tokens
        self.truncate = truncation_side
        self.return_tensors = return_tensors
        self.from_pretrained = from_pretrained

        self.bpe_model = BPEModel(is_train = not from_pretrained)


    def __call__(self, texts):
        """
            input: text
            output: tokenized tensor and attention masks
        """
        return self.encode(texts) 
    
    def __repr__(self):
        """
            input: None
            output: tokenizer object
        """
        return f"Tokenizer(truncation_side={self.truncate}, from_pretrained={self.from_pretrained}, return_tensors={self.return_tensors}, max_tokens={self.max_tokens})"


    def encode(self, texts):
        """
            input: text
            output: tokenized tensor and attention masks
        """
        tokenized_texts = []
        attention_masks = []

        ### Tokenize the texts
        for text in texts:
            tokens = self.bpe_model.encode(text)
            tokenized_texts.append(tokens)
            attention_masks.append([0] * len(tokens))

        ### Truncate
        if self.truncate == "left":
            tokenized_texts = [tokens[-self.max_tokens:] for tokens in tokenized_texts]
            attention_masks = [mask[-self.max_tokens:] for mask in attention_masks]
        elif self.truncate == "right":
            tokenized_texts = [tokens[:self.max_tokens] for tokens in tokenized_texts]
            attention_masks = [mask[:self.max_tokens] for mask in attention_masks]


        ### Pad ###
        for i in range(len(tokenized_texts)):
            if len(tokenized_texts[i]) < self.max_tokens:
                attention_masks[i].extend([float('-inf')] * (self.max_tokens - len(tokenized_texts[i])))
                tokenized_texts[i].extend([self.bpe_model.inverse_vocab["<PAD>"]] * (self.max_tokens - len(tokenized_texts[i])))

        ### Convert to tensors ###
        if self.return_tensors:
            tokenized_texts = [torch.tensor(tokens) for tokens in tokenized_texts]
            attention_masks = [torch.tensor(mask) for mask in attention_masks]

            tokenized_texts = torch.stack(tokenized_texts)
            attention_masks = torch.stack(attention_masks)

        return tokenized_texts, attention_masks


    def batch_decode(self, tensors):
        """
            Clasical Decoding Function
        """
        decoded_texts = []
        for tensor in tensors:
            tokens = tensor.tolist()
            tokens = self.bpe_model.decode(tokens)
            decoded_texts.append(tokens)
        return decoded_texts


class MLMTokenizer(AutoTokenizer):
    def __init__(self, truncation_side="right", from_pretrained=True, return_tensors=True, max_tokens=150, mask_rate = 0.15):
        super().__init__(truncation_side, from_pretrained, return_tensors, max_tokens)

        self.mask_rate = mask_rate

    
    def encode(self, texts):
        tokenized_texts = []
        attention_masks = []
        targets = []

        ### Tokenize the texts
        for text in texts:
            original_tokens = self.bpe_model.encode(text)
            tokens, mlm_ids, unmasked_ids = self.bpe_model.prepare_mlm(original_tokens, self.mask_rate)
            tokenized_texts.append(tokens)
            attention_masks.append([1] * len(tokens))

            
            ### Create target ###
            target = [-100] * len(tokens)

            for i in range(len(mlm_ids)):
                target[mlm_ids[i]] = unmasked_ids[i]

            targets.append(target)


        ### Truncate
        if self.truncate == "left":
            tokenized_texts = [tokens[-self.max_tokens:] for tokens in tokenized_texts]
            attention_masks = [mask[-self.max_tokens:] for mask in attention_masks]
            targets = [target[-self.max_tokens:] for target in targets]
        elif self.truncate == "right":
            tokenized_texts = [tokens[:self.max_tokens] for tokens in tokenized_texts]
            attention_masks = [mask[:self.max_tokens] for mask in attention_masks]
            targets = [target[:self.max_tokens] for target in targets]


        ### Pad ###
        for i in range(len(tokenized_texts)):
            if len(tokenized_texts[i]) < self.max_tokens:
                targets[i].extend([-100] * (self.max_tokens - len(tokenized_texts[i])))
                attention_masks[i].extend([0] * (self.max_tokens - len(tokenized_texts[i])))
                tokenized_texts[i].extend([self.bpe_model.inverse_vocab["<PAD>"]] * (self.max_tokens - len(tokenized_texts[i])))

        ### Convert to tensors ###
        if self.return_tensors:
            tokenized_texts = [torch.tensor(tokens) for tokens in tokenized_texts]
            attention_masks = [torch.tensor(mask) for mask in attention_masks]
            targets = [torch.tensor(target) for target in targets]

            tokenized_texts = torch.stack(tokenized_texts)
            attention_masks = torch.stack(attention_masks)
            targets = torch.stack(targets)

        return tokenized_texts, attention_masks, targets