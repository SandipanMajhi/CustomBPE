import os
import random
import pickle as pkl
from collections import Counter, deque

import re
import numpy as np

import unicodedata

class BPEModel:
    def __init__(self, is_train = False, max_vocab_size = 50000):
        self.vocab = {}
        self.merge_rules = {}
        self.inverse_vocab = {}

        self.max_vocab_size = max_vocab_size

        # self.save_root = "CustomBPE/Data"
        self.save_root = "Data"
        self.save_vocab_path = f"{self.save_root}/bpe_vocab.pkl"
        self.save_merge_path = f"{self.save_root}/bpe_merge.pkl"
        self.inverse_vocab_path = f"{self.save_root}/bpe_inverse_vocab.pkl"

        self.is_train = is_train

        if not self.is_train:
            self.load_vocab()
            self.load_merge_rules()
            self.load_inverse_vocab()

    
    def load_vocab(self):
        with open(self.save_vocab_path, 'rb') as f:
            self.vocab = pkl.load(f)
        

    def load_merge_rules(self):
        with open(self.save_merge_path, 'rb') as f:
            self.merge_rules = pkl.load(f)
    
    def load_inverse_vocab(self):
        with open(self.inverse_vocab_path, 'rb') as f:
            self.inverse_vocab = pkl.load(f)


    def load_text(self):
        if self.training_text_path is not None:
            with open(self.training_text_path, 'r', encoding = 'utf-8') as f:
                text = f.read()
        return text


    def train(self, text, special_tokens = ["<CLS>", "<SEP>", "<MASK>", "<PAD>"]):

        batch_merge_rules = {}

        preprocess_text = []
        for i in range(len(text)):
            if text[i] == " " and i != 0:
                preprocess_text.append("#")

            if text[i] != " ":
                preprocess_text.append(text[i])

        preprocess_text = "".join(preprocess_text)
        unique_chars = [chr(i) for i in range(256)]

        for character in sorted(set(preprocess_text)):
            if character not in unique_chars:
                unique_chars.append(character)

        if "#" not in unique_chars:
            unique_chars.append("#")

        unique_chars.extend(special_tokens)

        if os.path.exists(self.save_vocab_path):
            self.load_inverse_vocab()
            self.load_merge_rules()
            self.load_vocab()

        for char in unique_chars:
            if char not in self.inverse_vocab:
                idx = len(self.vocab)
                self.vocab[idx] = char
                self.inverse_vocab[char] = idx 

        token_ids = self.encode(text)
        new_idx  = len(self.vocab)
        
        while(new_idx < self.max_vocab_size):
            
            most_freq_pairs_idx = self.find_most_frequent(token_ids=token_ids)
            if most_freq_pairs_idx is None:
                break

            if most_freq_pairs_idx in self.merge_rules:
                token_ids = self.replace_pairs(token_ids, most_freq_pairs_idx, self.merge_rules[most_freq_pairs_idx])

            if most_freq_pairs_idx not in self.merge_rules:

                old_0, old_1 = most_freq_pairs_idx
                if old_0 in self.vocab and old_1 in self.vocab:
                    if self.vocab[old_0] + self.vocab[old_1] in self.inverse_vocab:

                        self.merge_rules[(old_0, old_1)] = self.inverse_vocab[self.vocab[old_0] + self.vocab[old_1]]
                        token_ids = self.replace_pairs(token_ids, most_freq_pairs_idx, self.merge_rules[(old_0, old_1)])

                    else:

                        token_ids = self.replace_pairs(token_ids, most_freq_pairs_idx, new_idx)
                        self.merge_rules[(old_0, old_1)] = new_idx
                        self.vocab[new_idx] = self.vocab[old_0] + self.vocab[old_1]
                        self.inverse_vocab[self.vocab[old_0] + self.vocab[old_1]] = new_idx
                        new_idx += 1

        self.save()
    

    def find_most_frequent(self, token_ids):
        pairs = Counter(zip(token_ids, token_ids[1:]))
        max_term, max_val = max(pairs.items(), key = lambda x : x[1])
        if max_val > 1:
            return max_term
        else:
            return None
    

    def replace_pairs(self, token_ids, pairs_idx, new_idx):
        token_dq = deque(token_ids)
        replaced = []

        while token_dq:
            left_most = token_dq.popleft()
            if token_dq and (left_most, token_dq[0]) == pairs_idx:
                replaced.append(new_idx)
                token_dq.popleft()
            else:
                replaced.append(left_most)

        return replaced
    
    def encode(self, text):
        tokens = []

        #### Temporary fix for special characters ####
        text = re.sub(r"’", "'", text)


        words = text.replace("\n", " \n ").split(' ')

        for i, word in enumerate(words):
            if i > 0 and not word.startswith("\n"):
                tokens.append("#" + word)
            else:
                tokens.append(word)

        token_ids = []

        for token in tokens:
            if token in self.inverse_vocab:
                # token_ids.append(self.inverse_vocab[token])
                token_ids.append(self.inverse_vocab[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))

        return token_ids


    def tokenize_with_bpe(self, token):
        
        token_ids = [self.inverse_vocab[char] for char in token]
        

        can_merge = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.merge_rules:
                    new_tokens.append(self.merge_rules[pair])
                    can_merge = True
                    i += 2
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            if i < len(token_ids):
                new_tokens.append(token_ids[i])

            token_ids = new_tokens

        return token_ids
    

    def decode(self, token_ids, is_special = True):
        decoded_string = ""

        for token_id in token_ids:
            token = self.vocab[token_id]
            if token.startswith("#"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token

        decoded_string = decoded_string.replace("#", " ")

        if not is_special:
            decoded_string = decoded_string.replace("<CLS>", "")
            decoded_string = decoded_string.replace("<SEP>", "")
            decoded_string = decoded_string.replace("<MASK>", "")
            decoded_string = decoded_string.replace("<PAD>", "")
        
        return decoded_string
    

    def save(self):

        with open(self.save_vocab_path, 'wb') as f:
            pkl.dump(self.vocab, f)
        with open(self.save_merge_path, 'wb') as f:
            pkl.dump(self.merge_rules, f)
        with open(self.inverse_vocab_path, 'wb') as f:
            pkl.dump(self.inverse_vocab, f)


    def prepare_mlm_seq(self, text, mask_rate = 0.15):
        encoded = self.encode(text)
        mlm_count = int(len(encoded) * mask_rate)
        mlm_idx = random.choices(range(len(encoded)), k = mlm_count)

        for idx in mlm_idx:
            if idx != 0:
                encoded[idx] = self.inverse_vocab["<MASK>"] 

        return encoded
    


    def prepare_mlm(self, tokens, mask_rate = 0.15):
        
        indices = np.random.choice(range(len(tokens)), size = int(len(tokens) * mask_rate), replace = False)
        mlm_index = np.random.choice(indices, size = int(indices.shape[0]), replace = False) 
        mlm_index = [x.item() for x in mlm_index]
        unmasked_ids = [tokens[i] for i in mlm_index]
        for idx in mlm_index:
            if idx != 0:
                tokens[idx] = self.inverse_vocab["<MASK>"]


        return tokens, mlm_index, unmasked_ids



class BERTTokenizer(BPEModel):
    def __init__(self, is_train = False, max_vocab_size = 50000):
        super().__init__(is_train, max_vocab_size)

    def encode(self, text):
        token_ids =  super().encode(text)
        token_ids.insert(0, self.inverse_vocab["<CLS>"])
        return token_ids
    
    def decode(self, token_ids):
        text = super().decode(token_ids)
        text = text.replace("<CLS>", "")
        return text
        


            
class DomainBPE(BPEModel):
    def __init__(self, reference_text, training_text_path = None, is_train = False, max_vocab_size = 50000):
        super().__init__(training_text_path, is_train, max_vocab_size)
        
        self.reference_text = reference_text







    