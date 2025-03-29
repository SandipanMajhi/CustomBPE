import json
from collections import Counter, deque

class BPEModel:
    def __init__(self, training_text_path = None, is_train = False, max_vocab_size = 50000):
        self.vocab = {}
        self.merge_rules = {}
        self.inverse_vocab = {}

        self.save_root = "Data"
        self.save_vocab_path = f"{self.save_root}/vocab.json"
        self.save_merge_path = f"{self.save_root}/merge.json"
        self.inverse_vocab_path = f"{self.save_root}/inverse_vocab.json"

        self.training_text_path = training_text_path
        self.is_train = is_train
        if self.is_train:
            text = self.load_text()
            self.train(text, vocab_size = 50000)

        if not self.is_train:
            self.load_vocab()
            self.load_merge_rules()
            self.load_inverse_vocab()

    
    def load_vocab(self):
        with open(self.save_vocab_path, 'r', encoding = 'utf-8') as f:
            self.vocab = json.load(f)
        print(f"Vocabulary loaded from {self.save_vocab_path}")
        

    def load_merge_rules(self):
        with open(self.save_merge_path, 'r', encoding = 'utf-8') as f:
            self.merge_rules = json.load(f)
        print(f"Merge rules loaded from {self.save_merge_path}")
    
    def load_inverse_vocab(self):
        with open(self.inverse_vocab_path, 'r', encoding = 'utf-8') as f:
            self.inverse_vocab = json.load(f)
        print(f"Inverse vocabulary loaded from {self.inverse_vocab_path}")


    def load_text(self):
        if self.training_text_path is not None:
            with open(self.training_text_path, 'r', encoding = 'utf-8') as f:
                text = f.read()
        return text


    def train(self, text, vocab_size, special_tokens = ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]):
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

        self.vocab = {i : char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char : i for i, char in enumerate(unique_chars)}

        token_ids = [self.inverse_vocab[char] for char in preprocess_text]

        for new_idx in range(len(self.vocab), vocab_size):
            most_freq_pairs_idx = self.find_most_frequent(token_ids=token_ids)
            if most_freq_pairs_idx is None:
                break
            ### Replace the pair ##
            token_ids = self.replace_pairs(token_ids, most_freq_pairs_idx, new_idx)
            ### Add a merge rule ##
            self.merge_rules[most_freq_pairs_idx] = new_idx

        for (old_0, old_1), new_idx in self.merge_rules.items():
            self.vocab[new_idx] = self.vocab[old_0] + self.vocab[old_1]
            self.inverse_vocab[self.vocab[old_0] + self.vocab[old_1]] = new_idx

        self.save()
    

    def find_most_frequent(self, token_ids):
        pairs = Counter(zip(token_ids, token_ids[1:]))
        return max(pairs.items(), key = lambda x : x[1])[0]
    

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
        words = text.replace("\n", " \n ").split(' ')

        for i, word in enumerate(words):
            if i > 0 and not word.startswith("\n"):
                tokens.append("#" + word)
            else:
                tokens.append(word)

        token_ids = []

        for token in tokens:
            if token in self.inverse_vocab:
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
    

    def decode(self, token_ids):
        decoded_string = ""

        for token_id in token_ids:
            token = self.vocab[token_id]
            if token.startswith("#"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string
    

    def save(self):

        with open(self.save_vocab_path, 'w', encoding = 'utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii = False)
        with open(self.save_merge_path, 'w', encoding = 'utf-8') as f:
            json.dump(self.merge_rules, f, ensure_ascii = False)
        with open(self.inverse_vocab_path, 'w', encoding = 'utf-8') as f:
            json.dump(self.inverse_vocab, f, ensure_ascii = False)
        print(f"Vocabulary saved to {self.save_vocab_path}")
        print(f"Merge rules saved to {self.save_merge_path}")
        print(f"Inverse vocabulary saved to {self.inverse_vocab_path}")


            

        









    