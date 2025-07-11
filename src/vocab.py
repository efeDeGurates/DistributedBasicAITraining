class Vocab:
    def __init__(self, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
        self.token_to_idx = {}
        self.idx_to_token = []
        for token in specials:
            self.add_token(token)

    def add_token(self, token):
        if token not in self.token_to_idx:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def build_vocab(self, token_lists):
        for tokens in token_lists:
            for token in tokens:
                self.add_token(token)

    def __call__(self, tokens):
        return [self.token_to_idx.get(token, self.token_to_idx.get("<unk>", 0)) for token in tokens]

    def lookup_token(self, idx):
        return self.idx_to_token[idx] if 0 <= idx < len(self.idx_to_token) else "<unk>"

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.token_to_idx.get("<unk>", 0))