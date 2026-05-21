import re
import json
from collections import Counter, defaultdict


class TinyStoriesTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = []
        self.ids = {} 
               


    def _pretokenize(self, text:str) -> list[str]:
        regexp = r" ?[A-Za-z0-9']+|[^A-Za-z0-9'\" ]| ?[\"]"
        tokens = re.findall(regexp, text)
        return tokens


    def _generate_word_frequencies(self, words:list[str]) -> dict[tuple[str], int]:
        # YOUR CODE HERE
        # 1 frequencies of word
        counts = {}
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        # 2. from word to tuple of characters + orginal frequency
        dicc = {}
        for word_string, freq in counts.items():
            word_tuple  = tuple(word_string)
            dicc[word_tuple] = freq

        return dicc  # REPLACE THIS EXPRESSION WITH YOUR CODE


    def _count_token_bigrams(self, word_freqs: dict) -> defaultdict(int):
        bigram_counts = defaultdict(int)

        # We need to decompose word in 2 characters.
        for word_tuple, freq in word_freqs.items():
            # Find pairs A,B -> ['h','o', 'l', 'a'] -> ho, ol, la
            for i in range(len(word_tuple)-1):
                bigram = (word_tuple[i], word_tuple[i+1])
                bigram_counts[bigram] += freq
        #YOUR CODE HERE
        return bigram_counts


    def _find_most_frequent_token_bigram(self, word_freqs:dict) -> tuple[str, str]:
        bigram_counts = self._count_token_bigrams(word_freqs)
        # If there are several most frequent bigrams, return the first one
        best_bigram = max(bigram_counts, key=bigram_counts.get)
        # REPLACE WITH YOUR CODE
        return best_bigram


    def _merge_bigram(self, word_freqs: dict, best_bigram: tuple, new_token: str) -> dict:
        new_word_freqs = {}

        for word_tuple, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                # Comparamos la pareja actual con la mejor pareja encontrada
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == best_bigram:
                    new_word.append(new_token)
                    i += 2  # Saltamos los dos que unimos
                else:
                    new_word.append(word_tuple[i])
                    i += 1


            new_word_freqs[tuple(new_word)] = freq  
        # REPLACE WITH YOUR CODE    
        return new_word_freqs


    def train(self, corpus_path:str):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        words = self._pretokenize(raw_text)
        word_freqs = self._generate_word_frequencies(words)

        # Initialize vocabulary with all unique individual characters found
        unique_chars = set()
        for word_tuple in word_freqs:
            for char in word_tuple:
                unique_chars.add(char)
        self.vocab = list(unique_chars)

        # Merging loop
        merges = {}  # (bigram) -> priority (lower number -> higher priority)
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            best_bigram = self._find_most_frequent_token_bigram(word_freqs)   
            self.merges[best_bigram] = i # Rank         
            new_token = "".join(best_bigram)
            self.vocab.append(new_token)
            word_freqs = self._merge_bigram(word_freqs, best_bigram, new_token)
            if (i + 1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_bigram} -> {new_token}")
        print(f"Merge {i+1}/{num_merges}: {best_bigram} -> {new_token}")
        self.ids = {v: k for k, v in enumerate(self.vocab)}


    def tokenize(self, text):
        tokens = []
        pretokens = self._pretokenize(text)
        for word in pretokens:
            parts = list(word) # [' ', 't', 'h', 'e', 'r', 'e']
            while len(parts) > 1:
                current_parts = {tuple(parts): 1} # fucntions expect dictionary
                bigrams_in_word = self._count_token_bigrams(current_parts)

                # Look for priority
                best_bigram = None
                min_priority = float('inf') 

                for bigram in bigrams_in_word: # we choose leat rpiority not the most common????
                    if bigram in self.merges:
                        priority = self.merges[bigram]
                        if priority < min_priority:
                            min_priority = priority
                            best_bigram = bigram

                # If no bigram to join
                if best_bigram is None:
                    break
                # Join
                new_token = "".join(best_bigram)
                new_word_dict = self._merge_bigram(current_parts, best_bigram, new_token)
                parts = list(list(new_word_dict.keys())[0])
            tokens.extend(parts)

        # Final tokens to numeric IDs
        return tokens, [self.ids[t] for t in tokens]

    def decode_to_tokens(self, ids: list[int]) -> list[str]:
    # Convierte una lista de IDs numéricos de vuelta a sus strings de BPE
        return [self.vocab[i] for i in ids]

    def decode(self, ids: list[int]) -> str:
        # Junta todo para recuperar el texto original
        return "".join(self.decode_to_tokens(ids))


    def save(self, path):
        serializable_merges = {f"{k[0]}<SPLIT>{k[1]}": v for k, v in self.merges.items()}
        data = {"vocab": self.vocab, "merges": serializable_merges, "vocab_size": self.vocab_size, "ids": self.ids}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)


    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls(vocab_size=data["vocab_size"])
        instance.vocab = data["vocab"]
        instance.ids = data["ids"]
        instance.merges = {tuple(k.split("<SPLIT>")): v for k, v in data["merges"].items()}
        return instance
