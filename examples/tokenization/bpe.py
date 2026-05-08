from collections import Counter, defaultdict

def get_vocab(text):
    vocab = {}
    for word in text.split():
        chars = ' '.join(list(word)) + ' </w>'
        vocab[chars] = vocab.get(chars, 0) + 1
    return vocab


def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)

    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab


def train_bpe(text, num_merges=10):
    vocab = get_vocab(text)
    
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        
        best = max(pairs, key=pairs.get)
        print(f"Step {i+1}: Merge {best}")
        
        vocab = merge_vocab(best, vocab)
    
    return vocab



text = "low lower newest widest longest"
vocab = train_bpe(text, num_merges=100)

print("\nFinal vocab:")
for k, v in vocab.items():
    print(k, ":", v)
