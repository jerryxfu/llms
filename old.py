import os
import wget

# Download the data --------------------------------
input_dir = "input"
dest = os.path.join(input_dir, "data.txt")
if not os.path.exists(input_dir):
    os.makedirs(input_dir)
if not os.path.isfile(dest):
    wget.download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", out=dest)
with open(dest, "r", encoding="utf-8") as f:
    text = f.read()
# --------------------------------------------------

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    """Encode a string into a list of integers."""
    encoded = []
    for c in s:
        encoded.append(stoi[c])
    return encoded


def decode(l):
    """Decode a list of integers into a string."""
    decoded = ''
    for i in l:
        decoded += itos[i]
    return decoded


print(f"nb. of characters in the text: {len(text)}")
print(f"nb. of unique characters: {vocab_size}")
print(f"first 100 characters (encoded): {encode(text[:100])}")
print(f"first 100 characters (decoded): {text[:100]}")
