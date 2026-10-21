import math
import torch
import torch.nn as nn

text = "hello hello hello"


chars=sorted(set(text))

stoi = {ch : i for i , ch in enumerate(chars)}
itos = {i : ch for ch , i in stoi.items()}


print("Unique characters:", chars)
print("Character to ID:", stoi)
print("ID to character:", itos)


encoded = [stoi[ch] for ch in text]

print("Original text:", text)
print("Encoded text:", encoded)


# Convert the token IDs back into text.
decoded = "".join(itos[i] for i in encoded)

print("Decoded text:", decoded)



# Step 4: Create input and target sequences

text = "hello"

# Convert characters into IDs using the mapping from your existing code
data = [stoi[ch] for ch in text]

# Input contains all characters except the last
x = data[:-1]


# Target contains all characters except the first
y = data[1:]

print("\n--- Training Data ---")
print("Text:", text)
print("Token IDs:", data)
print("Input (x):", x)
print("Target (y):", y)

for i in range(len(x)):
    print(f"Input: {itos[x[i]]} -> Target: {itos[y[i]]}")



# A sequence of token IDs
data = [2, 1, 3, 3, 4]

# Convert the list into a PyTorch tensor
x = torch.tensor(data)

print("Tensor:", x)
print("Shape:", x.shape)
print("Data type:", x.dtype)

# Example configuration
vocab_size = 5
embedding_dim = 4
block_size = 8

# Embedding for token identities
token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# Embedding for token positions
position_embedding = nn.Embedding(
    num_embeddings=block_size,
    embedding_dim=embedding_dim
)

# Create position IDs: 0, 1, 2, 3, 4
positions = torch.arange(len(x))

# Retrieve token embeddings
token_emb = token_embedding(x)

# Retrieve positional embeddings
pos_emb = position_embedding(positions)

# Combine token and positional embeddings
combined = token_emb + pos_emb

print("Input shape:", x.shape)
print("Token embedding shape:", token_emb.shape)
print("Position embedding shape:", pos_emb.shape)
print("Combined shape:", combined.shape)



# Dimensions of Query, Key, and Value
head_dim = 4

# Three learned transformations
W_query = nn.Linear(embedding_dim, head_dim, bias=False)
W_key = nn.Linear(embedding_dim, head_dim, bias=False)
W_value = nn.Linear(embedding_dim, head_dim, bias=False)

# Generate Q, K, V
Q = W_query(combined)
K = W_key(combined)
V = W_value(combined)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

sequence_length = len(x)

# Create a causal mask
mask = torch.tril(
    torch.ones(sequence_length, sequence_length)
).bool()

print("Causal mask:")
print(mask)

# Calculate and scale attention scores
scores = (Q @ K.T) / math.sqrt(head_dim)

# Block future positions
scores = scores.masked_fill(~mask, float("-inf"))

# Normalize the remaining scores
weights = torch.softmax(scores, dim=-1)

# Calculate attention output
output = weights @ V

print("Causal attention weights:")
print(weights)

print("Attention output shape:", output.shape)

