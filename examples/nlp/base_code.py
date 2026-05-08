import numpy as np

# Import your custom Neural Network classes
from deeplearning.network import Network
from deeplearning.layers import Dense, ActivationLayer
from deeplearning.activations import softmax, softmax_prime
from deeplearning.losses import cross_entropy, cross_entropy_prime

# ==========================================
# Part 0: Configuration and Vocabulary
# ==========================================
content = "the king sat on the royal throne and think about the future"
words = content.split()

# Create a dictionary to map words to IDs and vice versa
unique_words = list(set(words))
vocab_size = len(unique_words)
word_to_id = {word: i for i, word in enumerate(unique_words)}
id_to_word = {i: word for i, word in enumerate(unique_words)}

# ==========================================
# Part 1: Embedding Model (Building 10-D vectors)
# ==========================================
print("--- Step 1: Creating Embeddings ---")
embedding_size = 10

# We use a random matrix to represent the embedding weights for now.
# Shape: (vocab_size, embedding_size)
embeddings = np.random.randn(vocab_size, embedding_size) 

def get_vector(word):
    word_id = word_to_id[word]
    return embeddings[word_id]

# ==========================================
# Part 2: Preparing Data for Language Model
# ==========================================
print("--- Step 2: Preparing Dataset ---")
window_size = 3

def create_dataset(words_list, window):
    X, Y = [], []
    for i in range(len(words_list) - window):
        context_words = words_list[i : i + window]
        target_word = words_list[i + window]
        
        # Get vectors and concatenate them -> 30-D vector
        context_vectors = [get_vector(w) for w in context_words]
        x_concat = np.concatenate(context_vectors) 
        
        # Target word to One-Hot
        y_onehot = np.zeros(vocab_size)
        y_onehot[word_to_id[target_word]] = 1
        
        X.append(x_concat)
        Y.append(y_onehot)
        
    # Return as standard Numpy arrays (Shape: [4, 30] and [4, 6])
    return np.array(X), np.array(Y) 

X_train, Y_train = create_dataset(words, window_size)

# ==========================================
# Part 3: Building and Training the Model
# ==========================================
print("--- Step 3: Training Language Model ---")
input_size = window_size * embedding_size  # 3 * 10 = 30
hidden_size = 20  

lm_net = Network()
lm_net.add(Dense(input_size, hidden_size))
lm_net.add(Dense(hidden_size, vocab_size))
lm_net.add(ActivationLayer(softmax, softmax_prime))

lm_net.use(cross_entropy, cross_entropy_prime)

epochs = 1000
learning_rate = 0.05

# FIX: Safely slice Numpy arrays to keep shape (1, 30) 
# This prevents both the "Matrix misalignment" and the "Integer Scalar" errors!
for epoch in range(epochs):
    for i in range(len(X_train)):
        x_sample = X_train[i:i+1] # Slices out shape (1, 30) safely
        y_sample = Y_train[i:i+1] # Slices out shape (1, 6) safely
        lm_net.fit(x_sample, y_sample, epochs=1, learning_rate=learning_rate)

print("Training Complete!")

# ==========================================
# Part 4: Text Generation (The Magic)
# ==========================================
print("\n--- Step 4: Text Generation ---")

def generate_text(start_words, num_words_to_generate):
    current_words = list(start_words) 
    
    for _ in range(num_words_to_generate):
        # Take the last 'window_size' words
        window_words = current_words[-window_size:]
        
        # Convert to vector and concatenate
        context_vectors = [get_vector(w) for w in window_words]
        x_input = np.concatenate(context_vectors)
        
        # Reshape to (1, 30) for prediction
        x_input = np.array([x_input]) 
        
        # Predict the next word
        prediction = lm_net.predict(x_input)[0] 
        
        # FIX: Convert numpy integer to standard python int
        predicted_id = int(np.argmax(prediction))
        predicted_word = id_to_word[predicted_id]
        
        current_words.append(predicted_word)
        
    return " ".join(current_words)

# Test the model with the first 3 words
seed = ("the", "king", "sat")
generated_sentence = generate_text(seed, 4) 

print(f"Seed words: {' '.join(seed)}")
print(f"Generated sentence: {generated_sentence}")