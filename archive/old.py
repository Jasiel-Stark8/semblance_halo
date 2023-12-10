import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Tokenization
def tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

# Load data and tokenize
file_path = os.path.join(os.getcwd(), 'data.txt')
with open(file_path, 'r', encoding='UTF-8') as file:
    synthetic_data = file.read()

tokens = tokenize(synthetic_data)
vocabulary = set(tokens)

# Numericalization
word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
numericalized_tokens = [word_to_index[word] for word in tokens]

# Sequence Creation
seq_length = 10
sequences = [numericalized_tokens[i:i+seq_length+1] for i in range(len(numericalized_tokens) - seq_length)]

# Define Dataset
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_sequence = torch.tensor(sequence[:-1], dtype=torch.long)
        target_sequence = torch.tensor(sequence[1:], dtype=torch.long)  # Shifted by one for predicting the next word
        return input_sequence, target_sequence

# Create dataset and data loader
sequence_dataset = SequenceDataset(sequences)
data_loader = DataLoader(sequence_dataset, batch_size=64, shuffle=True)

# Model Definition
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, hidden_size),
                torch.zeros(1, batch_size, hidden_size))

# Model Initialization
vocab_size = len(vocabulary)
embed_size = 128
hidden_size = 256

model = RNNModel(vocab_size, embed_size, hidden_size)

# Loss and Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    hidden = model.init_hidden(64)

    for input_sequence, target_sequence in data_loader:
        output, hidden = model(input_sequence, hidden)
        hidden = tuple([each.data for each in hidden])  # Detach hidden state
        loss = loss_function(output.view(-1, vocab_size), target_sequence.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item()}')


# Save Model State after training
model_path = 'model_halo.pth'
torch.save(model.state_dict(), model_path)



# Validation
# Load and preprocess validation data
validation_data_path = 'validation_data.txt'
with open(validation_data_path, 'r', encoding='UTF-8') as file:
    validation_data = file.read()

# Tokenize and numericalize the validation data using the same word_to_index mapping
validation_tokens = tokenize(validation_data)
validation_numericalized = [word_to_index.get(token, word_to_index["<unk>"]) for token in validation_tokens]

# Ensure "<unk>" token is in your vocabulary
if "<unk>" not in word_to_index:
    word_to_index["<unk>"] = len(word_to_index)  # Assign a unique index to "<unk>"
    vocabulary.add("<unk>")

# Adjusted Model Initialization (if vocab_size changed due to "<unk>")
vocab_size = len(vocabulary)  # Recalculate vocab_size in case "<unk>" was added

# Tokenize and numericalize the validation data using the same word_to_index mapping
# Replace unknown words with "<unk>"
validation_tokens = tokenize(validation_data)
validation_numericalized = [word_to_index.get(token, word_to_index["<unk>"]) for token in validation_tokens]

# Create sequences for validation data
validation_sequences = [validation_numericalized[i:i+seq_length+1] for i in range(len(validation_numericalized) - seq_length)]

# Output the preprocessed validation data for verification
with open('validation_numericalized.txt', 'w', encoding='utf-8') as f:
    for sequence in validation_sequences:
        sequence_str = ','.join(str(idx) for idx in sequence)  # Use ',' for proper CSV format
        f.write(sequence_str + '\n')


# Load validation_dataset
validation_dataset = SequenceDataset(validation_sequences)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

# Load Model
model = RNNModel(vocab_size, embed_size, hidden_size)
model.load_state_dict(torch.load('model_halo.pth'))
model.eval()


# Evaluation loop
validation_loss = 0
with torch.no_grad():
    for input_sequence, target_sequence in validation_loader:
        output, _ = model(input_sequence, None)  # No need for hidden state in evaluation
        loss = loss_function(output.view(-1, vocab_size), target_sequence.view(-1))
        validation_loss += loss.item()

# Calculate the average loss
average_validation_loss = validation_loss / len(validation_loader)
print(f'Average Validation Loss: {average_validation_loss}')
