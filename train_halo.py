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

# Add <unk> token to vocabulary
vocabulary.add('<unk>')

# Numericalization
word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
numericalized_tokens = [word_to_index[word] for word in tokens]

# Sequence Creation
seq_length = 10
sequences = [numericalized_tokens[i:i+seq_length] for i in range(len(numericalized_tokens) - seq_length)]
targets = numericalized_tokens[1:1+len(sequences)]

# Define Dataset
class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sequence, target

# Create dataset and data loader
sequence_dataset = SequenceDataset(sequences, targets)
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
        output = output[:, -1, :]  # Get the last output for each sequence in the batch
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
    for input_sequence, target_sequence in data_loader:
        # Initialize hidden state for each batch
        hidden = model.init_hidden(input_sequence.size(0))

        # Forward pass
        output, hidden = model(input_sequence, hidden)
        hidden = tuple([each.data for each in hidden])  # Detach hidden state

        # Reshape output for loss calculation
        output = output.contiguous().view(-1, vocab_size)

        # Calculate loss
        loss = loss_function(output, target_sequence)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        # Print loss
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Save Model State after training
model_path = 'model_halo.pth'
torch.save(model.state_dict(), model_path)

# Load and preprocess validation data
validation_data_path = 'validation_data.txt'
with open(validation_data_path, 'r', encoding='UTF-8') as file:
    validation_data = file.read()

validation_tokens = tokenize(validation_data)
validation_numericalized = [word_to_index.get(token, word_to_index["<unk>"]) for token in validation_tokens]

# Create sequences and targets for validation data
validation_sequences = [validation_numericalized[i:i+seq_length+1] for i in range(len(validation_numericalized) - seq_length)]
validation_targets = validation_numericalized[1:1+len(validation_sequences)]
# Load validation_dataset
validation_dataset = SequenceDataset(validation_sequences, validation_targets)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

# Load the trained model for evaluation
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluation loop
validation_loss = 0
with torch.no_grad():
    for input_sequence, target_sequence in validation_loader:
        output, _ = model(input_sequence, None)  # No need for hidden state in evaluation
        loss = loss_function(output.view(-1, vocab_size), target_sequence)
        validation_loss += loss.item()

# Calculate average loss
average_validation_loss = validation_loss / len(validation_loader)
print(f'Average Validation Loss: {average_validation_loss}')

# Scores and stats will be shared in repo (screenshots or table)
