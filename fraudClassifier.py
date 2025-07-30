#import packages
import polars as pl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
#initializing model

class FraudClassifier(nn.Module):
    def __init__(self, sequence_input_size, static_input_size=0, hidden_size=128, num_layers=2, 
                 fc_hidden_sizes=[256, 128], num_classes=2, dropout_rate=0.3):
        super(FraudClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.static_input_size = static_input_size
        
        # LSTM for sequential transaction data
        self.lstm = nn.LSTM(
            input_size=sequence_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Calculate the combined feature size
        lstm_output_size = hidden_size * 2  # bidirectional
        combined_size = lstm_output_size + static_input_size
        
        # Fully connected layers
        fc_layers = []
        prev_size = combined_size
        
        for fc_hidden_size in fc_hidden_sizes:
            fc_layers.extend([
                nn.Linear(prev_size, fc_hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(fc_hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = fc_hidden_size
        
        # Output layer
        fc_layers.append(nn.Linear(prev_size, num_classes))
        
        self.classifier = nn.Sequential(*fc_layers)
    
    def forward(self, sequences, static_features=None):
        batch_size = sequences.size(0)
        
        # Initialize hidden state
        h0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(sequences.device)
        c0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(sequences.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(sequences, (h0, c0))
        
        # Use the last output from LSTM
        lstm_features = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Combine LSTM features with static features if available
        if static_features is not None and self.static_input_size > 0:
            combined_features = torch.cat([lstm_features, static_features], dim=1)
        else:
            combined_features = lstm_features
        
        # Pass through classifier
        output = self.classifier(combined_features)
        return output

        
model = FraudClassifier(sequence_input_size=10, hidden_size=128, num_layers=4)

class AddressFraudDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'sequences': self.sequences[idx],
            'label': self.labels[idx]
        }
        return item
    
def prepare_data():
    #loading data
    addresses = pl.read_csv('data/transaction_dataset.csv')
    transactions = pl.read_csv('data/data-1752772895586.csv', schema_overrides={ "value": pl.Float64})

    #remove irrelevant columns from transactions dataset
    irrelevant_columns = ['r','s','nonce','type','v','max_priority_fee_per_gas','max_fee_per_gas','transaction_index','method_id','function_name']
    existing_irrelevant_columns = [col for col in irrelevant_columns if col in transactions.columns]
    if existing_irrelevant_columns:
        transactions = transactions.drop(existing_irrelevant_columns)


    model_y_values = addresses['FLAG'].to_numpy()
    #preparing data
    sequence_length = 50
    model_x_values = []


    unique_addresses = set(addresses["Address"].to_list())
    print(f"Found {len(unique_addresses)} unique addresses in the addresses dataset.")


    transactions_filtered = transactions.filter(
        (pl.col('from_address').is_in(unique_addresses)) |
        (pl.col('to_address').is_in(unique_addresses)) |
        (pl.col('source_address').is_in(unique_addresses))
    )

    # Convert created_at to date format using polars
    transactions_filtered = transactions_filtered.with_columns(
        pl.col('created_at').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S.%f').dt.strftime('%Y%m%d').cast(pl.Int64, strict=False).alias('created_at')
    )


    # Create a mapping of addresses to their transactions
    print("Creating address-transaction mapping...")
    address_to_transactions = {}

    for address in tqdm.tqdm(unique_addresses, desc="Mapping addresses"):
        # Use polars filter for efficient filtering
        address_transactions = transactions_filtered.filter(
            (pl.col('from_address') == address) |
            (pl.col('to_address') == address) |
            (pl.col('source_address') == address)
        )
        # Select only numeric columns
        numeric_columns = [col for col in address_transactions.columns if address_transactions[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]]
        transactions_numeric = address_transactions.select(numeric_columns)
        # Convert to numpy for the sequence
        sequence = transactions_numeric.limit(sequence_length).to_numpy()
        if sequence.shape[0] < sequence_length:
            padding_rows = sequence_length - sequence.shape[0]
            padding = np.zeros((padding_rows, sequence.shape[1]))
            sequence = np.vstack([sequence, padding])

        address_to_transactions[address] = sequence
    # Now build the final sequences for addresses that have labels
    print("Building sequences for labeled addresses...")
    valid_labels = []
    addresses_list = addresses["Address"].to_list()
    for i, address in enumerate(tqdm.tqdm(addresses_list, desc="Processing labeled addresses")):
        if address in address_to_transactions:
            model_x_values.append(address_to_transactions[address])
            valid_labels.append(model_y_values[i])

    # Update labels to match the sequences
    model_y_values = valid_labels

    # Convert lists to numpy arrays
    model_x_values = np.array(model_x_values)
    model_y_values = np.array(model_y_values)

    # Save the processed sequences and labels to files
    np.save('processed_sequences.npy', model_x_values)
    np.save('processed_labels.npy', model_y_values)
#prepare_data()
# load processed sequences and labels
model_x_values = np.load('processed_sequences.npy')
model_y_values = np.load('processed_labels.npy')
print(f"Loaded {len(model_x_values)} sequences and {len(model_y_values)} labels.")
train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(
    model_x_values, model_y_values, 
    test_size=0.3, 
    random_state=42, 
    stratify=model_y_values  # This ensures both classes are in all splits
)

# Second split: split the 30% temp into 15% val and 15% test
val_sequences, test_sequences, val_labels, test_labels = train_test_split(
    temp_sequences, temp_labels,
    test_size=0.5,  # 50% of 30% = 15% of total
    random_state=42,
    stratify=temp_labels  # This ensures both classes are in val and test
)

print(f"Train set: {len(train_sequences)} samples")
print(f"Validation set: {len(val_sequences)} samples") 
print(f"Test set: {len(test_sequences)} samples")
#training model
def train_model(model, train_sequences, train_labels, val_sequences, val_labels, epochs=20, batch_size=32):
    # Create datasets and dataloaders
    train_dataset = AddressFraudDataset(train_sequences, train_labels)
    val_dataset = AddressFraudDataset(val_sequences, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            sequences = batch['sequences'].float()
            labels = batch['label'].long()  # Changed to long() for CrossEntropyLoss, removed unsqueeze
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                sequences = batch['sequences'].float()
                labels = batch['label'].long()  # Changed to long() for CrossEntropyLoss, removed unsqueeze
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        print(f"Validation Loss: {val_loss / len(val_loader)}")
    
    return model

def test_model(model, test_sequences, test_labels, batch_size=32):
    """Test the trained model on unseen test data"""
    test_dataset = AddressFraudDataset(test_sequences, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()  # Changed from BCEWithLogitsLoss to CrossEntropyLoss
    model.eval()
    
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Testing"):
            sequences = batch['sequences'].float()
            labels = batch['label'].long()  # Changed to long() for CrossEntropyLoss, removed unsqueeze
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Calculate accuracy - changed for multi-class classification
            _, predictions = torch.max(outputs, 1)  # Get the class with highest probability
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    
    test_accuracy = correct_predictions / total_predictions
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return avg_test_loss, test_accuracy

model = train_model(model, train_sequences, train_labels, val_sequences, val_labels, epochs=55, batch_size=32)

# Test the model on unseen test data
test_loss, test_accuracy = test_model(model, test_sequences, test_labels, batch_size=32)

# Generate predictions for confusion matrix
def get_predictions(model, test_sequences, test_labels, batch_size=32):
    """Get predictions from the model for confusion matrix"""
    test_dataset = AddressFraudDataset(test_sequences, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Getting predictions"):
            sequences = batch['sequences'].float()
            labels = batch['label'].long()
            
            outputs = model(sequences)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

# Get predictions for confusion matrix
y_pred, y_true = get_predictions(model, test_sequences, test_labels)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Fraud', 'Fraud'], 
            yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

#saving model
torch.save(model.state_dict(), 'fraud_classifier.pth')
