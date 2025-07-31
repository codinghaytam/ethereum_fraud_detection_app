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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
#initializing model

class FraudClassifier(nn.Module):
    def __init__(self, sequence_input_size, static_input_size=0, hidden_size=128, num_layers=3, 
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

        
# Model will be initialized after loading data

class AddressFraudDataset(Dataset):
    def __init__(self, sequences, static_features, labels):
        self.sequences = sequences
        self.static_features = static_features  # Add static features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'sequences': self.sequences[idx],
            'label': self.labels[idx]
        }
        # Add static features to the item if available
        if self.static_features is not None:
            item['static_features'] = self.static_features[idx]
        return item
    
def prepare_data():
    #loading data
    addresses = pl.read_csv('data/transaction_dataset.csv')
    transactions = pl.read_csv('data/data-1752772895586.csv', schema_overrides={ "value": pl.Float64})
    addresses= pl.concat([addresses,addresses.filter(pl.col('FLAG').is_in([1])),addresses.filter(pl.col('FLAG').is_in([1]))])  # Ensure addresses with labels are included
    # Extract static features from address dataset
    print("Available columns in address dataset:")
    print(addresses.columns)
    
    # Define static feature columns (excluding Address, Index, and FLAG)
    static_feature_columns = [
        'Avg min between sent tnx', 'Avg min between received tnx', 
        'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
        'Number of Created Contracts', 'Unique Received From Addresses', 
        'Unique Sent To Addresses', 'min value received', 'max value received ',
        'avg val received', 'min val sent', 'max val sent', 'avg val sent',
        'min value sent to contract', 'max val sent to contract', 
        'avg value sent to contract', 'total transactions (including tnx to create contract',
        'total Ether sent', 'total ether received', 'total ether sent contracts',
        'total ether balance', 'Total ERC20 tnxs', 'ERC20 total Ether received',
        'ERC20 total ether sent', 'ERC20 total Ether sent contract',
        'ERC20 uniq sent addr', 'ERC20 uniq rec addr', 'ERC20 uniq sent addr.1',
        'ERC20 uniq rec contract addr', 'ERC20 avg time between sent tnx',
        'ERC20 avg time between rec tnx', 'ERC20 avg time between rec 2 tnx',
        'ERC20 avg time between contract tnx', 'ERC20 min val rec', 'ERC20 max val rec',
        'ERC20 avg val rec', 'ERC20 min val sent', 'ERC20 max val sent',
        'ERC20 avg val sent', 'ERC20 min val sent contract', 'ERC20 max val sent contract',
        'ERC20 avg val sent contract', 'ERC20 uniq sent token name', 'ERC20 uniq rec token name'
    ]
    
    # Filter to only include columns that actually exist in the dataset
    existing_static_columns = [col for col in static_feature_columns if col in addresses.columns]
    print(f"Using {len(existing_static_columns)} static features:")
    print(existing_static_columns)
    
    #remove irrelevant columns from transactions dataset
    irrelevant_columns = ['r','s','nonce','type','v','max_priority_fee_per_gas','max_fee_per_gas','transaction_index','method_id','function_name']
    existing_irrelevant_columns = [col for col in irrelevant_columns if col in transactions.columns]
    if existing_irrelevant_columns:
        transactions = transactions.drop(existing_irrelevant_columns)

    model_y_values = addresses['FLAG'].to_numpy()
    
    # Extract static features and convert to numpy, handle missing values
    static_features_df = addresses.select(existing_static_columns)
    
    # Convert to pandas for easier preprocessing
    static_features_pd = static_features_df.to_pandas()
    
    # Handle missing values by filling with 0 or median
    for col in static_features_pd.columns:
        if static_features_pd[col].dtype in ['object', 'string']:
            # For string columns, convert to numeric if possible, otherwise fill with 0
            static_features_pd[col] = pl.Series(static_features_pd[col]).cast(pl.Float64, strict=False).fill_null(0).to_numpy()
        else:
            # For numeric columns, fill NaN with median
            median_val = static_features_pd[col].median()
            static_features_pd[col] = static_features_pd[col].fillna(median_val if not np.isnan(median_val) else 0)
    
    static_features_array = static_features_pd.values.astype(np.float32)
    print(f"Static features shape: {static_features_array.shape}")
    
    #preparing sequence data
    sequence_length = 50
    model_x_values = []

    unique_addresses = addresses["Address"].to_list()
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
        
    # Now build the final sequences and static features for addresses that have labels
    print("Building sequences and static features for labeled addresses...")
    valid_labels = []
    valid_static_features = []
    addresses_list = addresses["Address"].to_list()
    
    for i, address in enumerate(tqdm.tqdm(addresses_list, desc="Processing labeled addresses")):
        if address in address_to_transactions:
            model_x_values.append(address_to_transactions[address])
            valid_labels.append(model_y_values[i])
            valid_static_features.append(static_features_array[i])

    # Update labels and static features to match the sequences
    model_y_values = valid_labels
    model_static_features = np.array(valid_static_features)

    # Convert lists to numpy arrays
    model_x_values = np.array(model_x_values)
    model_y_values = np.array(model_y_values)

    # Save the processed data to files
    np.save('processed_sequences.npy', model_x_values)
    np.save('processed_labels.npy', model_y_values)
    np.save('processed_static_features.npy', model_static_features)
    
    print(f"Saved {len(model_x_values)} sequences, {len(model_y_values)} labels, and {model_static_features.shape} static features")
#prepare_data()
# load processed sequences, labels, and static features
model_x_values = np.load('processed_sequences.npy')
model_y_values = np.load('processed_labels.npy')
model_static_features = np.load('processed_static_features.npy')

print(f"Loaded {len(model_x_values)} sequences, {len(model_y_values)} labels, and {model_static_features.shape} static features.")

# Normalize static features
static_scaler = StandardScaler()
model_static_features_scaled = static_scaler.fit_transform(model_static_features)

print(f"Static features shape after scaling: {model_static_features_scaled.shape}")

train_sequences, temp_sequences, train_labels, temp_labels, train_static, temp_static = train_test_split(
    model_x_values, model_y_values, model_static_features_scaled,
    test_size=0.3, 
    random_state=42, 
    stratify=model_y_values  # This ensures both classes are in all splits
)

# Second split: split the 30% temp into 15% val and 15% test
val_sequences, test_sequences, val_labels, test_labels, val_static, test_static = train_test_split(
    temp_sequences, temp_labels, temp_static,
    test_size=0.5,  # 50% of 30% = 15% of total
    random_state=42,
    stratify=temp_labels  # This ensures both classes are in val and test
)

print(f"Train set: {len(train_sequences)} samples")
print(f"Validation set: {len(val_sequences)} samples") 
print(f"Test set: {len(test_sequences)} samples")

# Initialize model after loading data
model = FraudClassifier(
    sequence_input_size=10, 
    static_input_size=model_static_features_scaled.shape[1],  # Add static features size
    hidden_size=128, 
    num_layers=4
)

print(f"Model initialized with {model_static_features_scaled.shape[1]} static features")

#training model
def train_model(model, train_sequences, train_labels, train_static, val_sequences, val_labels, val_static, epochs=20, batch_size=32):
    # Create datasets and dataloaders
    train_dataset = AddressFraudDataset(train_sequences, train_static, train_labels)
    val_dataset = AddressFraudDataset(val_sequences, val_static, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
    
    for epoch in range(epochs):
        # Training step
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            sequences = batch['sequences'].float()
            labels = batch['label'].long()
            static_features = batch.get('static_features', None)
            if static_features is not None:
                static_features = static_features.float()
            
            optimizer.zero_grad()
            outputs = model(sequences, static_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                sequences = batch['sequences'].float()
                labels = batch['label'].long()
                static_features = batch.get('static_features', None)
                if static_features is not None:
                    static_features = static_features.float()
                
                outputs = model(sequences, static_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Display epoch results
        print(f"\nEpoch {epoch+1}/{epochs} Results:")
        print(f"  Training   - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"  Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        print("-" * 60)
    
    return model

def test_model(model, test_sequences, test_labels, test_static, batch_size=32):
    """Test the trained model on unseen test data"""
    test_dataset = AddressFraudDataset(test_sequences, test_static, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Testing"):
            sequences = batch['sequences'].float()
            labels = batch['label'].long()
            static_features = batch.get('static_features', None)
            if static_features is not None:
                static_features = static_features.float()
            
            outputs = model(sequences, static_features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predictions = torch.max(outputs, 1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    
    test_accuracy = correct_predictions / total_predictions
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return avg_test_loss, test_accuracy

model = train_model(model, train_sequences, train_labels, train_static, val_sequences, val_labels, val_static, epochs=5, batch_size=32)

# Test the model on unseen test data
test_loss, test_accuracy = test_model(model, test_sequences, test_labels, test_static, batch_size=32)

# Generate predictions for confusion matrix
def get_predictions(model, test_sequences, test_labels, test_static, batch_size=32):
    """Get predictions from the model for confusion matrix"""
    test_dataset = AddressFraudDataset(test_sequences, test_static, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Getting predictions"):
            sequences = batch['sequences'].float()
            labels = batch['label'].long()
            static_features = batch.get('static_features', None)
            if static_features is not None:
                static_features = static_features.float()
            
            outputs = model(sequences, static_features)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

# Get predictions for confusion matrix
y_pred, y_true = get_predictions(model, test_sequences, test_labels, test_static)

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

# Save model and related data for API use
print("Saving model and preprocessing components...")

# Save static scaler
import pickle
with open('static_scaler.pkl', 'wb') as f:
    pickle.dump(static_scaler, f)

# Save static feature columns (from earlier in the script)
static_feature_columns = [
    'Avg min between sent tnx', 'Avg min between received tnx', 
    'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
    'Number of Created Contracts', 'Unique Received From Addresses', 
    'Unique Sent To Addresses', 'min value received', 'max value received ',
    'avg val received', 'min val sent', 'max val sent', 'avg val sent',
    'min value sent to contract', 'max val sent to contract', 
    'avg value sent to contract', 'total transactions (including tnx to create contract',
    'total Ether sent', 'total ether received', 'total ether sent contracts',
    'total ether balance', 'Total ERC20 tnxs', 'ERC20 total Ether received',
    'ERC20 total ether sent', 'ERC20 total Ether sent contract',
    'ERC20 uniq sent addr', 'ERC20 uniq rec addr', 'ERC20 uniq sent addr.1',
    'ERC20 uniq rec contract addr', 'ERC20 avg time between sent tnx',
    'ERC20 avg time between rec tnx', 'ERC20 avg time between rec 2 tnx',
    'ERC20 avg time between contract tnx', 'ERC20 min val rec', 'ERC20 max val rec',
    'ERC20 avg val rec', 'ERC20 min val sent', 'ERC20 max val sent',
    'ERC20 avg val sent', 'ERC20 min val sent contract', 'ERC20 max val sent contract',
    'ERC20 avg val sent contract', 'ERC20 uniq sent token name', 'ERC20 uniq rec token name'
]

# Save sequence feature columns (from transaction data after preprocessing)
# Get feature names from the actual processed data
sequence_feature_names = [f'sequence_feature_{i}' for i in range(model_x_values.shape[2])]

# Save model configuration
model_config = {
    'sequence_input_size': 10,
    'static_input_size': model_static_features_scaled.shape[1],
    'hidden_size': 128,
    'num_layers': 4,
    'fc_hidden_sizes': [256, 128],
    'num_classes': 2,
    'dropout_rate': 0.3,
    'sequence_length': 50
}

# Save everything to files
with open('static_feature_columns.pkl', 'wb') as f:
    pickle.dump(static_feature_columns, f)

with open('sequence_feature_names.pkl', 'wb') as f:
    pickle.dump(sequence_feature_names, f)

with open('model_config.pkl', 'wb') as f:
    pickle.dump(model_config, f)

#saving model
torch.save(model.state_dict(), 'fraud_classifier.pth')

print("Saved:")
print("- fraud_classifier.pth (model state dict)")
print("- static_scaler.pkl (StandardScaler for static features)")
print("- static_feature_columns.pkl (list of static feature names)")
print("- sequence_feature_names.pkl (list of sequence feature names)")
print("- model_config.pkl (model configuration)")

# Copy files to backend API directory
import shutil
import os

backend_model_dir = 'backendApi/model'
os.makedirs(backend_model_dir, exist_ok=True)

# Copy model files to backend
files_to_copy = [
    'fraud_classifier.pth',
    'static_scaler.pkl', 
    'static_feature_columns.pkl',
    'sequence_feature_names.pkl',
    'model_config.pkl'
]

for file in files_to_copy:
    if os.path.exists(file):
        shutil.copy(file, os.path.join(backend_model_dir, file))
        print(f"Copied {file} to {backend_model_dir}")

print("Model and preprocessing components ready for API use!")
