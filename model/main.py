import polars as pl
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import os

data = pl.read_csv("ETFD_Dataset.txt", separator="\t", has_header=True)
# Filter to keep only numeric columns
numeric_data = data.select([
    pl.col(col) for col in data.columns 
    if data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]
])

numeric_data = numeric_data.fill_null(0)

numeric_data = numeric_data.fill_nan(0)

# Convert to pandas for sklearn (since sklearn expects pandas/numpy)
numeric_data_pd = numeric_data.to_pandas()


# Split the data
train_split, test_split = train_test_split(numeric_data_pd, test_size=0.2, random_state=42)

# Separate features and target
X_train = train_split.drop(["Fraud","blockNumber"], axis=1)
y_train = train_split["Fraud"]
X_test = test_split.drop(["Fraud","blockNumber"], axis=1)
y_test = test_split["Fraud"]

#data scalling
from sklearn.preprocessing import StandardScaler,Normalizer
import joblib

#normalization
normalizer = Normalizer()
X_train_normalized = normalizer.fit_transform(X_train)
X_test_normalized = normalizer.transform(X_test)


# Train model
model = GradientBoostingClassifier(n_estimators=300, random_state=42,learning_rate=0.1)
model.fit(X_train_normalized, y_train)
print (model.score(X_test, y_test))
print( model.feature_importances_)


# Make predictions
y_pred = model.predict(X_test_normalized)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Minimal print output
print(f"Model Performance - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# Create comprehensive single plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
           xticklabels=['Not Fraud', 'Fraud'], 
           yticklabels=['Not Fraud', 'Fraud'])
ax1.set_title('Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# Add confusion matrix values as text
tn, fp, fn, tp = cm.ravel()
ax1.text(0.5, -0.15, f'TN:{tn} FP:{fp} FN:{fn} TP:{tp}', 
         transform=ax1.transAxes, ha='center', fontsize=10)

# 2. Performance Metrics Bar Chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
bars = ax2.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
ax2.set_title('Performance Metrics')
ax2.set_ylabel('Score')
ax2.set_ylim(0, 1)

# Add value labels on bars
for bar, value in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom', fontsize=10)

# 3. Feature Importance (Top 10)
feature_names = X_train.columns
feature_importance = model.feature_importances_
indices = np.argsort(feature_importance)[::-1][:10]

ax3.bar(range(10), feature_importance[indices], color='lightsteelblue')
ax3.set_title('Top 10 Feature Importances')
ax3.set_xlabel('Features')
ax3.set_ylabel('Importance')
ax3.set_xticks(range(10))
ax3.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')

# 4. Dataset Distribution
labels = ['Not Fraud', 'Fraud']
sizes = [len(y_test) - sum(y_test), sum(y_test)]
colors = ['lightblue', 'lightcoral']
wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax4.set_title('Test Set Distribution')

# Add dataset info as text
info_text = f'Total: {len(numeric_data_pd)} samples\nFeatures: {len(X_train.columns)}\nTest samples: {len(X_test)}'
ax4.text(1.3, 0.5, info_text, transform=ax4.transAxes, fontsize=10, 
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('model_evaluation_complete.png', dpi=300, bbox_inches='tight')
joblib.dump(model, "gradient_boosting_model.pkl")
plt.show()