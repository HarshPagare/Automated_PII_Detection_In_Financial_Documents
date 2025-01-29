"""# Generating Graphs and Matrix"""

# Import necessary libraries
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file with multiple sheets
file_path = r"Results.xlsx"  
xls = pd.ExcelFile(file_path)

# Load the "Predicted Results" sheet
predicted_results_df = pd.read_excel(xls, sheet_name='Predicted Results')

# Function to parse the string representations of entities
def parse_entities(entity_string):
    if isinstance(entity_string, str):
        return eval(entity_string)
    return []

# Parse the True and Predicted Results
predicted_results_df['True Parsed'] = predicted_results_df['True Results'].apply(parse_entities)
predicted_results_df['Predicted Parsed'] = predicted_results_df['Predicted Results'].apply(parse_entities)

# Initialize lists for confusion matrix
true_labels = []
predicted_labels = []
predicted_probabilities = []

# Iterate through each row to compare and prepare confusion matrix data
for idx, row in predicted_results_df.iterrows():
    true_entities = {(start, end, label) for start, end, label in row['True Parsed']}
    predicted_entities = {(start, end, label) for start, end, label in row['Predicted Parsed']}

    # Align true and predicted entities by their labels
    for entity in true_entities:
        start, end, label = entity
        if entity in predicted_entities:
            true_labels.append(label)
            predicted_labels.append(label)
            predicted_probabilities.append(1)  # Assuming perfect prediction for demonstration
        else:
            true_labels.append(label)
            predicted_labels.append('O')  # 'O' for incorrect prediction or missed detection
            predicted_probabilities.append(0)  # Assuming missed prediction

    for entity in predicted_entities:
        if entity not in true_entities:
            _, _, label = entity
            true_labels.append('O')
            predicted_labels.append(label)
            predicted_probabilities.append(0.5)  # Assuming some confidence for incorrect prediction

# Calculate average metrics
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
accuracy = accuracy_score(true_labels, predicted_labels)

# Print the average metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

# Data for plotting
metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
values = [precision, recall, f1, accuracy]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['skyblue', 'orange', 'green', 'red'])
plt.ylim(0, 1)
plt.title('Model Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.show()

# Generate the confusion matrix
labels = sorted(set(true_labels + predicted_labels))  # Get unique labels
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)

# Plot the confusion matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix for PII Detection')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# ROC Curve for each class
plt.figure(figsize=(10, 8))
for i, label in enumerate(labels):
    y_true = [1 if l == label else 0 for l in true_labels]
    y_pred = [1 if l == label else 0 for l in predicted_labels]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line for random classifier
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall curve for each class
plt.figure(figsize=(10, 8))
for i, label in enumerate(labels):
    y_true = [1 if l == label else 0 for l in true_labels]
    y_pred = [1 if l == label else 0 for l in predicted_labels]
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision, label=label)

plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
