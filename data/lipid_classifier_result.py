import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, accuracy_score
import matplotlib.pyplot as plt

# Load your test predictions and ground truth
# Assuming a CSV file with columns 'predicted_prob' and 'actual_label'
results_df = pd.read_csv('Models/lipid_classifier_chemprop_1_epochs/model_0_test_preds.csv')

# Extract the probability scores for the positive class and the actual labels
y_scores = results_df['prediction']
y_true = results_df['target']

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('lipid_classifier_ROC.png')

auc_score = roc_auc_score(y_true, y_scores)
print(f"The AUC Score is: {auc_score:.6f}")


plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('lipid_classifier_PR.png')

auc_pr = auc(recall, precision)
print(f"The Area Under the Precision-Recall Curve is: {auc_pr:.6f}")


y_pred = results_df['prediction'].apply(lambda x: 1 if x >= 0.5 else 0)
accuracy = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {accuracy:.6f}")