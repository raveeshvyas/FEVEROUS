from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from finetuning import TextDataset
from utils import load_data,create_dev_data,create_test_from_retrieved
import json

def evaluate_model(model, dataloader, epoch):
    model.eval() 
    predictions, true_labels = [], []
    
    total_loss=0
    # Wrap the validation dataloader with tqdm
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.append(preds)
            true_labels.append(labels)

    predictions = torch.cat(predictions).cpu().numpy()
    true_labels = torch.cat(true_labels).cpu().numpy()
    
    # Calculate per-label precision, recall, and F1 score
    precision_per_label, recall_per_label, f1_per_label, _ = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    
    f1 = f1_score(true_labels, predictions, average='macro')
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)

    for label in range(len(precision_per_label)):
        print(f"Label {label} - Precision: {precision_per_label[label]:.4f}, Recall: {recall_per_label[label]:.4f}, F1 Score: {f1_per_label[label]:.4f}")

    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    return predictions
        
batch_size=16
model_name = 'roberta-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3, problem_type="multi_label_classification").to(device)
    
### EVALUATE THE MODEL ################################

## DEV ###############################################
create_dev_data()
x_dev, y_dev = load_data("dev_data.jsonl")
print(len(x_dev))
val_dataset = TextDataset(x_dev, y_dev, tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

## TEST ##############################################
# create_test_from_retrieved()

# COMBINED ############################################
x_test, y_test = load_data("test_data.jsonl")
test_dataset = TextDataset(x_test, y_test, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

# SENTENCE ONLY #######################################
# x_test, y_test = load_data("sentence_only.jsonl")
# test_dataset = TextDataset(x_test, y_test, tokenizer)
# test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

# TABLE ONLY ###########################################
# x_test, y_test = load_data("table_only.jsonl")
# test_dataset = TextDataset(x_test, y_test, tokenizer)
# test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

epoch=2
checkpoint = torch.load(f"Model2/checkpoint_{epoch}.pt")
# model.load_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(checkpoint)
predictions=evaluate_model(model,val_dataloader,epoch=0)

predictions_list = predictions.tolist()

output_data = []
for i, (x, y_true, y_pred) in enumerate(zip(x_test, y_test, predictions_list)):
    output_data.append({
        "input": x,
        "predicted_label": int(y_pred) 
    })

output_filename = "Predictions/sent_only_model2.jsonl"
with open(output_filename, "w") as json_file:
    for entry in output_data:
        json_file.write(json.dumps(entry) + "\n")



