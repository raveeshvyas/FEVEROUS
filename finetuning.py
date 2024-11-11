from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import wandb
from utils import create_train_data, create_dev_data, load_data
import json

wandb.init(project="ANLP_Project", name="Full Finetuning")

max_length=512
num_epochs=3
lr=5e-5
batch_size=16

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        label = self.labels[idx]
        
        item = {key: val.squeeze() for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        
        return item

def train_model(model, train_dataloader, val_dataloader):
    model = model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        # Wrap the train_dataloader with tqdm
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", unit="batch"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            wandb.log({"Train Loss": loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        wandb.log({"Average Train Loss": avg_loss, "Epoch": epoch + 1})

        evaluate_model(model, val_dataloader, epoch)

        torch.save(model.state_dict(), f"checkpoint_{epoch}.pt")
        wandb.save(f"checkpoint_{epoch}.pt")

def evaluate_model(model, dataloader, epoch):
    model.eval() 
    predictions, true_labels = [], []

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

    f1 = f1_score(true_labels, predictions, average='macro')
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)

    print(f"Validation F1 Score: {f1:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")

    wandb.log({
        "Validation F1 Score": f1,
        "Validation Accuracy": accuracy,
        "Validation Precision": precision,
        "Validation Recall": recall,
        "Epoch": epoch + 1
    })


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### CREATE TRAIN DATA ###############################################
# create_train_data()
# create_dev_data()

### LOAD DATA #######################################################
x_train, y_train = load_data("train_data.jsonl")
print(len(x_train))
x_dev, y_dev = load_data("dev_data.jsonl")

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3, problem_type="multi_label_classification").to(device)

train_dataset = TextDataset(x_train, y_train, tokenizer)
val_dataset = TextDataset(x_dev, y_dev, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

train_model(model, train_dataloader, val_dataloader)
wandb.finish()
