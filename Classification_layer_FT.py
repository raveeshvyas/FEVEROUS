from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import wandb
from utils import create_train_data, create_dev_data, load_data,create_test_from_retrieved
import json

'''
Only the final linear classification layer (or other task-specific heads) is trained, while the RoBERTa encoder layers are frozen.
'''

wandb.init(project="ANLP_Project", name="Classifer Layer tuning")

max_length=512
num_epochs=5
lr=5e-5
batch_size=16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()


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

def train_model(model, train_dataloader, val_dataloader,test_dataloader):
    model = model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr)

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
        test_model(model,test_dataloader,epoch)

        torch.save(model.state_dict(), f"classifierLayerFT_checkpoints/checkpoint_{epoch}.pt")
        wandb.save(f"checkpoint_{epoch}.pt")

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
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            total_loss=total_loss+loss.item()

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
        "Epoch": epoch + 1,
        "Validation Loss":total_loss/len(dataloader)
    })

def test_model(model, dataloader, epoch):
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
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            total_loss=total_loss+loss.item()
            predictions.append(preds)
            true_labels.append(labels)

    predictions = torch.cat(predictions).cpu().numpy()
    true_labels = torch.cat(true_labels).cpu().numpy()

    f1 = f1_score(true_labels, predictions, average='macro')
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)

    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

    wandb.log({
        "Test F1 Score": f1,
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Epoch": epoch + 1,
        "Test Loss":total_loss/len(dataloader),
    })

def main():

    ### CREATE TRAIN DATA ###############################################
    # create_train_data()
    # create_dev_data()
    # create_test_from_retrieved()

    ### LOAD DATA #######################################################
    x_train, y_train = load_data("train_data.jsonl")
    print(len(x_train))
    x_dev, y_dev = load_data("dev_data.jsonl")
    print(len(x_dev))
    x_test, y_test = load_data("test_data.jsonl")
    print(len(x_test))

    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3, problem_type="multi_label_classification").to(device)

    train_dataset = TextDataset(x_train, y_train, tokenizer)
    val_dataset = TextDataset(x_dev, y_dev, tokenizer)
    test_dataset = TextDataset(x_test, y_test, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    # Freeze all RoBERTa layers
    for param in model.roberta.parameters():
        param.requires_grad = False

    # Only the classification head will have requires_grad=True, meaning it will be updated
    for param in model.classifier.parameters():
        param.requires_grad = True

    train_model(model, train_dataloader, val_dataloader,test_dataloader)
    wandb.finish()
    
if __name__=="__main__":
    main()
