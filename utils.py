import json
import random

# 0 = Supported, 1 = Refuted, 2 = NEI (Not Enough Information)
labels = [0, 1, 2]

# concatenate the context preceded by content - [context][content]- separated by [SEP]
def concatenate_evidence(evidence):
    for item in evidence:
        context = item['context']
        content = item['content']

        for i in range(0,len(content)):
            for j in range(0,len(context[content[i]])):
                title = context[content[i]][j]
                concatenated_text = title + " [SEP] "
        
            sentences = " [SEP] ".join(content)
            concatenated_text += sentences
            
    return concatenated_text

def create_train_data():
    file_path = "data/feverous_train_challenges.jsonl"
    output_file = "train_data.jsonl"
    # x_train=[]
    # y_train=[]
    count_0=0
    count_1=0
    count_2=0
    with open(file_path, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
        for line in f:
            row = json.loads(line)
            if(row.get("claim", "")):
                concatenate_evidence_text=concatenate_evidence(row.get("evidence", ""))
                input_text = f"{row.get("claim", "")} [SEP] {concatenate_evidence_text}"
                target_text=f"{row.get("label", "")}"
                if target_text == "SUPPORTS":
                    label = 1
                    count_0=count_0+1
                elif target_text == "REFUTES":
                    label = 0
                    count_1=count_1+1
                elif target_text == "NOT ENOUGH INFO":
                    label = 2
                    count_2=count_2+1
                # x_train.append(input_text)
                # y_train.append(label)
                
                json.dump({"claim": input_text, "label": label}, out_f)
                out_f.write("\n")
                
        print(count_0,count_1,count_2)

def create_nei_train_data():
    file_path = "data/train_data_nei.jsonl"
    output_file = "train_data_with_nei_samples.jsonl"
    # x_train=[]
    # y_train=[]
    count_0=0
    count_1=0
    count_2=0
    with open(file_path, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
        for line in f:
            row = json.loads(line)
            if(row.get("claim", "")):
                concatenate_evidence_text=concatenate_evidence(row.get("evidence", ""))
                input_text = f"{row.get("claim", "")} [SEP] {concatenate_evidence_text}"
                target_text=f"{row.get("label", "")}"
                if target_text == "SUPPORTS":
                    label = 1
                    count_0=count_0+1
                elif target_text == "REFUTES":
                    label = 0
                    count_1=count_1+1
                elif target_text == "NOT ENOUGH INFO":
                    label = 2
                    count_2=count_2+1
                # x_train.append(input_text)
                # y_train.append(label)
                
                json.dump({"claim": input_text, "label": label}, out_f)
                out_f.write("\n")
        print(count_0,count_1,count_2)

def add_nei_samples():
    file_path = "data/feverous_train_challenges.jsonl"
    output_file = "data/train_data_nei.jsonl"
    with open(file_path, 'r') as file:
        feverous_data = [json.loads(line) for line in file]

    support_samples = []
    for sample in feverous_data:
        if sample['label'] == 'SUPPORTS':
            support_samples.append(sample)

    selected_samples = random.sample(support_samples, 8000)

    modified_samples = []
    i = 0
    for sample in selected_samples:
        if i < 7995:
            modified_copy = sample
            modified_copy['evidence'] = selected_samples[i+5]['evidence']
            modified_copy['label'] = "NOT ENOUGH INFO"
            modified_samples.append(modified_copy)
            i += 1
        else :
            modified_copy = sample
            modified_copy['evidence'] = selected_samples[7999-i]['evidence']
            modified_copy['label'] = "NOT ENOUGH INFO"
            modified_samples.append(modified_copy)
            i += 1

    nei_sample_list = []
    i = 0
    for sample in feverous_data:
        if sample in selected_samples:
            nei_sample_list.append(modified_samples[i])
            i += 1
        else:
            nei_sample_list.append(sample)

    with open(output_file, 'w') as file:
        for sample in nei_sample_list:
            json.dump(sample, file)
            file.write('\n')


def create_dev_data():
    
    file_path = "data/feverous_dev_challenges.jsonl"
    output_file = "dev_data.jsonl"
    # x_train=[]
    # y_train=[]
    
    with open(file_path, "r", encoding="utf-8") as f,open(output_file, "w", encoding="utf-8") as out_f:
        for line in f:
            row = json.loads(line)
            if(row.get("claim", "")):
                concatenate_evidence_text=concatenate_evidence(row.get("evidence", ""))
                input_text = f"{row.get("claim", "")} {concatenate_evidence_text}"
                target_text=f"{row.get("label", "")}"
                if target_text == "SUPPORTS":
                    label = 1
                elif target_text == "REFUTES":
                    label = 0
                elif target_text == "NOT ENOUGH INFO":
                    label = 2
                # x_train.append(input_text)
                # y_train.append(label)
                
                json.dump({"claim": input_text, "label": label}, out_f)
                out_f.write("\n")
                
def load_data(file_path):
    x_train = []
    y_train = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())  # Load each line as a JSON object
            x_train.append(data["claim"])    # Extract the claim text
            y_train.append(data["label"])    # Extract the label (integer)
    
    return x_train, y_train

def concatenate_sentences_and_tables(data):
    sentences = data.get('sentences', [])
    tables = data.get('tables', [])
    
    concatenated_sentences = ' [SEP] '.join(sentence[1] for sentence in sentences)
    
    concatenated_tables = ' [SEP] '.join(tables)
    
    result = concatenated_sentences + ' [SEP] ' + concatenated_tables
    return result.strip()

def create_test_from_retrieved():
    file_path = "data/retrieved_data_2.jsonl"
    output_file = "test_data.jsonl"
    
    with open(file_path, "r", encoding="utf-8") as f,open(output_file, "w", encoding="utf-8") as out_f:
            for line in f:
                row = json.loads(line)
                concatenate_evidence_text=concatenate_sentences_and_tables(row)
                input_text = f"{row.get("claim", "")} {concatenate_evidence_text}"
                target_text=f"{row.get("label", "")}"
                if target_text == "SUPPORTS":
                    label = 1
                elif target_text == "REFUTES":
                    label = 0
                elif target_text == "NOT ENOUGH INFO":
                    label = 2
                
                json.dump({"claim": input_text, "label": label}, out_f)
                out_f.write("\n")
       
        


