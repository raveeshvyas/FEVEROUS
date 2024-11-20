# README

## Directory Structure

```
│   .gitignore
│   README.md
│   requirements.txt
│
├───BaselineOutput
│       combined_predictions_model1.jsonl
│       combined_predictions_model2.jsonl
│       sent_only_predictions_model1.jsonl
│       sent_only_predictions_model2.jsonl
│       table_only_predictions_model1.jsonl
│       table_only_predictions_model2.jsonl
│
├───data
│       data.jsonl
│       retrieved_data_3.jsonl
│
└───Scripts
    ├───Retriever
    │       retriever.py
    │       sent_retriever.py
    │       table_retriever.py
    │       verification.py
    │
    └───VerdictPredictor
            evaluate.py
            RoBERTa.py
            RoBERTa_NLI.py
            Roberta_NLI_NEI.py
            utils.py

```
## Source files 
    - BaselineOutput - Stores the retrieved evidence and predicted verdicts for each of the claims in the test data. It contains the predicted verdicts when only sentence, table or combined input is provided as evidence when predicting the labels.
    - RoBERTa.py - To finetune the RoBERTa base classifier model on the train data
    - RoBERTa_NLI.py - To finetune the NLI-pretrained RoBERTa base classifier model on the train data
    - Roberta_NLI_NEI.py - To finetune the NEI-balanced Nli-pretrained RoBERTa base classifier model on the train data
    - utils.py - To execute helper functions for reading data to feed as input to the model.
    - evaluate.py - To evaluate any of the trained models on the dev or test data

## Running the model

1. Clone the directory with the following link:

```
https://github.com/raveeshvyas/ANLP-project.git

```

2. Install all the dependencies. For a clean installation, use a virtual environemt.
```bash
pip install -r requirements.txt
```

3. Download the sqlite dataset and extract it. The size of the database is 53 GB
https://fever.ai/dataset/feverous.html


4. Evidence Retrieval
Run the script `retriever.py` to retrieve the evidence for claim in the FEVEROUS train data.

```bash
python3 retriever.py
```

5. Verdict Prediction

In the ``` ./Scripts/Verdict Predictor``` directory, find the three finetuning scripts of the experiments.
- To finetune the RoBERTa base classifier model on the train data, run 

```
python3 RoBERTa.py

```
- To finetune the NLI-pretrained RoBERTa base classifier model on the train data, run 

```

python3 RoBERTa_NLI.py

```

- To finetune the NEI-balanced Nli-pretrained RoBERTa base classifier model on the train data, run 

```

python3 Roberta_NLI_NEI.py

```
- To evaluate any of the trained models on the dev or test data, run the following command

```

python3 evaluate.py

```
Ensure that the data used for training/evaluating is present in the same directory structure as specified above.
