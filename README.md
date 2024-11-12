# Running the model
1. Install all the dependencies. For a clean installation, use a virtual environemt.
```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Download the sqlite dataset and extract it. The size of the database is 53 GB
https://fever.ai/dataset/feverous.html


3. Run the script `retriever.py`
```bash
python3 retriever.py
```

4. To finetune the RoBERTa classifier model, run the following command

```
python3 finetuning.py
```

5. The checkpoints during finetuning for the RoBERTa model is uploaded here:
- Use Checkpoint-4
Link[https://iiitaphyd-my.sharepoint.com/:f:/g/personal/ketaki_shetye_research_iiit_ac_in/Ev6AEwbja9dAnaYAWKNQQkEByj7bs_LQjp4cjvbfpoj65A?e=8Aa4fB]