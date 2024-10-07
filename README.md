# Running the model
1. Install all the dependencies. For a clean installation, use a virtual environemt.
```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

2. Download the sqlite dataset and extract it. The size of the database is 53 GB
https://fever.ai/dataset/feverous.html


3. Run the script `retriever.py`
```bash
python3 retriever.py
```