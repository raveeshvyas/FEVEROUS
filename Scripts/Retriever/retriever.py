import sqlite3
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from table_retriever import get_tables
from sent_retriever import getSentence

ner = spacy.load("en_core_web_sm")

conn = sqlite3.connect('feverous_wikiv1.db')
cursor = conn.cursor()

vectorizer = TfidfVectorizer()

def getPages(claim, k):
    ents = ner(claim)
    namedEntities = [ent.text for ent in ents.ents]

    if not namedEntities:
        print("No named entities found.")
        return
    
    # Optimized SQL query with LIMIT and refined LIKE
    like_clauses = " OR ".join([f"id LIKE ?" for _ in namedEntities])
    sql_query = f"SELECT id FROM wiki WHERE {like_clauses} LIMIT 200;"
    like_params = [f"{entity}%" for entity in namedEntities]

    cursor.execute(sql_query, like_params)
    ids = cursor.fetchall()

    if not ids:
        print("No relevant pages found.")
        return

    # Fetch only the necessary 'data' for top `k` ids
    ids_placeholder = ", ".join("?" for _ in ids)
    cursor.execute(f"SELECT id, data FROM wiki WHERE id IN ({ids_placeholder})", [id[0] for id in ids])
    entries = cursor.fetchall()

    scores = {}
    firstSentences = [json.loads(entry[1]).get("sentence_0", "") for entry in entries]
    firstSentences.append(claim)

    tfidf_matrix = vectorizer.fit_transform(firstSentences)
    claim_vector = tfidf_matrix[-1]
    relevance_scores = np.dot(tfidf_matrix[:-1], claim_vector.T).toarray().flatten()

    for i, entry in enumerate(entries):
        id_val = entry[0]
        scores[id_val] = relevance_scores[i]

    top_k_ids = np.argsort(relevance_scores)[-k:][::-1]
    top_k_scores = {entries[i]: scores[entries[i][0]] for i in top_k_ids}

    return list(top_k_scores.keys())



def main():
    # claim = "Aramais Yepiskoposyan played for FC Ararat Yerevan, an Armenian football club based in Yerevan during 1986 to 1991."
    data = []
    with open("data.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))

    k = 5
    l = 5
    q = 3

    for i, line in enumerate(data):
        if i <= 21:
            continue
        try:
            claim = line['claim']
            pages = getPages(claim,k)
            sentences = getSentence(pages,l,claim)
            tables = get_tables(pages,claim,q)
            print(tables)
            dumpDict = {'claim': claim, 'label': line['label'], 'sentences': sentences, 'tables': tables}
            with open("retrieved_data.jsonl", "a") as f:
                f.write(json.dumps(dumpDict) + "\n")
        except:
            pass

if __name__=="__main__":
    main()
