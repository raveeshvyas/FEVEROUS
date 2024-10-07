import sqlite3
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

ner = spacy.load("en_core_web_sm")

conn = sqlite3.connect('feverous_wikiv1.db')
cursor = conn.cursor()

vectorizer = TfidfVectorizer()

def getPages(claim):
    """
    Returns top k pages in the form of a list 
    Each page contains its title and contents in json format 
    """
    """
    Returns top k pages in the form of a list 
    Each page contains its title and contents in json format 
    """
    ents = ner(claim)
    namedEntities = [ent.text for ent in ents.ents]

    if not namedEntities:
        print("No named entities found.")
        return
    
    like_clauses = " OR ".join([f"id LIKE ?" for _ in namedEntities])
    sql_query = f"SELECT id, data FROM wiki WHERE {like_clauses};"
    like_params = [f"%{entity}%" for entity in namedEntities]

    cursor.execute(sql_query, like_params)
    entries = cursor.fetchall()

    scores = {}
    firstSentences = []

    for entry in entries:
        id_val = entry[0]
        data_val = entry[1]
        jsonData = json.loads(data_val)
        if "sentence_0" in jsonData:
            firstSentences.append(jsonData["sentence_0"])
        else:
            firstSentences.append("")
    
    firstSentences.append(claim)

    
    tfidf_matrix = vectorizer.fit_transform(firstSentences)

    claim_vector = tfidf_matrix[-1]

    relevance_scores = np.dot(tfidf_matrix[:-1], claim_vector.T).toarray().flatten()

    for i, entry in enumerate(entries):
        id_val = entry[0]
        scores[id_val] = relevance_scores[i]

    k = 5
    top_k_ids = np.argsort(relevance_scores)[-k:][::-1]
    top_k_scores = {entries[i]: scores[entries[i][0]] for i in top_k_ids}

    return list(top_k_scores.keys())
