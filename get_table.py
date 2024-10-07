import sqlite3
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

ner = spacy.load("en_core_web_sm")

conn = sqlite3.connect('D:/feverous_wikiv1.db')
cursor = conn.cursor()

vectorizer = TfidfVectorizer()

def getPages(claim):
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

def get_tables(wiki_article_dict,claim):
    table_entries=[]
    scores={}
    for k, v in wiki_article_dict.items():
        if k.startswith("table_") and k[6:].isdigit():
            for dict_entry in v["table"]:
                for list_entry in dict_entry:
                    concatenated_value = list_entry['value'] + "_" + list_entry['id']
                    table_entries.append(concatenated_value)
                    concatenated_value=""
                    
    table_entries.append(claim)
                  
    tfidf_matrix = vectorizer.fit_transform(table_entries)

    claim_vector = tfidf_matrix[-1]

    relevance_scores = np.dot(tfidf_matrix[:-1], claim_vector.T).toarray().flatten()

    q = 3
    top_q_ids = np.argsort(relevance_scores)[-q:][::-1]

    top_q_scores= [(table_entries[i], relevance_scores[i]) for i in top_q_ids] 
          
    return top_q_scores
    
def main():
    
    claim="Aramais Yepiskoposyan played for FC Ararat Yerevan, an Armenian football club based in Yerevan during 1986 to 1991"
    with open('article_sample.json', 'r') as file:
        article_list = json.load(file)

    for article in article_list:
        top_q_scores=get_tables(article,claim)
        print(top_q_scores)
            
if __name__=="__main__":
    main()




