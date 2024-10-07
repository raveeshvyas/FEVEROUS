import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import json
vectorizer = TfidfVectorizer()

def get_tables(pages,claim,q):
    table_entries=[]
    scores={}
    for page in pages:
        wiki_article_dict=json.loads(page[1])
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

    top_q_ids = np.argsort(relevance_scores)[-q:][::-1]

    top_q_scores= [table_entries[i]for i in top_q_ids] 

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




