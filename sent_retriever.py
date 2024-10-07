from retriever import getPages

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def getSentence (claim,k):
    sentences = []
    pages = getPages(claim)
    for page in pages:
        output = json.loads(page[1])
        order = output['order']
        for item in order:
            if item.startswith("sentence"):
                sentences.append(output[item])
            elif item.startswith("list"):
                for list_item in output[item]['list']:
                    sentences.append(list_item['value'])

    vectorizer = TfidfVectorizer()
    text_data = [claim] + sentences

    tfidf_matrix = vectorizer.fit_transform(text_data)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = similarity_scores.argsort()[::-1][:min(k,len(sentences))]
    top_sentences = [sentences[i] for i in top_indices]
    top_scores = [similarity_scores[i] for i in top_indices]

    # Display the top sentences and their scores
    for i in range(len(top_sentences)):
        print(f"Sentence: {top_sentences[i]}, Score: {top_scores[i]}")
        print()

    top_k_results = list(zip(top_sentences, top_scores))
    return top_k_results


# claim = "Algebraic logic has five Logical system and Lindenbaum–Tarski algebra which includes Physics algebra and Nodal algebra (provide models of propositional modal logics)"
# getSentence(claim,3)

