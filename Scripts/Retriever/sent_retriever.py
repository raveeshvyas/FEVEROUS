import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def getSentence (pages,l,claim):
    sentences = []
    ids = []
    for page in pages:
        output = json.loads(page[1])
        order = output['order']
        for item in order:
            if item.startswith("sentence"):
                sentences.append(output[item])
                ids.append(item)

    vectorizer = TfidfVectorizer()
    text_data = [claim] + sentences

    tfidf_matrix = vectorizer.fit_transform(text_data)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = similarity_scores.argsort()[::-1][:min(l,len(sentences))]
    top_sentences = [sentences[i] for i in top_indices]
    top_scores = [similarity_scores[i] for i in top_indices]
    top_ids = [ids[i] for i in top_indices]

    # Display the top sentences and their scores
    for i in range(len(top_sentences)):
        print(f"Sentence: {top_sentences[i]}, Score: {top_scores[i]}")
        print()

    top_l_results = list(zip(top_ids, top_sentences))
    return top_l_results

