import numpy as np 

db = np.array([
    [0.1, 0.3, 0.7, 0.9],
    [0.2, 0.1, 0.4, 0.8],
    [0.9, 0.8, 0.3, 0.2],
    [0.4, 0.4, 0.6, 0.7],
    [0.7, 0.9, 0.1, 0.2]
])

query = np.array([0.3, 0.2, 0.8, 0.9])

def cosine_similarity(a,b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query, db, top_k=2):
    similarities = [cosine_similarity(query, v) for v in db]
    sorted_idx = np.argsort(similarities)[::-1]  
    return [(i, similarities[i]) for i in sorted_idx[:top_k]]


results = search(query, db)
print("Top results:", results)