import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from utils.config import cos_top_n, knn_top_n


class SimilaritySearch:

    def __init__(self):
        self.top_n = cos_top_n
        # Initialize the NearestNeighbors model with cosine similarity
        self.knn_model = NearestNeighbors(n_neighbors=knn_top_n, metric='cosine')

    # Function to perform KNN similarity search using cosine similarity
    def knn_similarity_search(self, target_vector, vectors):
        # Fit the model on the dataset of vectors
        self.knn_model.fit(vectors)
        
        # Reshape target vector to 2D (as required by scikit-learn)
        target_vector = np.array(target_vector).reshape(1, -1)
        
        # Perform the KNN search
        distances, indices = self.knn_model.kneighbors(target_vector)
        
        # Return the indices of the nearest vectors and their distances
        return indices.flatten(), distances.flatten()


    # Function to compute cosine similarity and return top N most similar vectors
    def cosine_similarity_search(self, target_vector, vectors):
        # Reshape target_vector to 2D since cosine_similarity expects 2D arrays
        target_vector = np.array(target_vector).reshape(1, -1)
        
        # Compute cosine similarities between the target vector and all other vectors
        similarities = cosine_similarity(vectors, target_vector)
        
        # Flatten the result array
        similarities = similarities.flatten()

        # Get indices of top N most similar vectors (in descending order of similarity)
        top_n_indices = similarities.argsort()[::-1][:self.top_n]
        
        # Get corresponding similarity scores
        top_n_similarities = similarities[top_n_indices]
        return top_n_indices, top_n_similarities
    


    def search(self, target_vector, vectors):
        
        knn_top_indices, knn_top_distances = self.knn_similarity_search(target_vector, vectors)
        cos_top_indices, cos_top_similarities = self.cosine_similarity_search(target_vector, vectors)

        context_pages = set()
        for i in cos_top_indices:
            context_pages.add(i)
        for i in knn_top_indices:
            context_pages.add(i)
        
        return context_pages



