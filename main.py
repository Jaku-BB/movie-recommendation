import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import pandas


def get_user_id_to_index(user_set, user_id=None):
    user_id_to_index_map = {user_id: index for index, user_id in enumerate(list(user_set))}

    return user_id_to_index_map if user_id is None else user_id_to_index_map[user_id]


def get_movie_id_to_index(movie_set, movie_id=None):
    movie_id_to_index_map = {movie_id: index for index, movie_id in enumerate(list(movie_set))}

    return movie_id_to_index_map if movie_id is None else movie_id_to_index_map[movie_id]


def read_data():
    file = pandas.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'])

    user_set = sorted(file['userId'].unique())
    movie_set = sorted(file['movieId'].unique())

    user_movie_rating_matrix = np.zeros((len(user_set), len(movie_set)))

    user_id_to_index = {user_id: index for index, user_id in enumerate(list(user_set))}
    movie_id_to_index = {movie_id: index for index, movie_id in enumerate(list(movie_set))}

    user_indices = file['userId'].map(user_id_to_index)
    movie_indices = file['movieId'].map(movie_id_to_index)
    user_movie_rating_matrix[user_indices, movie_indices] = file['rating']

    return user_movie_rating_matrix, user_set, movie_set


def find_nearest_neighbors(user_id, user_movie_rating_matrix, amount, user_set):
    user_index = get_user_id_to_index(user_set, user_id)

    nearest_neighbors_model = NearestNeighbors(n_neighbors=amount + 1, metric='correlation')
    nearest_neighbors_model.fit(user_movie_rating_matrix)

    indices = nearest_neighbors_model.kneighbors(user_movie_rating_matrix[user_index].reshape(1, -1),
                                                 return_distance=False)

    nearest_neighbors = [list(get_user_id_to_index(user_set).keys())
                         [list(get_user_id_to_index(user_set).values()).index(index)] for index in indices.flatten()]

    nearest_neighbors.remove(user_id)

    return nearest_neighbors


def recommend_movie(user_id, user_movie_rating_matrix, nearest_neighbors, user_similarity, user_set, movie_set):
    user_index = get_user_id_to_index(user_set, user_id)
    movies_without_user_rating = np.where(user_movie_rating_matrix[user_index] == 0)[0]
    movie_ratings = {}

    for movie_index in movies_without_user_rating:
        movie_id = list(get_movie_id_to_index(movie_set).keys())[movie_index]
        ratings = []
        similarities = []

        for neighbor_id in nearest_neighbors:
            neighbor_index = get_user_id_to_index(user_set, neighbor_id)
            rating = user_movie_rating_matrix[neighbor_index, movie_index]

            if rating != 0:
                similarity = user_similarity[user_index, neighbor_index]

                similarities.append(similarity)
                ratings.append(rating)

        if ratings:
            weighted_sum = np.dot(ratings, similarities)
            similarity_sum = np.sum(similarities)

            movie_ratings[movie_id] = weighted_sum / similarity_sum

    return max(movie_ratings, key=movie_ratings.get)


def main():
    user_id = 1

    user_movie_rating_matrix, user_set, movie_set = read_data()
    user_similarity = 1 - pairwise_distances(user_movie_rating_matrix, metric='correlation')
    np.fill_diagonal(user_similarity, 0)

    nearest_neighbors = find_nearest_neighbors(user_id, user_movie_rating_matrix, 10, user_set)

    print(f"Najbliżsi sąsiedzi użytkownika {user_id}: {nearest_neighbors}")
    print("Rekomendowany film:", recommend_movie(user_id, user_movie_rating_matrix, nearest_neighbors, user_similarity,
                                                 user_set, movie_set))


if __name__ == '__main__':
    main()
