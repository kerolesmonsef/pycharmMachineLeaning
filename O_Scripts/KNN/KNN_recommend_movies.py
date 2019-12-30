from O_Scripts.KNN.OwnKKN import knn, euclidean_distance
import pandas as pd
import numpy as np


def recommend_movies(movie_query, k_recommendations):
    raw_movies_data = pd.read_csv("Data/Classification/movies_recommendation_data.csv")
    # Prepare the data for use in the knn algorithm by picking
    # the relevant columns and converting the numeric columns
    # to numbers since they were read in as strings
    movies_recommendation_data = raw_movies_data.drop(['Movie ID', 'Movie Name', 'IMDB Rating'], 1)
    movies_recommendation_data = np.array(movies_recommendation_data)
    # Use the KNN algorithm to get the 5 movies that are most
    # similar to The Post.
    recommendation_indices, _ = knn(
        movies_recommendation_data, movie_query, k=k_recommendations,
        distance_fn=euclidean_distance, choice_fn=lambda x: None
    )

    movie_recommendations = []
    for _, index in recommendation_indices:
        movie_recommendations.append(raw_movies_data.loc[index])
        print('index', index)
    return movie_recommendations


if __name__ == '__main__':
    the_post = [7.2, 1, 1, 0, 0, 0, 0, 1, 0]  # feature vector for The Post
    recommended_movies = recommend_movies(movie_query=the_post, k_recommendations=5)

    # Print recommended movie titles
    for recommendation in recommended_movies:
        print(recommendation[1])
