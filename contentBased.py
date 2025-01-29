import os
import pandas
import numpy
from matplotlib import pyplot
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

movies = pandas.read_csv("../moviesReduced.csv")
movies.head() #read movies from the dataset


ratings = pandas.read_csv("../ratingsReduced.csv")
ratings.head() #read ratings from the dataset


n_movies = movies['movieId'].nunique()
print(f"There are {n_movies} unique movies in our movies dataset.")



#model and training

from sklearn.model_selection import train_test_split

# split the dataset again in percentage 80/10/10
train_data, temp_data = train_test_split(movies, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")



from sklearn.metrics.pairwise import cosine_similarity

# split and get all genres
genres = set('|'.join(movies['genres']).split('|'))
for g in genres:
    movies[g] = movies['genres'].apply(lambda x: 1 if g in x else 0)

movie_genres = movies.drop(columns=['movieId', 'title', 'genres'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(movie_genres, movie_genres)
print(f"Cosine similarity matrix dimensions: {cosine_sim.shape}")

# map movie titles to indices so we can reference them later
movie_idx = dict(zip(movies['title'], list(movies.index)))




#testing
from rapidfuzz import process

# this is a helper function to find the exact movie title for a movie
# function get the closest match to that movie, if it doesnt find the exact one
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    if closest_match:
        return closest_match[0]
    else:
        return None

# function to recommend movies
# the thing is: when we give the movie title as input, the function
# gets the cosine scores in cosine matrix and then finds movies with similar movies with similar scores
def get_recommendations(title_string, n_recommendations=5):
    title = movie_finder(title_string)  # Find closest match for input movie
    print(f"Closest match found: {title}")

    if title in movie_idx:
        idx = movie_idx[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:(n_recommendations + 1)]  # Exclude the movie itself
        similar_movies = [i[0] for i in sim_scores]

        # show the movies and genres
        recommendations = movies.iloc[similar_movies][['title', 'genres']]
        print(f"\nRecommendations for '{title}':")
        print(recommendations)

        return recommendations
    else:
        print("The movie was not found in the dataset.")
        return None


# this function is used to calculate the statistics of the accuracy of the recommendations
# the thing is: the function gets the input movie genres and compares them to the recommended
# movies genres. If the genres have an intersection, then the correctness will increase.
def calculate_statistics(input_movie, recommendations):
    input_genres = set(input_movie['genres'].iloc[0].split('|'))
    correct_count = 0
    total_similarity = 0
    num_recommendations = len(recommendations)

    for _, row in recommendations.iterrows():
        recommended_genres = set(row['genres'].split('|'))
        intersection = input_genres & recommended_genres
        union = input_genres | recommended_genres

        # Calculate Jaccard similarity
        jaccard_similarity = len(intersection) / len(union)
        total_similarity += jaccard_similarity

        # Count correct classifications
        if len(intersection) > 0:
            correct_count += 1

    # Calculate statistics
    correctly_classified = correct_count
    incorrectly_classified = num_recommendations - correct_count
    mae = 1 - (total_similarity / num_recommendations)
    rmse = (mae ** 0.5)
    rae = mae / num_recommendations
    rrse = rmse / num_recommendations

    # Print results
    print(f"Correctly Classified Instances: {correctly_classified}")
    print(f"Incorrectly Classified Instances: {incorrectly_classified}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Relative Absolute Error (RAE): {rae:.2f}")
    print(f"Root Relative Squared Error (RRSE): {rrse:.2f}")



#Finally we test the above model with a random movie from the testing dataset
# with repeated tests we found out that the recommendations are quite good, so there is no need to try
# and use another model to try and optimize the model
random_movie = test_data.sample(n=1)
recommendations = get_recommendations(random_movie['title'].iloc[0], n_recommendations=10)
if recommendations is not None:
    calculate_statistics(random_movie, recommendations)




