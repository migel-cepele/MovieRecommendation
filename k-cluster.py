from sklearn.preprocessing import StandardScaler
import pandas
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

movies = pandas.read_csv("../movies.csv")
movies.head() #read movies from the dataset


ratings = pandas.read_csv("../ratingsReduced.csv")
ratings.head() #read ratings from the dataset



# extract genres
genres = set('|'.join(movies['genres']).split('|'))
for g in genres:
    movies[g] = movies['genres'].apply(lambda x: 1 if g in x else 0)

# add the avarage rating for each movie in movies dataset
movie_avg_ratings = ratings.groupby('movieId')['rating'].mean()
movies = movies.merge(movie_avg_ratings, on='movieId', how='left')

# drop irrelevant columns for clustering
clustering_data = movies.drop(columns=['movieId', 'title', 'genres']).fillna(0)




def recommend_movies_by_cluster(movie_title, movies, num_recommendations=5):
    # Find the cluster for the given movie
    if movie_title not in movies['title'].values:
        return f"Movie '{movie_title}' not found in the dataset."
    
    movie_cluster = movies.loc[movies['title'] == movie_title, 'cluster'].values[0]
    
    # Get all movies from the same cluster
    cluster_movies = movies[movies['cluster'] == movie_cluster]
    
    # Exclude the input movie and return recommendations
    recommendations = cluster_movies[cluster_movies['title'] != movie_title].head(num_recommendations)

    # Ensure genres are formatted correctly (in case of missing or invalid genres)
    recommendations['genres'] = recommendations['genres'].apply(lambda x: str(x) if isinstance(x, str) else '')
    
    return recommendations[['title', 'genres', 'cluster']]
    

#recommendations = recommend_movies_by_cluster("Toy Story (1995)", movies, num_recommendations=5)
#print(recommendations)




from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Define the statistics calculation function
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
    rmse = np.sqrt(mae)
    rae = mae / num_recommendations
    rrse = rmse / num_recommendations

    return correctly_classified, incorrectly_classified, mae, rmse, rae, rrse

# Function to perform recommendation and evaluate statistics for different cluster sizes
def evaluate_clusters_for_recommendations(movie_title, movies, cluster_sizes=[10, 30, 50], num_recommendations=5):
    stats_results = {}
    
    for num_clusters in cluster_sizes:
        # Prepare clustering data
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        
        # Prepare data for clustering (without the columns that are not features)
        clustering_data = movies.drop(columns=['movieId', 'title', 'genres']).fillna(0)
        
        # Standardize the data
        scaler = StandardScaler()
        clustering_data_scaled = scaler.fit_transform(clustering_data)
        
        # Fit KMeans model
        kmeans.fit(clustering_data_scaled)
        
        # Assign clusters to movies
        movies['cluster'] = kmeans.labels_

        #visualize data
        # Reduce dimensions to 2 for visualization
        pca = PCA(n_components=2)
        clustering_data_pca = pca.fit_transform(clustering_data_scaled)

        # Plot clusters
        plt.figure(figsize=(10, 6))
        plt.scatter(clustering_data_pca[:, 0], clustering_data_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=10)
        plt.title("Movie Clusters")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(label="Cluster")
        plt.show()

        
        # get recommendations based on that cluster
        recommendations = recommend_movies_by_cluster(movie_title, movies, num_recommendations)
        print(recommendations)
        
        if not recommendations.empty:
            # calculate statistics for that cluster
            correct, incorrect, mae, rmse, rae, rrse = calculate_statistics(movies[movies['title'] == movie_title], recommendations)
            
            #statistics for that cluster
            stats_results[num_clusters] = {
                'Correctly Classified Instances': correct,
                'Incorrectly Classified Instances': incorrect,
                'Mean Absolute Error (MAE)': mae,
                'Root Mean Squared Error (RMSE)': rmse,
                'Relative Absolute Error (RAE)': rae,
                'Root Relative Squared Error (RRSE)': rrse
            }
    
    return stats_results

# test with movie toy story
movie_title = "Toy Story (1995)"
stats_results = evaluate_clusters_for_recommendations(movie_title, movies, cluster_sizes=[10, 30, 50])

# the results for every cluster
for num_clusters, stats in stats_results.items():
    print(f"Results for {num_clusters} clusters:")
    for stat, value in stats.items():
        print(f"{stat}: {value:.2f}")
    print("\n")

