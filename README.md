# Movie Recommendation Description  
The repository has the final project of my AI course in UPT for the AI master's. The files of the project and their descriptions are listed below.  
**The thing is** : based on some data of some movies that I got from Kaggle, this model should recommend some movies for a user of the dataset, based on the preferences of some other similar users, the user preferences and the clusters of similar movies. The dataset is relatively large, but I have filtered only users who have rated more than 50 movies and movies who are watched by more than 10 users. Each method of the recommendation above is done in a separate file,  
**for collaborative filtering**: The movies are recommended based on the preferences of the similar users  
**for content filtering**: The movies are recommended based on the previous genres of the movies that the user has seen, and  
**for cluster filtering**: The movies are separated in clusters of similar movies and the recommended movies will be part of the same cluster. This method becomes more effective when the number of clusters keeps increasing.  

**1. collaborativeBased.py**  

Implements a collaborative filtering recommendation system using user-movie rating data.
Loads movie and ratings datasets, merges them, and constructs a user-movie ratings matrix.
Filters movies and users to those with sufficient interaction to improve recommendation quality.
Uses k-nearest neighbors (KNN) to find similar movies based on user ratings, and evaluates the system's accuracy with various statistical metrics (MAE, RMSE, etc.).
Includes hyperparameter optimization for the KNN model and functions to recommend movies and evaluate recommendation effectiveness.  

**2. contentBased.py**  

Implements a content-based recommendation system focusing on movie genres.
Loads reduced movie and ratings datasets.
Converts movie genres into binary features and computes a cosine similarity matrix between movies.
Maps movie titles to indices and uses fuzzy matching to handle inexact movie title queries.
Recommends movies similar in genre to a given movie and evaluates the accuracy of recommendations using statistics like Jaccard similarity, MAE, and RMSE.  

**3. k-cluster.py**  

Uses clustering (KMeans) to group movies based on their genre and average rating features.
Loads movies and ratings datasets, processes genre information, and merges average ratings.
Drops irrelevant columns and applies clustering to identify movie clusters.
Recommends movies from the same cluster as a given movie and evaluates the recommendations using statistics similar to the other scripts.
Includes visualization of clusters using PCA for dimensionality reduction and matplotlib for plotting.
