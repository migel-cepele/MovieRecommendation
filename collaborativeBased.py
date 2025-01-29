import os
import pandas
import numpy
from matplotlib import pyplot
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

movies = pandas.read_csv("../movies.csv")
movies.head() #read movies from the dataset


ratings = pandas.read_csv("../ratingsReduced.csv")
ratings.head() #read ratings from the dataset


#join datasets based on the common field: movie ids
data = ratings.merge(movies, on='movieId')
print(data.head())

#create a user-movie matrix, that shows the ratings for the movie that user has seen
#if he hasnt seen it, then we will fill it with 0
user_movie_matrix = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
user_movie_matrix


user_voted_no = ratings.groupby('movieId')['rating'].agg('count') #users who voted
voted_movies_no = ratings.groupby('userId')['rating'].agg('count') #movies that were voted

#plot users
f,ax = pyplot.subplots(1,1,figsize=(16,4))
pyplot.scatter(user_voted_no.index,user_voted_no,color='mediumseagreen')
pyplot.axhline(y=10,color='r')
pyplot.xlabel('Movie Id')
pyplot.ylabel('No of users voted')
#pyplot.show()


# plot movies
f,ax = pyplot.subplots(1,1,figsize=(16,4))
pyplot.scatter(voted_movies_no.index,voted_movies_no,color='mediumseagreen')
pyplot.axhline(y=10,color='r')
pyplot.xlabel('User id')
pyplot.ylabel('No of votes by user')
#pyplot.show()


#choose the films who are voted by more than 10 users
user_movie_matrix = user_movie_matrix.loc[user_voted_no[user_voted_no > 10].index,:]

#choose the users who voted more than 50 movies, because we will have much more data to help recommend
user_movie_matrix=user_movie_matrix.loc[:,voted_movies_no[voted_movies_no > 50].index]
print(user_movie_matrix)





from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

# Split the data into 80/10/10 for training, optimization, and testing
train_data, temp_data = train_test_split(user_movie_matrix, test_size=0.2, random_state=42)
opt_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# form the sparse matrices
csr_train_data = csr_matrix(train_data.values)
csr_test_data = csr_matrix(test_data.values)
csr_opt_data = csr_matrix(opt_data.values)


# trainingthe model with knn with some arbitrary parameters
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_train_data)

# Evaluate recommendations for the testing dataset
# The thing is that this function will predict the ratings that the users in testing dataset will give these movies
# Then it will compare the real ratings these users have given these movies
# And, at the end will calculate some statistics about the effectivness of this function
def evaluate_recommendations(model, test_data, k=10):
    total_mae, total_rmse = 0, 0
    instances_classified_correctly = 0
    instances_classified_incorrectly = 0
    
    for idx in range(test_data.shape[0]):
        movie_idx = test_data.index[idx]
        
        # get real ratings
        real_ratings = test_data.iloc[idx].values
        
        # Find neighbors and predict ratings
        distances, indices = model.kneighbors(csr_test_data[idx], n_neighbors=k + 1)
        predicted_ratings = np.mean(train_data.iloc[indices.flatten()[1:]], axis=0)
        
        # errors
        mae = mean_absolute_error(real_ratings, predicted_ratings)
        rmse = np.sqrt(mean_squared_error(real_ratings, predicted_ratings))
        
        # Now, we will not compare exactly the predicted vs real rating, but will use a threshhold to determine only
        # if the user liked or not the movie. 
        for actual, predicted in zip(real_ratings, predicted_ratings):
            if (actual >= 3.5 and predicted >= 3.5) or (actual < 3.5 and predicted < 3.5):
                instances_classified_correctly += 1
            else:
                instances_classified_incorrectly += 1
        
        total_mae += mae
        total_rmse += rmse
    
    #other statistics
    n = test_data.shape[0]
    rel_squared_error = total_rmse / n

    actual_mean_rating = np.mean([rating for row in test_data.values for rating in row])
    rel_absolute_error = total_mae / np.sum([abs(r - actual_mean_rating) for row in test_data.values for r in row])

    total_instances = instances_classified_correctly + instances_classified_incorrectly
    observed_accuracy = instances_classified_correctly / total_instances
    chance_accuracy = ((sum(real_ratings >= 3.5) / n) * (sum(predicted_ratings >= 3.5) / n)) + \
                    ((sum(real_ratings < 3.5) / n) * (sum(predicted_ratings < 3.5) / n))
    epsilon = 1e-6  #floating point errors
    if chance_accuracy != 1:
        kappa = max(0, min(1, 1 - abs(1 - (observed_accuracy - chance_accuracy) / (1 - chance_accuracy)) + epsilon))
    else:
        kappa = 1  # Perfect agreement

    return {
        "1. Correctly Classified Instances": instances_classified_correctly,
        "2. Incorrectly Classified Instances": instances_classified_incorrectly,
        "3. Kappa Statistic": kappa,
        "4. Mean Absolute Error": total_mae / n,
        "5. Root Mean Squared Error": total_rmse / n,
        "6. Relative Absolute Error": rel_absolute_error,
        "7. Root Relative Squared Error": np.sqrt(rel_absolute_error)
    }

# execute the function
results = evaluate_recommendations(knn, test_data)

# Results
for key, value in results.items():
    print(f"{key}: {value}")




###################################################################
#optimizing the model parameters
# the thing is: NearestNeighbors model contains 3 important parameters in the above function
# Now we chose 3 arbitrary values for these parameters, based on the their popularity, but
# if some other parameters are better for the model, we should choose them
# So below we try different values for these parameters and calculate some statistics, most importantly
# the RMSE statistics, which we will use to determine the best combination of parameters.
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# hyperparameter grid
param_grid = {
    'n_neighbors': [5, 10, 20, 30],
    'metric': ['cosine', 'euclidean', 'manhattan'],
    'algorithm': ['brute']
}

# similiar to the above function, we will compare the real ratings and predicted ratings, but this time,
# based on the optimization dataset and based on the grid of parameters
def evaluate_model_with_params(params, train_data, val_data):
    knn = NearestNeighbors(metric=params['metric'], algorithm=params['algorithm'], 
                           n_neighbors=params['n_neighbors'], n_jobs=-1)
    knn.fit(csr_matrix(train_data.values))

    # Evaluate using validation set
    total_mae, total_rmse = 0, 0
    instances_classified_correctly, ininstances_classified_correctly = 0, 0

    for idx in range(val_data.shape[0]):
        movie_idx = val_data.index[idx]
        actual_ratings = val_data.iloc[idx].values

        # Find neighbors and predict ratings
        distances, indices = knn.kneighbors(csr_matrix(val_data.iloc[idx].values), 
                                            n_neighbors=params['n_neighbors'] + 1)
        predicted_ratings = np.mean(train_data.iloc[indices.flatten()[1:]], axis=0)

        # Compute errors
        mae = mean_absolute_error(actual_ratings, predicted_ratings)
        rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

        # Count classifications
        for actual, predicted in zip(actual_ratings, predicted_ratings):
            if (actual >= 3.5 and predicted >= 3.5) or (actual < 3.5 and predicted < 3.5):
                instances_classified_correctly += 1
            else:
                ininstances_classified_correctly += 1

        total_mae += mae
        total_rmse += rmse

    n = val_data.shape[0]
    return {
        "MAE": total_mae / n,
        "RMSE": total_rmse / n,
        "Correctly Classified": instances_classified_correctly,
        "Incorrectly Classified": ininstances_classified_correctly
    }

# iterate in parameter grid, and try each combination of parameters
best_params = None
best_rmse = float('inf')
results = []

for params in ParameterGrid(param_grid):
    print(f"Testing parameters: {params}")
    metrics = evaluate_model_with_params(params, train_data, opt_data)
    results.append({**params, **metrics})
    print(metrics['RMSE'])

    if metrics['RMSE'] < best_rmse:
        best_rmse = metrics['RMSE']
        best_params = params
# print the best parameters
print(f"Best Parameters: {best_params}")












#Example, of recommending movies

#removing sparse values, to save memory for training set
csr_train_data = csr_matrix(user_movie_matrix.values)
user_movie_matrix.reset_index(inplace=True)



#train the model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_train_data)



#trying the model
#function that for a given movie will recommend 10 other similiar movies
def movie_recommendations(movie_name):
    recommended_movies_no = 10 #no of the movies that will be recommended

    #control if the movie exists
    movie_list = movies[movies['title'].str.contains(movie_name)] 
    if len(movie_list):   
        #take movie
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = user_movie_matrix[user_movie_matrix['movieId'] == movie_idx].index[0]

        #calculate distances and indices
        distances , indices = knn.kneighbors(csr_train_data[movie_idx], n_neighbors=recommended_movies_no + 1) 

        #idk
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), 
            distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        recommend_frame = []

        for val in rec_movie_indices:
            movie_idx = user_movie_matrix.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})

        df = pandas.DataFrame(recommend_frame, index=range(1, recommended_movies_no + 1))

        return df #returns movies
    else:
        return "No movie by that name exists in the dataset of movies"



print(movie_recommendations('Inglourious Basterds'))
