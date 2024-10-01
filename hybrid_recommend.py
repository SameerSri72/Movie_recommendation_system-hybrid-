#!/usr/bin/env python
# coding: utf-8

# In[1]:


from surprise import SVD, SVDpp, KNNBasic
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz


# In[2]:


from surprise import SVD, SVDpp, KNNBasic
from surprise import Dataset
from surprise.model_selection import cross_validate,train_test_split, GridSearchCV
from surprise import NormalPredictor
from surprise import Reader


# In[3]:


import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk


# In[4]:


nltk.download('stopwords')


# In[5]:


print(stopwords.words('english'))


# In[6]:


movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")


# In[7]:


ratings.head()


# In[8]:


movies.head(10)


# In[9]:


ratings_arr = ratings['rating'].unique()
max_rating = np.amax( ratings_arr )
min_rating = np.amin( ratings_arr )
print( ratings_arr )
print(f'max rating:{max_rating} and min rating :{min_rating}')


# In[10]:


movie_dict = pd.Series(movies.movieId.values,index=movies.title).to_dict()
movie_dict_reverse = {val: key for key, val in movie_dict.items()} 
#for key, val in movie_map.items() loops through each key-value pair in movie_map
#The val: key part creates the new key-value pair in the resulting dictionary,
movieId_to_index_map = pd.Series(movies.index.values,index=movies.movieId).to_dict()
movieId_all_array = movies['movieId'].unique()


# In[11]:


print(movie_dict_reverse)
print(movieId_to_index_map)
print(movieId_all_array)


# In[12]:


def get_movieId( movie_name ):
    """
    return the movieId which is corresponding to the movie name

    Parameters
    ----------
    movie_name: string, the name of the movie w/ or w/o the year

    Return
    ------
    the movieId
    """

    # If movie name is 100% equal to a name writen in the database,
    # then return the id corresponding to the name.
    # Or we need to consider the similarity between strings 
    if (movie_name in movie_dict):
      return movie_dict[movie_name]
    else:
      similar = []
      for title, movie_id in movie_dict.items():
        ratio = fuzz.ratio(title.lower(), movie_name.lower())
        if ( ratio >= 60):
          similar.append( (title, movie_id, ratio ) )
      if (len(similar) == 0):
        print("This movie does not exist in the database.")
      else:
        match_item = sorted( similar , key=lambda x: x[2] )[::-1]
        print( "The matched movie in database:", match_item[0][0], ", ratio=",match_item[0][2] )
        return match_item[0][1]


# Content based filtering algorithm with pairwise apporoach in TF-idf vector space

# In[13]:


def tokenizer(text):
  '''The tokenizer function processes a string of text by splitting it into individual words using the pipe character (|),
    removing common English stopwords (like "is" and "the"), converting the remaining words to lowercase,
    and applying stemming using the Porter Stemmer to reduce each word to its root form. 
    This function returns a list of processed tokens, making the text suitable for further analysis in natural language processing tasks.
    For example, the input "Action|Adventure|is|the|Crime" would yield the output ['action', 'adventur', 'crime'].'''
  torkenized = [PorterStemmer().stem(word).lower() for word in text.split('|') if word not in stopwords.words('english')]
  return torkenized


# In[14]:


tfid=TfidfVectorizer(analyzer='word', tokenizer=tokenizer)


# In[15]:


tfidf_matrix = tfid.fit_transform(movies['genres'])


# In[16]:


unique_genres = tfid.get_feature_names_out()
print(len(unique_genres))
print(unique_genres)


# In[17]:


cos_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)


# In[18]:


print(tfidf_matrix.shape)
print(cos_sim.shape)
print(movies.shape)


# Singular value decomposition matrix factorization model for collaborative filtering algorithm

# In[19]:


features = ['userId','movieId', 'rating']
reader = Reader(rating_scale=(min_rating, max_rating))
data = Dataset.load_from_df(ratings[features], reader)
param_grid = {'n_epochs': [5, 14], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
gs_model = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)


# In[20]:


gs_model.fit(data)


# In[21]:


print(gs_model.best_score['rmse'])
print(gs_model.best_params['rmse'])


# In[22]:


best_params = gs_model.best_params['rmse']
model_svd = gs_model.best_estimator['rmse']
model_svd.fit(data.build_full_trainset())


# In[23]:


def get_rating_from_prediction( prediction, ratings_arr ):
    """
    Returns the closest rating from a list of possible ratings to the predicted value.

    Parameters
    ----------
    prediction: float value from the model

    ratings_arr: the 1D array of the possible ratings as per dataset

    Return
    ------
    The rating number from ratings_arr that is closest to the prediction.
    
    The function calculates the absolute difference between the predicted value
    and each rating in the array, and returns the one with the smallest difference
    """
    rating = ratings_arr[ np.argmin( [ np.abs(item - prediction) for item in ratings_arr ] ) ]
    return rating


# In[24]:


prediction = model_svd.predict(1,50)
print("rating", ratings[(ratings.userId ==1 ) & (ratings.movieId ==50 ) ]['rating']  )
print("prediction",prediction.est)


# Recommendation based on Similar movies of User's favourite movie

# In[25]:


def make_recommendation_item_based( similarity_matrix ,movieId_all_array, ratings_data, id_to_movie_map, movieId_to_index_map, fav_movie_list, n_recommendations, userId=-99):
    """
    return top n movie recommendation based on user's input list of favorite movies
    Currently, fav_movie_list only support one input favorate movie

    Parameters
    ----------
    similarity_matrix: 2d array, the pairwise similarity matrix

    movieId_all_array: 1d array, the array of all movie Id

    ratings_data: ratings data

    id_to_movie_map: the map from movieId to movie title

    movieId_to_index_map: the map from movieId to the index of the movie dataframe

    fav_movie_list: list, user's list of favorite movies

    n_recommendations: int, top n recommendations

    userId: int optional (default=-99), the user Id
            if userId = -99, the new user will be created
            if userId = -1, the latest inserted user is chosen

    Return
    ------
    list of top n movie recommendations

    """

    if (userId == -99):
      userId = np.amax( ratings_data['userId'].unique() ) + 1
    elif (userId == -1):
      userId = np.amax( ratings_data['userId'].unique() )

    movieId_list = []
    for movie_name in fav_movie_list:
      movieId_list.append( get_movieId(movie_name) )    

    # Get the movie id which corresponding to the movie the user didn't watch before
    movieId_user_exist = list( ratings_data[ ratings_data.userId==userId ]['movieId'].unique() )
    movieId_user_exist = movieId_user_exist + movieId_list
    movieId_input = []
    for movieId in movieId_all_array:
      if (movieId not in movieId_user_exist):
         movieId_input.append( movieId )


    index = movieId_to_index_map[movieId_list[0]]
    cos_sim_scores=list(enumerate(similarity_matrix[index]))
    cos_sim_scores=sorted(cos_sim_scores,key=lambda x:x[1],reverse=True) 
  
    topn_movieIndex = []
    icount = 0
    for i in range(len(cos_sim_scores)):
      if( cos_sim_scores[i][0] in [movieId_to_index_map[ids] for ids in movieId_input ]  ):
        icount += 1
        topn_movieIndex.append( cos_sim_scores[i][0] )
        if( icount == n_recommendations ):
          break
    
    topn_movie = [ movies.loc[index].title for index in topn_movieIndex ]
    return topn_movie


     


# Make movie recommendation (user-based)

# In[26]:


def make_recommendation_user_based(best_model_params, movieId_all_array, ratings_data, id_to_movie_map,
                        fav_movie_list, n_recommendations, userId=-99 ):
    """
    return top n movie recommendation based on user's input list of favorite movies
    Currently, fav_movie_list only support one input favorate movie


    Parameters
    ----------
    best_model_params: dict, {'iterations': iter, 'rank': rank, 'lambda_': reg}

    movieId_all_array: the array of all movie Id

    ratings_data: ratings data

    id_to_movie_map: the map from movieId to movie title

    fav_movie_list: list, user's list of favorite movies

    n_recommendations: int, top n recommendations

    userId: int optional (default=-99), the user Id
            if userId = -99, the new user will be created
            if userId = -1, the latest inserted user is chosen

    Return
    ------
    list of top n movie recommendations
    """

    movieId_list = []
    for movie_name in fav_movie_list:
      movieId_list.append( get_movieId(movie_name) )

    if (userId == -99):
      userId = np.amax( ratings_data['userId'].unique() ) + 1
    elif (userId == -1):
      userId = np.amax( ratings_data['userId'].unique() )

    ratings_array = ratings['rating'].unique()
    max_rating = np.amax( ratings_array )
    min_rating = np.amin( ratings_array )
    
    # create the new row which corresponding to the input data
    user_rows = [[userId, movieId, max_rating] for movieId in movieId_list]
    df = pd.DataFrame(user_rows, columns =['userId', 'movieId', 'rating']) 
    train_data = pd.concat([ratings_data, df], ignore_index=True, sort=False)

    # Get the movie id which corresponding to the movie the user didn't watch before
    movieId_user_exist = train_data[ train_data.userId==userId ]['movieId'].unique()
    movieId_input = []
    for movieId in movieId_all_array:
      if (movieId not in movieId_user_exist):
         movieId_input.append( movieId )

    reader = Reader(rating_scale=(min_rating, max_rating))

    data = Dataset.load_from_df(train_data, reader)

    model = SVD(**best_model_params)
    model.fit(data.build_full_trainset())

    predictions = []
    for movieId in movieId_input:
      predictions.append( model.predict(userId,movieId) )

    
    sort_index = sorted(range(len(predictions)), key=lambda k: predictions[k].est, reverse=True)
    topn_predictions = [ predictions[i].est for i in sort_index[0:min(n_recommendations,len(predictions))] ]
    topn_movieIds = [ movieId_input[i] for i in sort_index[0:min(n_recommendations,len(predictions))] ]
    topn_rating = [ get_rating_from_prediction( pre, ratings_array ) for pre in topn_predictions ]

    topn_movie = [ id_to_movie_map[ ids ] for ids in topn_movieIds ]
    return topn_movie


# Making a recommendation

# In[40]:


my_favorite_movies = ['Roll Bounce']

# get recommends
n_recommendations = 10

recommends_item_based = make_recommendation_item_based( 
    similarity_matrix = cos_sim,
    movieId_all_array = movieId_all_array,
    ratings_data = ratings[features], 
    id_to_movie_map = movie_dict_reverse, 
    movieId_to_index_map = movieId_to_index_map,
    fav_movie_list = my_favorite_movies, 
    n_recommendations = n_recommendations)

recommends_user_based = make_recommendation_user_based(
    best_model_params = best_params, 
    movieId_all_array = movieId_all_array,
    ratings_data = ratings[features], 
    id_to_movie_map = movie_dict_reverse, 
    fav_movie_list = my_favorite_movies, 
    n_recommendations = n_recommendations)

print("-------------Search based on item's content similarity--------------------")
print('The movies similar to' , my_favorite_movies , ':' )
for i, title in enumerate(recommends_item_based):
    print(i+1, title)  
if( len(recommends_item_based) < n_recommendations ):
  print("Sadly, we couldn't offer so many recommendations :(")    

print("--------------Search based on similarity between user's preference--------------------------------------")
print('The users like' , my_favorite_movies , 'also like:')
for i, title in enumerate(recommends_user_based):
    print(i+1, title)
if( len(recommends_user_based) < n_recommendations ):
  print("Sadly, we couldn't offer so many recommendations :(")

