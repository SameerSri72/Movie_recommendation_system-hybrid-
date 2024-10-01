from surprise import SVD, SVDpp, KNNBasic
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from surprise import SVD, SVDpp, KNNBasic
from surprise import Dataset
from surprise.model_selection import cross_validate,train_test_split, GridSearchCV
from surprise import NormalPredictor
from surprise import Reader
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import streamlit as st


# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Tokenizer for content-based filtering
nltk.download('stopwords')
stop_words = stopwords.words('english')
ratings_arr = ratings['rating'].unique()
max_rating = np.amax( ratings_arr )
min_rating = np.amin( ratings_arr )


# Movie dictionaries
movie_dict = pd.Series(movies.movieId.values, index=movies.title).to_dict()
movie_dict_reverse = {val: key for key, val in movie_dict.items()}
movieId_to_index_map = pd.Series(movies.index.values, index=movies.movieId).to_dict()
movieId_all_array = movies['movieId'].unique()
features = ['userId','movieId', 'rating']
reader = Reader(rating_scale=(min_rating, max_rating))
data = Dataset.load_from_df(ratings[features], reader)

# Streamlit App Layout
st.title("Movie recommender system")
st.write("Made by Sameer Srivastava")

# User input: Favorite movies
favorite_movie = st.selectbox("Select your favorite movie", movies['title'].tolist())
fav_movie_list = [favorite_movie]
num_recommendations = 10

if favorite_movie:
    def get_movieId(movie_name):
        """Returns movieId from the movie name"""
        if movie_name in movie_dict:
            return movie_dict[movie_name]
        else:
            similar = []
            for title, movie_id in movie_dict.items():
                ratio = fuzz.ratio(title.lower(), movie_name.lower())
                if ratio >= 60:
                    similar.append((title, movie_id, ratio))
            if len(similar) == 0:
                return None
            else:
                match_item = sorted(similar, key=lambda x: x[2], reverse=True)
                return match_item[0][1]
    def tokenizer(text):
        torkenized = [PorterStemmer().stem(word).lower() for word in text.split('|') if word not in stopwords.words('english')]
        return torkenized
    tfid=TfidfVectorizer(analyzer='word', tokenizer=tokenizer)
    tfidf_matrix = tfid.fit_transform(movies['genres'])
    cos_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)

    def get_rating_from_prediction( prediction, ratings_arr ):
        rating = ratings_arr[ np.argmin( [ np.abs(item - prediction) for item in ratings_arr ] ) ]
        return rating


    # Content-based recommendation function
    def make_recommendation_item_based( similarity_matrix ,movieId_all_array, ratings_data, id_to_movie_map, movieId_to_index_map, fav_movie_list, n_recommendations, userId=-99):

        if (userId == -99):
            userId = np.amax( ratings_data['userId'].unique() ) + 1
        elif (userId == -1):
            userId = np.amax( ratings_data['userId'].unique() )

        movieId_list = []
        for movie_name in fav_movie_list:
            movieId = get_movieId(movie_name)
            if movieId:
                movieId_list.append(movieId)
            else:
                st.error(f"Movie '{movie_name}' not found.")
                return []
        if len(movieId_list) == 0:
            st.error("No valid movie IDs found for recommendation.")
            return []    

    # Get the movie id which corresponding to the movie the user didn't watch before
        movieId_user_exist = list( ratings_data[ ratings_data.userId==userId ]['movieId'].unique() )
        movieId_user_exist = movieId_user_exist + movieId_list
        movieId_input = []
        for movieId in movieId_all_array:
            if (movieId not in movieId_user_exist):
                movieId_input.append( movieId )


        index = movieId_to_index_map[movieId_list[0]]
        if index is None:
            st.error(f"Movie ID {movieId_list[0]} not found in the map.")
            return []
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

    # Collaborative filtering recommendation function
    def make_recommendation_user_based(best_model_params, movieId_all_array, ratings_data, id_to_movie_map,
                        fav_movie_list, n_recommendations, userId=-99 ):
    

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
    

    # Best model parameters (use the ones from GridSearchCV in your notebook)
    best_params = {'n_epochs': 14, 'lr_all': 0.005, 'reg_all': 0.4}

    # Generate recommendations
    recommends_item_based = make_recommendation_item_based( 
        similarity_matrix = cos_sim,
        movieId_all_array = movieId_all_array,
        ratings_data = ratings[features], 
        id_to_movie_map = movie_dict_reverse, 
        movieId_to_index_map = movieId_to_index_map,
        fav_movie_list = [favorite_movie], 
        n_recommendations = num_recommendations)

    recommends_user_based = make_recommendation_user_based(
    best_model_params = best_params, 
    movieId_all_array = movieId_all_array,
    ratings_data = ratings[features], 
    id_to_movie_map = movie_dict_reverse, 
    fav_movie_list = [favorite_movie], 
    n_recommendations = num_recommendations)
    st.subheader(f"Movies similar to '{favorite_movie}' (Item-based):")
    if recommends_item_based:
        for i, title in enumerate(recommends_item_based):
            st.write(f"{i+1}. {title}")
        if( len(recommends_item_based) < num_recommendations ):
            st.write("Sadly, we couldn't offer so many recommendations :(")
    st.subheader(f"Movies liked by users who liked '{favorite_movie}' (User-based):")
    if recommends_user_based:
        for i, title in enumerate(recommends_user_based):
            st.write(f"{i+1}. {title}")
        if( len(recommends_user_based) < num_recommendations ):
            st.write("Sadly, we couldn't offer so many recommendations :(")