# make a streamlit app for movie-ratings-and-recommendation-using-knn

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import operator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy import spatial


def Similarity(movieId1, movieId2):
    a = movies.iloc[movieId1]
    b = movies.iloc[movieId2]
    
    genresA = a['genres_bin']
    genresB = b['genres_bin']
    
    genreDistance = spatial.distance.cosine(genresA, genresB)
    
    scoreA = a['cast_bin']
    scoreB = b['cast_bin']
    scoreDistance = spatial.distance.cosine(scoreA, scoreB)
    
    directA = a['director_bin']
    directB = b['director_bin']
    directDistance = spatial.distance.cosine(directA, directB)
    
    wordsA = a['words_bin']
    wordsB = b['words_bin']
    wordsDistance = spatial.distance.cosine(directA, directB)
    return genreDistance + directDistance + scoreDistance + wordsDistance

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def predict_score(name):
    #name = input('Enter a movie title: ')
    new_movie = movies[movies['original_title'].str.contains(name)].iloc[0].to_frame().T
    def getNeighbors(baseMovie, K):
        distances = []
    
        for index, movie in movies.iterrows():
            if movie['new_id'] != baseMovie['new_id'].values[0]:
                dist = Similarity(baseMovie['new_id'].values[0], movie['new_id'])
                distances.append((movie['new_id'], dist))
    
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
    
        for x in range(K):
            neighbors.append(distances[x])
        return neighbors

    K = 5
    avgRating = 0
    neighbors = getNeighbors(new_movie, K)
    
    recommended_movies = []
    
    for neighbor in neighbors:
        movie_id = neighbor[0]
        title = movies.iloc[movie_id]['original_title']
        genres = movies.iloc[movie_id]['genres']
        rating = movies.iloc[movie_id][2]
        
        recommended_movies.append((title, genres, rating))
    
    return recommended_movies

movies_dict = pickle.load(open('movies_dict.pbz2','rb'))
movies = pd.DataFrame(movies_dict)
# Similarity = pickle.load(open('similarity.pkl','rb'))

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Enter movie name',
    movies['original_title'].values
)

if st.button('Get Recommendations'):
    recommended_movies = predict_score(selected_movie_name) 
    recommended_movies = sorted(recommended_movies, key=lambda x: x[2], reverse=True)
 # Get top 5 recommendations
    
    st.header('Top 5 Recommended Movies:')
    
    for title, genres, rating in recommended_movies:
        # Find the movie ID based on the title
        movie_id = movies[movies['original_title'] == title]['new_id'].values[0]
        
        st.subheader(title)
        
        # Call fetch_poster to get the movie poster URL
        # poster_url = fetch_poster(movie_id)
        # st.image(poster_url, use_column_width=True)
        
        st.write(f"Predicted Rating: {rating}")
