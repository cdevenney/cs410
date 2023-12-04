import pandas as pd
import json
from collections import defaultdict
import os

file_path1 = "user_genres.json"
file_path2 = "user_ratings.json"

def build_user_profiles(): 
    """
    Populate a dictionary of user ids to list of movies that they rated.
    Also populate a dictionary of user ids to a dict of genre-count pairs. 
    Helps in the calculation of similarity between users. 
    """
    user_profile_ratings = {}
    user_profile_genres = {}
    ratings = pd.read_csv('data/ratings_small.csv')
    movies = pd.read_csv('data/movies_metadata.csv')
    links = pd.read_csv('data/links_small.csv', dtype={'imdbId': str})

    for index, row in ratings.iterrows():
        print(index)
        userId, movieId, rating, timestamp = row['userId'], row['movieId'], float(row['rating']), row['timestamp']
        # Get the Imdb ID, which is how we can grab data about the movie.
        try:
            imdbId = "tt" + str(links.loc[links['movieId'] == movieId]['imdbId'].values[0])
        except:
            imdbId = ""

        # Get movie metadata
        movie_data = movies.loc[movies['imdb_id'] == imdbId]
        try:
            genres = movie_data['genres'].values[0][1:-1]

            # Manipulate data to get it in the form of a dict
            genres = [g + "}" if g[-1] != "}" else g for g in genres.split('}, ')]
            genres = [json.loads(g.replace("'", '"')) for g in genres]
        except:
            genres = {}

        # Build dictionaries
        if userId not in user_profile_ratings:
            user_profile_ratings[userId] = [(movieId, rating, timestamp)]
            user_profile_genres[userId] = defaultdict(int) # This will be a collection of genres to counts
            for id_name in genres:
                user_profile_genres[userId][id_name['name']] += 1 
        else:
            user_profile_ratings[userId].append((movieId, rating, timestamp))
            for id_name in genres:
                user_profile_genres[userId][id_name['name']] += 1 

    for k, v in user_profile_ratings.items():
        user_profile_ratings[k] = sorted(v, key=lambda x:x[1], reverse=True)

    # File path to save the dictionary
    file_path1 = "user_genres.json"

    # Writing the dictionary to a file
    with open(file_path1, 'w') as file:
        json.dump(user_profile_genres, file)

    # File path to save the dictionary
    file_path2 = "user_ratings.json"

    # Writing the dictionary to a file
    with open(file_path2, 'w') as file:
        json.dump(user_profile_ratings, file)

if not os.path.isfile("user_genres.json") and not os.path.isfile("user_ratings.json"):
    build_user_profiles()

with open(file_path1, 'r') as file:
    user_genres = json.load(file)

with open(file_path2, 'r') as file:
    user_ratings = json.load(file)

print(len(user_genres.keys()))