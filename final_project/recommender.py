import pandas as pd
import json

from collections import defaultdict
def build_user_profiles(): 
    """
    Populate a dictionary of user ids to list of movies that they rated.
    Helps in the calculation of similarity between users. 
    """
    user_profile_ratings = {}
    user_profile_genres = {}
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies_metadata.csv')
    links = pd.read_csv('data/links_small.csv', dtype={'imdbId': str})

    for index, row in ratings.iterrows():
        userId, movieId, rating, timestamp = row['userId'], row['movieId'], float(row['rating']), row['timestamp']
        # Get the Imdb ID, which is how we can grab data about the movie.
        try:
            imdbId = "tt" + str(links.loc[links['movieId'] == movieId]['imdbId'].values[0])
        except:
            imdbId = ""

        # Get movie metadata
        movie_data = movies.loc[movies['imdb_id'] == imdbId]

        genres = movie_data['genres'].values[0][1:-1]

        genres = [g + "}" if g[-1] != "}" else g for g in genres.split('}, ')]
        genres = [json.loads(g.replace("'", '"')) for g in genres]

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

    return user_profile_ratings

build_user_profiles()