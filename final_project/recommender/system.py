from compute import load_user_info
import pandas as pd

def recommend(prefs: dict, threshold:float =0.0, n:int = 10):
    """
    This is the function that we should call to generate recommendations. 
    It takes in a prefs dictionary (which is genres:counts pairs). 
    By default, the minimum score for any movie that will be returned is 0.
    The parameter n is how many results it should return. 
    """
    # Load in the relevant dictionaries that store userid:ratings and userid:genres 
    user_genres, user_ratings = load_user_info() 

    ranking = []
    for user, genres_watched in user_genres.items():
        overlapping_genres = set(prefs.keys()).intersection(genres_watched.keys())

        similarity = 0
        for o in overlapping_genres:
            similarity += prefs[o] * genres_watched[o]

        ranking.append((user, similarity))

    # Sort the list of users. Higher score = higher similarity 
    ranking = sorted(ranking, key=lambda x:x[1], reverse=True)

    recommendations = []
    for tup in ranking: 
        user, similarity = tup

        # Lookup the user's reviews/ratings
        reviews = user_ratings[user]

        # List of movies rated is already sorted in descending order of score
        for r in reviews: 
            if len(recommendations) < n: 
                similar_genre = set(r[4]).intersection(overlapping_genres)
                if len(similar_genre) > 0:
                    recommendations.append(r[3]) # Movie title is 4th element in the tuple
            else:
                break
        
        if len(recommendations) >= n: 
            break
    
    return recommendations


def random_recommendation(n:int = 10):
    """
    Randomly select n movies from the dataset.
    """
    movies = pd.read_csv('../data/movies_metadata.csv')
    sample = movies.sample(n=n)
    return sample['original_title'].values


# Sample usage
fake_user_prefs = {'Comedy': 1, 'Animation': 35, 'Family': 1, 'Fantasy': 20}
recs = recommend(fake_user_prefs, threshold=3, n=15)
print(recs)



        
