import pandas as pd

import ast
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
# Constants

TITLE_WEIGHT = 2  # Weight for movie titles in TF-IDF calculation

# Load datasets
def load_datasets():
    movies = pd.read_csv("data/movies_metadata.csv")
    credits = pd.read_csv("data/credits.csv")
    keywords = pd.read_csv("data/keywords.csv")
    return movies, credits, keywords

# Preprocess and combine data
def preprocess_data(movies, credits, keywords):
    
   movies['id'] = movies['id'].astype(str)
   credits['id'] = credits['id'].astype(str)
   keywords['id'] = keywords['id'].astype(str)

   movies = movies.merge(credits, on='id')
   movies = movies.merge(keywords, on='id')
   
   
   
   return movies

def clean_data(df):
    
    
    for index, row in df.iterrows():
        
        # Combine title and description for complete text
        text = row['all_info']
        
        # Convert all words to lowercase
        text = text.lower()
      
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
      
        # Remove numbers
        text = ''.join(word for word in text if not word.isdigit())
      
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        text = ' '.join([word for word in words if word not in stop_words])
      
        # Remove excess whitespaces
        text = ' '.join(text.split())
      
        df.at[index, 'all_info'] = text
        

    return df



def extract_language_names(row):
    try:
        # Safely evaluate the string as a list of dictionaries
        obj_list = ast.literal_eval(row)
        # Extract the 'name' field from each dictionary in the list
        language_names = [obj['name'] for obj in obj_list if 'name' in obj]
        return ', '.join(language_names)  # Join names into a single string
    except:
        return None  # In case of any error

# Apply the function to the 'spoken_language' column


# Main execution
#movies, credits, keywords = load_datasets()


#combined_data = preprocess_data(movies, credits, keywords)




#combined_data['keywords'] = combined_data['keywords'].apply(extract_language_names)   
#combined_data['cast'] = combined_data['cast'].apply(extract_language_names)
#combined_data['crew'] = combined_data['crew'].apply(extract_language_names)
#combined_data['production_companies'] = combined_data['production_companies'].apply(extract_language_names)
#combined_data['genres'] = combined_data['genres'].apply(extract_language_names)



#combined_data['all_info'] = combined_data['keywords'] + ', ' + combined_data['cast'] + ', ' + combined_data['overview'] +', ' + combined_data['crew'] + ', ' +combined_data['production_companies']+', ' + combined_data['genres']
                             
#print(combined_data['all_info'])
#combined_data = combined_data[['id', 'adult', 'all_info', 'budget', 'revenue', 'title', 'runtime', 'vote_average']]

#combined_data['all_info'] = combined_data['all_info'].astype(str)
#clean_data(combined_data)

#combined_data.to_csv("final_movies.csv")
#print(combined_data.head(10))

df = pd.read_csv("final_movies.csv")

df['all_info'] = df['all_info'].astype(str)
df = clean_data(df)

df.to_csv("movies.csv")

#print(combined_data['keywords']['name'])
# Example search
