import pandas as pd
import ast


from nltk.corpus import stopwords


# Load datasets
def load_datasets():
    movies = pd.read_csv("data/movies_metadata.csv")
    credits = pd.read_csv("data/credits.csv")
    keywords = pd.read_csv("data/keywords.csv")
    return movies, credits, keywords

# Preprocess and combine data
def preprocess_data(movies, credits, keywords):
    # Ensure IDs are strings
    movies['id'] = movies['id'].astype(str)
    credits['id'] = credits['id'].astype(str)
    keywords['id'] = keywords['id'].astype(str)

    # Merge datasets
    movies = movies.merge(credits, on='id')
    movies = movies.merge(keywords, on='id')

    # Concatenate genre and keywords
    return movies

# Cleaning crew and cast names
#def clean_names(names):
    # Lowercase and remove whitespace
   # print("made it to clean names")
    
    #return [''.join(name.lower().split()) for name in ast.literal_eval(names)]

# Clean the data for TF-IDF
#def clean_data(df):
    #for index, row in df.iterrows():
        # Convert all words to lowercase
        
        #print("made it into clean_data")
        #row['title'] = row['title'].lower().replace(" ", "")

        # Cleaning names for 'crew' and 'cast'
        
        #row['crew'] = clean_names(row['crew'])
        #row['cast'] = clean_names(row['cast'])
        #row['production_companies'] = clean_names(row['production_companies'])
        
        #print("did we accomplish somethin")

    #return df

# Process user input
def process_user_input(input_str):
    # Split input by commas and clean each part
    return [part.lower().replace(" ", "") for part in input_str.split(',')]

def extract_names(row):
    if pd.isnull(row):
        return None
    try:
        # Safely evaluate the string as a list of dictionaries
        obj_list = ast.literal_eval(row)
        # Extract the 'name' field from each dictionary in the list
        names = [obj['name'].lower().replace(" ", "") for obj in obj_list if 'name' in obj]
        return ', '.join(names)  # Join names into a single string
    except SyntaxError:
        print(f"SyntaxError in row: {row}")
        return None
    except Exception as e:
        print(f"Error {e} in row: {row}")
        return None



# Main execution
movies, credits, keywords = load_datasets()

combined_data = preprocess_data(movies, credits, keywords)


combined_data['keywords'] = combined_data['keywords'].apply(extract_names)   

#print(combined_data["keywords"].head(10))

combined_data['genres'] = combined_data['genres'].apply(extract_names)

#print(combined_data.columns)

#print(combined_data["genres"].head(10))

combined_data['keywords'] = combined_data['genres'].astype(str) + ', ' + combined_data['keywords'].astype(str)


combined_data['cast'] = combined_data['cast'].apply(extract_names).astype(str)

print(combined_data['cast'].head(1))
combined_data['crew'] = combined_data['crew'].apply(extract_names).astype(str)
combined_data['production_companies'] = combined_data['production_companies'].apply(extract_names).astype(str)




# Apply cleaning functions


# Save to CSV
combined_data.to_csv("final_movies.csv")


