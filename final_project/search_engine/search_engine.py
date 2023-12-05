import pandas as pd
import numpy as np
def preprocess_query(query, split_terms=True):
    # Lowercase the query
    if not query:
        return None
    query = query.lower()
    if split_terms:
        # Split by commas and strip spaces
        return [word.strip().replace(" ","") for word in query.split(',')]
    else:
        # For single-word queries, just remove extra spaces
        return query.strip().replace(" ","")

def calculate_match_score(query_terms, document, weight, is_title=False):
    
    score = 0
    document_lower = document.lower()  # Convert document to lowercase and strip spaces
    
    if not query_terms:
        return 0
    
    elif is_title:
        # For titles, check exact match
       if query_terms == document_lower:
           score += weight
    else:
        # For other fields, check if any term is in the document
        for term in query_terms:
            if term in document_lower:
                score += weight

    return score
# Load the dataset
df = pd.read_csv("../data/final_movies.csv")


s = "her"

test = "her "
#print(s==test)
# Constants for weights
TITLE_WEIGHT = 50
ACTORS_WEIGHT = 5
CREW_WEIGHT = 4
KEYWORDS_WEIGHT = 2

# Preprocess and score each field
query_title = "GoodFellas"
query_actors = "Robert De Niro"
query_crew = 'Martin Scorsese'
query_keywords = "crime,thriller, adventure, drama"

title_term = preprocess_query(query_title, split_terms=False) 
actors_terms = preprocess_query(query_actors, split_terms=True)
crew_terms = preprocess_query(query_crew, split_terms=True)
keywords_terms = preprocess_query(query_keywords, split_terms=True)

df.fillna('', inplace=True)

print(keywords_terms)
# Calculate scores
df['relevance_score'] = (
    df['title'].apply(lambda x: calculate_match_score(title_term, x, TITLE_WEIGHT, is_title=True)) +
    df['cast'].apply(lambda x: calculate_match_score(actors_terms, x, ACTORS_WEIGHT)) +
    df['crew'].apply(lambda x: calculate_match_score(crew_terms, x, CREW_WEIGHT)) +
    df['keywords'].apply(lambda x: calculate_match_score(keywords_terms, x, KEYWORDS_WEIGHT))
)



df['title'].replace('', np.nan, inplace=True)
df.dropna(subset=['title'], inplace=True)
# Sort by relevance score and display results
sorted_df = df.sort_values(by=['relevance_score', 'vote_average'], ascending=False)
print(sorted_df[['title', 'relevance_score', 'vote_average']].head(20))
