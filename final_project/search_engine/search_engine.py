import pandas as pd

def preprocess_query(query):
    # Lowercase and split by commas, then strip spaces
    return [word.strip().lower() for word in query.split(',')]

def calculate_match_score(query_terms, document, weight):
    
    if not query_terms:
        return 0
    
    score = 0
    document_lower = document.lower()  # Convert document to lowercase for matching
    for term in query_terms:
        # Check if the term is in the document and add the weight once if it is
        if term in document_lower:
            score += weight
    return score

# Load the dataset
df = pd.read_csv("final_movies.csv")

# Constants for weights
TITLE_WEIGHT = 3
ACTORS_WEIGHT = 2
CREW_WEIGHT = 1
KEYWORDS_WEIGHT = 1

# Preprocess and score each field
query_title = ""
query_actors = "tomhardy"
query_crew = "christophernolan"
query_keywords = "action, sciencefiction"

title_terms = preprocess_query(query_title)
actors_terms = preprocess_query(query_actors)

print(actors_terms)
crew_terms = preprocess_query(query_crew)
keywords_terms = preprocess_query(query_keywords)

df.fillna('', inplace=True)
df['title'] = df['title'].str.lower()  # Lowercase titles for matching

# Calculate scores
df['relevance_score'] = (
    df['title'].apply(lambda x: calculate_match_score(title_terms, x, TITLE_WEIGHT)) +
    df['cast'].apply(lambda x: calculate_match_score(actors_terms, x, ACTORS_WEIGHT)) +
    df['crew'].apply(lambda x: calculate_match_score(crew_terms, x, CREW_WEIGHT)) +
    df['keywords'].apply(lambda x: calculate_match_score(keywords_terms, x, KEYWORDS_WEIGHT))
)

# Sort by relevance score and display results
sorted_df = df.sort_values(by='relevance_score', ascending=False)
print(sorted_df[['title', 'relevance_score', 'vote_average']].head(20))
