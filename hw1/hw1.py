import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
import random
import math
from gensim.models import Word2Vec
import numpy as np


#                   #                            
#      PART 1       #
#                   #
#                   #

nltk.download('stopwords')


num_rows = sum(1 for line in open('train.csv')) - 1

#function to generate skip index to generate subset of 12500
skip_idx = random.sample(range(1, num_rows + 1), num_rows - 12500)

# Read in only the non-skipped rows
df = pd.read_csv('train.csv', skiprows=skip_idx)



def clean_data(df):
    
    cleaned_data = []
    
        
    for index, row in df.iterrows():
        title = row[1]
        description = row[2]
      
        # Combine title and description for complete text
        text = f"{title} {description}"
      
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
      
        # Append to cleaned_data list
        cleaned_data.append(text)

    return cleaned_data

c_data = clean_data(df)
# Concatenate all cleaned data texts
all_text = ' '.join(c_data)


word_tokens = all_text.split()

word_freq = Counter(word_tokens)

top_200_words = word_freq.most_common(200)

vocab = [word for word, freq in top_200_words]

with open('PART 1.txt', 'w') as f:
    print(top_200_words, file=f)


#                   #                            
#      PART 2       #
#    question 1     #
#                   #



#this function returns 1 if the word is in the text of a document and returns 0 otherwise
def simple_bit_vector(text, word):
    text = set(text.split())
    if word in text:
        return 1 
    else:
        return 0

#keep track of word rankings for each word in vocab (doc frequency)
ranking_dict = {}

for word in vocab:
        
    total_matches = 0
    
    for doc in c_data:
        total_matches += simple_bit_vector(doc, word)
        
    ranking_dict[word] = total_matches

sorted_word_relevance = sorted(ranking_dict.items(), key=lambda x: x[1], reverse=True)[:10]


    
# Print the top 10 highest-ranked words and their ranking function value (number of matches)
with open('PART 2 question 1.txt', 'w') as f:
    print("Top 10 most relevant words:", sorted_word_relevance, file =f)



#                   #                            
#      PART 2       #
#    question 2     #
#                   #
#for this part, I used the cleaned data as the document and then returned the clean data. 
#it makes things a little less clear but the result is still the same. It calculates the
#relevance of each document for each query. (The max would be 3 and the min would be 0)
#because there's only 3 words in each query. 
queries = ['olympic gold athens', 'reuters stocks friday', 'investment market prices']

relevance_scores = {}

def bit_vector_relevance(data, query):
    query_relevance = []
    if isinstance(query, str):
        query = query.split()

    for doc in data:
        score = 0
        for w in query:
            score += simple_bit_vector(doc, w)
        query_relevance.append((doc, score))
    
    query_relevance.sort(key=lambda x: x[1])

    top_5 = query_relevance[-5:]
    bottom_5 = query_relevance[:5]

    return top_5, bottom_5
    
for query in queries:
    top_5, bottom_5 = bit_vector_relevance(c_data, query)
    relevance_scores[query] = {'Top 5': top_5, 'Bottom 5': bottom_5}

with open('PART 2 question 2.txt', 'w') as f:
    for query, result in relevance_scores.items():
        print(f"Results for query: '{query}'", file = f)
        print("\n", file = f)
        print("Top 5 most relevant:", result['Top 5'], file = f)
        print("\n", file = f)
        print("Bottom 5 least relevant:", result['Bottom 5'], file = f)
        print("\n", file = f)




#                   #                            
#      PART 2       #
#    question 3     #
#                   #
#this part is very similar to the last, but instead of doing it for the queries, we will
#do it for the vocab we found for the test set. We will then test this on the "test" set
#note that we do not use a subset of test

df2 = pd.read_csv('test.csv')

test_relevance_scores = {}

test_c_data = clean_data(df2)

test_all_text = ' '.join(c_data)


test_word_tokens = all_text.split()

test_word_freq = Counter(word_tokens)
test_top_200_words = word_freq.most_common(200)

test_vocab = [word for word, freq in test_top_200_words]



top_5, bottom_5 = bit_vector_relevance(test_c_data, test_vocab)
test_relevance_scores = {'Top 5': top_5, 'Bottom 5': bottom_5}

with open('PART 2 question 3.txt', 'w') as f:
    
        print(f"Results for test_vocab: '{test_vocab}'", file = f)
        print("\n", file = f)
        print("Top 5 most relevant:", top_5, file = f)
        print("\n", file = f)
        print("Bottom 5 least relevant:", bottom_5, file = f)
        print("\n", file = f)


#                   #                            
#      PART 3       #
#    question 1     #
#                   #

def compute_df(word, data):
    
    total = 0
    
    for doc in data:
        text = doc.split()
        if(word in text):
            total += 1
            
    return total

#Now we will do the same thing, but instead of bit vector representation we will do it for
#okapi bm25.



#note that c(w,q) will always be 1 because in each of our queries each word is unique
#and for our vocab calculations each word in the vocab is unique. 

def bm25_okapi_word_relevance_score(M, word_count, df_w, k):
    #arbitrary k selection
    log_val = math.log((M+1)/df_w)
    
    score = (k+1)*word_count*log_val
    
    score = score/(word_count + k)
    
    return score

bm25_okapi_word_rankings = {}
M = len(c_data)

# Pre-calculate document frequencies for all vocab words
df_w_cache = {word: compute_df(word, c_data) for word in vocab}

for word in vocab:
    df_w = df_w_cache[word]  # Retrieve pre-computed df_w
    rel_score = 0

    for doc in c_data:
        word_counts = Counter(doc.split())  # Calculate word counts only once for each doc
        word_count = word_counts.get(word, 0)  # Retrieve pre-computed word_count
        
        if word_count > 0:
            rel_score += bm25_okapi_word_relevance_score(M, word_count, df_w, k=2)

    bm25_okapi_word_rankings[word] = rel_score

bm25_sorted_word_relevance = sorted(bm25_okapi_word_rankings.items(), key=lambda x: x[1], reverse=True)[:10]

with open('PART 3 question 1.txt', 'w') as f:
    print("Top 10 most relevant words:", bm25_sorted_word_relevance, file=f)
    
    
    
    
#                   #                            
#      PART 3       #
#    question 2     #
#                   #


bm25_relevance_scores = {}

def bm25_okapi_relevance(data, query):
    M = len(data)
    bm25_query_relevance = []

    # Pre-calculate document frequencies for all query words.
    df_w_cache = {}
    if isinstance(query, str):
        query = query.split()
    
    for word in query:
        df_w_cache[word] = compute_df(word, data)

    for doc in data:
        score = 0
        word_counts = Counter(doc.split())  # Calculate word counts only once for each doc

        for word in query:
            df_w = df_w_cache.get(word, 0)  # Retrieve pre-computed df_w
            word_count = word_counts.get(word, 0)  # Retrieve pre-computed word_count
            
            if word_count > 0:
                score += bm25_okapi_word_relevance_score(M, word_count, df_w, k=2)
        
        bm25_query_relevance.append((doc, score))

    bm25_query_relevance.sort(key=lambda x: x[1])
    top_5 = bm25_query_relevance[-5:]
    bottom_5 = bm25_query_relevance[:5]

    return top_5, bottom_5


for query in queries:
    top_5, bottom_5 = bm25_okapi_relevance(c_data, query)
    bm25_relevance_scores[query] = {'Top 5': top_5, 'Bottom 5': bottom_5}

with open('PART 3 question 2.txt', 'w') as f:
    for query, result in bm25_relevance_scores.items():
        print(f"Results for query: '{query}'", file = f)
        print("\n", file = f)
        print("Top 5 most relevant:", result['Top 5'], file = f)
        print("\n", file = f)
        print("Bottom 5 least relevant:", result['Bottom 5'], file = f)
        print("\n", file = f)
        
#                   #                            
#      PART 3       #
#    question 3     #
#                   #

top_5, bottom_5 = bm25_okapi_relevance(test_c_data, test_vocab)
test_relevance_scores = {'Top 5': top_5, 'Bottom 5': bottom_5}

with open('PART 3 question 3.txt', 'w') as f:
    
        print(f"Results for test_vocab: '{test_vocab}'", file = f)
        print("\n", file = f)
        print("Top 5 most relevant:", top_5, file = f)
        print("\n", file = f)
        print("Bottom 5 least relevant:", bottom_5, file = f)
        print("\n", file = f)


#                   #                            
#      PART 4       #
#    question 1     #
#                   #
#now we are using the word2vec model. The process of computing the relevance scores is the same, but this time we
#are using word2vec to calculate the score. I used the gensim library for this.
#additionally, I computed the average_log_likelihood by using the most_similar method which
#finds the top most similar words in the training data and then returns the cosine similar

tokenized_data = [doc.split() for doc in c_data]
model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
model.save("train_word2vec.model")

def average_log_likelihood(word, model):
    #using cosine simliarity between words as the probability, we can calculate the average
    #likelihood among the returned similar words (each one has its own value)
    similar_words = model.wv.most_similar(word)
    log_likelihoods = [np.log(similarity) for x, similarity in similar_words]
    return np.mean(log_likelihoods)

w2v_word_relevances = {}
for word in vocab:
    try:
        w2v_word_relevances[word] = average_log_likelihood(word, model)
    except KeyError:  # Word not in Word2Vec vocab
        w2v_word_relevances[word] = 0

# Sort words by relevance and get the top 10
w2v_sorted_words = sorted(w2v_word_relevances.items(), key=lambda x: x[1], reverse=True)[:10]

with open('PART 4 question 1.txt', 'w') as f:
    
    print("Top 10 most relevant words:" , w2v_sorted_words, file = f)
    
    
#                   #                            
#      PART 4       #
#    question 2     #
#                   #

#for this,instead of calculating the similarity scores for words, I implemented the softmax
#function and then took the dot product of each softmax for a query and the softmax for a doc
#this way I was able to achieve a similarity score based on the vector representations of the query
#and document. Reference for converting a query/doc to a vector: https://stackoverflow.com/questions/60561960/how-to-convert-the-text-into-vector-using-word2vec-embedding


#code for computing softmax, source: https://wandb.ai/krishamehta/softmax/reports/How-to-Implement-the-Softmax-Function-in-Python--VmlldzoxOTUwNTc#:~:text=%EF%BB%BFSummary%EF%BB%BF-,What%20Is%20The%20Softmax%20Function%3F,the%20distribution%20add%20to%201.
def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

#code for converting a word to a vector
def word_to_vec(word, model):
    return model.wv[word] if word in model.wv.index_to_key else np.zeros(model.vector_size)

#calculate the softmax value for a doc/query
def words_to_softmax(words, model):
    if isinstance(words, str):
        words = words.split()
    
    return [softmax(word_to_vec(word, model).reshape(1,-1)) for word in words if word in model.wv.index_to_key]

#relevance is given by the dot product of the softmaxes. Had to flatten each one in order for
#the implementation to work. 
def calculate_relevance(query, document, model):
    query_softmax = np.array(words_to_softmax(query, model)).sum(axis=0)
    doc_softmax = np.array(words_to_softmax(document, model)).sum(axis=0)
    return np.dot(query_softmax.flatten(), doc_softmax.flatten())


word2vec_relevance_scores = {}
#top level function to operate on each document
def word2vec_relevance(data, query, model):
    word2vec_relevances = []
    for doc in data:
        score = calculate_relevance(query,doc,model)
        word2vec_relevances.append((doc,score))
        
    word2vec_relevances.sort(key=lambda x: x[1])
    top_5 = word2vec_relevances[-5:]
    bottom_5 = word2vec_relevances[:5]

    return top_5, bottom_5


for query in queries:
    top_5, bottom_5 = word2vec_relevance(c_data, query, model)
    word2vec_relevance_scores[query] = {'Top 5': top_5, 'Bottom 5': bottom_5}

with open('PART 4 question 2.txt', 'w') as f:
    for query, result in word2vec_relevance_scores.items():
        print(f"Results for query: '{query}'", file = f)
        print("\n", file = f)
        print("Top 5 most relevant:", result['Top 5'], file = f)
        print("\n", file = f)
        print("Bottom 5 least relevant:", result['Bottom 5'], file = f)
        print("\n", file = f)
        
        
#                   #                            
#      PART 4       #
#    question 3     #
#                   #

#Same exact process, but I used the test_vocab as the query
tokenized_data = [doc.split() for doc in test_c_data]
test_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
test_model.save("test_word2vec.model")


top_5, bottom_5 = word2vec_relevance(test_c_data, test_vocab, test_model)
test_relevance_scores = {'Top 5': top_5, 'Bottom 5': bottom_5}

with open('PART 4 question 3.txt', 'w') as f:
    
        print(f"Results for test_vocab: '{test_vocab}'", file = f)
        print("\n", file = f)
        print("Top 5 most relevant:", top_5, file = f)
        print("\n", file = f)
        print("Bottom 5 least relevant:", bottom_5, file = f)
        print("\n", file = f)
        
        