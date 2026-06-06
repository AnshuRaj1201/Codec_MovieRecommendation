import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Create a simple dataset
data = {
    'Movie_ID': [1, 2, 3, 4, 5],
    'Title': ['The Matrix', 'Inception', 'The Notebook', 'Avengers', 'Titanic'],
    'Genres': ['Action Sci-Fi', 'Action Sci-Fi Thriller', 'Romance Drama', 'Action Adventure Sci-Fi', 'Romance Drama']
}

movies_df = pd.DataFrame(data)
print("Our Movie Database:\n", movies_df[['Title', 'Genres']], "\n")

# 2. Convert the 'Genres' text into a numerical matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies_df['Genres'])

# 3. Calculate the similarity score between all movies
similarity_scores = cosine_similarity(tfidf_matrix, tfidf_matrix)


# 4. Create the recommendation logic
def get_recommendations(movie_title, dataframe, similarity_matrix):
    # Find the index of the movie the user typed in
    movie_index = dataframe[dataframe['Title'] == movie_title].index[0]
    
    # Get the similarity scores for that specific movie
    scores = list(enumerate(similarity_matrix[movie_index]))
    
    # Sort the movies based on the highest similarity scores (ignoring the first one, which is the movie itself)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:3]
    
    # Print the recommended titles
    print(f"Because you watched '{movie_title}', we recommend:")
    for index, score in sorted_scores:
        recommended_title = dataframe.iloc[index]['Title']
        print(f"- {recommended_title} (Match Score: {score:.2f})")

# 5. Test the system!
get_recommendations('The Matrix', movies_df, similarity_scores)