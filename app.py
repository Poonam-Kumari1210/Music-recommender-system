import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
# data_path = "D:\\D Drive\\Data Science\\PYCharm Projects\\song recommendation system\\Songs_dataset.csv"
data_path = "Songs_dataset.csv"
df = pd.read_csv(data_path)

# Preprocess the data
df['lyrics'] = df['lyrics'].fillna('')  # Fill NaN values with empty strings

# Vectorize the lyrics using CountVectorizer (Bag of Words)
vectorizer = CountVectorizer()
lyrics_matrix = vectorizer.fit_transform(df['lyrics'])


# Function to recommend songs based on lyrics
def recommend_songs(track_name, df, lyrics_matrix):
    track_idx = df[df['track_name'].str.contains(track_name, case=False, na=False)].index
    if len(track_idx) == 0:
        st.write("No track found with the given name.")
        return None

    # Get the feature vector of the input song
    track_vector = lyrics_matrix[track_idx[0]]

    # Calculate similarity with other songs
    similarity = cosine_similarity(track_vector, lyrics_matrix)

    # Get the top 5 most similar songs
    similar_indices = similarity[0].argsort()[-6:][::-1][1:]
    similar_songs = df.iloc[similar_indices][['track_name', 'artist_name', 'lyrics']]

    return similar_songs


# Apply custom CSS styles to make the app colorful
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
        color: #333333;
    }
    h1 {
        color: #ff4500;
        text-align: center;
        font-family: 'Arial Black', Gadget, sans-serif;
    }
    .stSelectbox label {
        color: #4682b4;
        font-weight: bold;
        font-size: 18px;
    }
    .stButton button {
        background-color: #ffa07a;
        color: white;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #ff6347;
    }
    .stDataFrame {
        border: 2px solid #ffa07a;
    }
    </style>
    """, unsafe_allow_html=True)

# User input
st.title("🎵 Music Recommendation System 🎵")

# Dropdown list of track names
track_name_list = df['track_name'].unique().tolist()
user_input = st.selectbox("Select a Track Name:", track_name_list)

if st.button("Recommend"):
    recommendations = recommend_songs(user_input, df, lyrics_matrix)
    if recommendations is not None:
        st.write("Recommended Songs:")
        st.dataframe(recommendations)
