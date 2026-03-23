import streamlit as st
from app import recommend
st.title("Movie Recommender System")
movie_name = st.text_input("Enter a movie name:")
if st.button("Recommend"):
    if movie_name:
        results = recommend(movie_name)
        for movie in results:
            st.write(movie)