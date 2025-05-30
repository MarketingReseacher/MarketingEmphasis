# Import packages
import streamlit as st

st.markdown("## Welcome")
st.write("A firm’s marketing emphasis is reflected in its market orientation, marketing capabilities, and marketing excellence, and their 19 lower-order components.")
st.write("This website applies a pre-trained Large Language Model (Sentence-BERT) to embed the CEO-, Firm-, or Analyst-specific text that you provide. \n\nNext, it applies semantic projection to measure how closely the meaning of your text aligns with the embedding vectors for the terms in the dictionaries of each of the 19 marketing emphasis components. \n\nEach component of a firm's marketing emphasis is measured as the **Cosine similarity** between the input text embeddings and the centroid of the embeddings of each component.")

st.write("Visit the **Text entry** page to get the measures of a firm's marketing emphasis using a single text entry (e.g., sentence or paragraph).  \n\nAlternatively, if you are looking to use the website for measuring the marketing emphasis for multiple firms, or a firm's marketing emphasis in various time periods, visit the **File entry** page of the website to upload a .CSV file. The **File entry instructions** page details the acceptable file format.")
st.sidebar.markdown("# Main page")
