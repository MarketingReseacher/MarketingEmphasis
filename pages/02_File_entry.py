# Import packages
import json
import numpy as np
import pandas as pd
import streamlit as st
import os
import sentence_transformers

# Sidebar for file upload and source selection
st.sidebar.markdown("# File entry")
CEOOrAnalyst = st.sidebar.selectbox(
    "Select text source",
    ["CEO or Firm", "Financial analyst"],
    help='Choose "CEO or Firm" if the source of the text is CEO\'s words (e.g., response to analyst questions, excerpts in media interviews, social media posts) or firm documents (e.g., 10-k reports).  \n\nChoose "Financial analyst" if the source of the text is a financial analyst\'s words (e.g., analyst questions in earning calls).'
)

File = st.file_uploader(
    "Upload .CSV file",
    help="You may run the program with a test input CSV file, which uses sample text for three gvkey-year observations. After the program runs on the test CSV file, you can download the output CSV file."
)

model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")

# Core function to compute cosine similarity scores
def ProcessEachRow(text):
    def load_keywords(filename):
        with open(filename, "r") as f:
            return json.load(f)

    def compute_embeddings(keywords_dict):
        embeddings = {}
        for dim, keywords in keywords_dict.items():
            vecs = model.encode(keywords)
            mean_vec = np.mean(vecs, axis=0)
            mean_vec /= np.linalg.norm(mean_vec)
            embeddings[dim] = mean_vec
        return embeddings

    def compute_scores(construct):
        keywords = load_keywords(f"{construct}.json")
        embeddings = compute_embeddings(keywords)
        text_vec = model.encode(text)
        text_vec /= np.linalg.norm(text_vec)
        return {dim: np.dot(text_vec, emb) for dim, emb in embeddings.items()}

    return {
        "Market orientation": compute_scores("Market orientation"),
        "Marketing capabilities": compute_scores("Marketing capabilities"),
        "Marketing excellence": compute_scores("Marketing excellence")
    }

# Output builders
def OutputFull(df):
    rows = []
    for idx, row in df.iterrows():
        r = row['Concepts']
        rows.append({
            'GVKEY': row.get('gvkey', 'Missing'),
            'Year': row.get('year', 'Missing'),
            'Text': row['text'],
            'Customer orientation': r['Market orientation']['Customer orientation'],
            'Competitor orientation': r['Market orientation']['Competitor orientation'],
            'Interfunctional coordination': r['Market orientation']['Interfunctional coordination'],
            'Long-term focus': r['Market orientation']['Long-term focus'],
            'Profit focus': r['Market orientation']['Profit focus'],
            'Intelligence generation': r['Market orientation']['Intelligence generation'],
            'Intelligence dissemination': r['Market orientation']['Intelligence dissemination'],
            'Responsiveness': r['Market orientation']['Responsiveness'],
            'Marketing information management': r['Marketing capabilities']['Marketing information management'],
            'Marketing planning capabilities': r['Marketing capabilities']['Marketing planning capabilities'],
            'Marketing implementation capabilities': r['Marketing capabilities']['Marketing implementation capabilities'],
            'Pricing capabilities': r['Marketing capabilities']['Pricing capabilities'],
            'Product development capabilities': r['Marketing capabilities']['Product development capabilities'],
            'Channel management': r['Marketing capabilities']['Channel management'],
            'Marketing communication capabilities': r['Marketing capabilities']['Marketing communication capabilities'],
            'Selling capabilities': r['Marketing capabilities']['Selling capabilities'],
            'Marketing ecosystem': r['Marketing excellence']['Marketing ecosystem'],
            'End user': r['Marketing excellence']['End user'],
            'Marketing agility': r['Marketing excellence']['Marketing agility']
        })
    return pd.DataFrame(rows)

def OutputAnalyst(df):
    return pd.DataFrame([{
        'GVKEY': row.get('gvkey', 'Missing'),
        'Year': row.get('year', 'Missing'),
        'Text': row['text'],
        'Customer orientation': row['Concepts']['Market orientation']['Customer orientation']
    } for _, row in df.iterrows()])

# Unified processing logic
def process_input(df):
    df.columns = [c.lower() for c in df.columns]
    if "text" not in df.columns:
        st.markdown(":red[File does not contain a text column. Please refer to the ***File entry instructions*** page for details on an acceptable file format.]")
        return None
    df['Concepts'] = df['text'].apply(ProcessEachRow)
    return df

# Main execution
if File is not None:
    filename, ext = os.path.splitext(File.name)
    if ext.lower() == ".csv":
        df = pd.read_csv(File)
        df = process_input(df)
    else:
        st.markdown(":red[File format is not CSV. Please refer to the ***File entry instructions*** page for details.]")
        df = None
else:
    df = pd.read_csv("Sample.csv")[:3]
    df = process_input(df)

# Display results and download button
if df is not None:
    if CEOOrAnalyst == "CEO or Firm" and st.button("Calculate marketing emphasis' lower-order dimensions"):
        result = OutputFull(df)
        st.write("Output preview")
        st.write(result.head())
        st.download_button("Download output file", result.to_csv(index=False), file_name="MarketingConceptOutput.csv")
    elif CEOOrAnalyst == "Financial analyst" and st.button("Calculate analyst's Customer orientation"):
        result = OutputAnalyst(df)
        st.write("Output preview")
        st.write(result.head())
        st.download_button("Download output file", result.to_csv(index=False), file_name="MarketingConceptOutput.csv")