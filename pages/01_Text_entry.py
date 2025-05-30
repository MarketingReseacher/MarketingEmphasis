# Import packages
import json
import numpy as np
import streamlit as st
import sentence_transformers

# Sidebar: Text entry section
st.sidebar.markdown("# Text entry")

CEOOrAnalyst = st.sidebar.selectbox(
    "Select text source",
    ["CEO or Firm", "Financial analyst"],
    help='Choose "CEO or Firm" if the source of the text is CEO\'s words (e.g., response to analyst questions, excerpts in media interviews, social media posts) or firm documents (e.g., 10-k reports).  \n\nChoose "Financial analyst" if the source of the text is a financial analyst\'s words (e.g., analyst questions in earning calls).'
)

text = st.text_area(
    "Enter text:",
    help="If you don't enter any text, the following text is used as default:  \n\nKey to this ecosystem is the network approach, which leverages cloud networks and information networks to support a wide range of services. This approach enables businesses to offer services and support solutions more effectively. By utilizing application expertise and a diverse skill set, companies can co-engineer and join together multiple modalities to create comprehensive business process solutions. These solutions often include video content management and BPM (Business Process Management) solutions, which are essential in today's digital landscape."
)

if len(text.split()) == 0: 
    text = "Key to this ecosystem is the network approach, which leverages cloud networks and information networks to support a wide range of services. This approach enables businesses to offer services and support solutions more effectively. By utilizing application expertise and a diverse skill set, companies can co-engineer and join together multiple modalities to create comprehensive business process solutions. These solutions often include video content management and BPM (Business Process Management) solutions, which are essential in today's digital landscape."

# Load SentenceTransformer model
model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")

# Normalize function with zero safeguard
def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

# Load construct dictionaries
with open("Market orientation.json", "r") as f:
    MO_dict = json.load(f)
with open("Marketing capabilities.json", "r") as f:
    MC_dict = json.load(f)
with open("Marketing excellence.json", "r") as f:
    ME_dict = json.load(f)

construct_dicts = {
    "Market orientation": MO_dict,
    "Marketing capabilities": MC_dict,
    "Marketing excellence": ME_dict
}

# Compute mean-normalized embeddings for all components
def compute_all_component_embeddings(construct_dicts, model):
    all_embeddings = {}
    for construct, components in construct_dicts.items():
        component_embeddings = {}
        for component, keywords in components.items():
            embeddings = model.encode(keywords)
            mean_vector = normalize_vector(np.mean(embeddings, axis=0))
            component_embeddings[component] = mean_vector
        all_embeddings[construct] = component_embeddings
    return all_embeddings

component_embeddings = compute_all_component_embeddings(construct_dicts, model)

# Embed and normalize input text
text_embedding = normalize_vector(model.encode(text))

# Compute cosine similarity (dot product) for each component
Constructs = {}
Components = {}

for construct, components in component_embeddings.items():
    component_scores = {component: np.dot(text_embedding, vec) for component, vec in components.items()}
    Components[construct] = component_scores
    Constructs[construct] = sum(component_scores.values()) / len(component_scores)

# UI for CEO or Analyst paths
if CEOOrAnalyst == "CEO or Firm":
    Selected_tab = st.sidebar.selectbox(
        "Select desired output", 
        ["Marketing emphasis", "Marketing emphasis lower-order components"], 
        help="Choose 'Marketing emphasis' to see the average Market orientation, Marketing capabilities, and Marketing excellence scores.  \n\nChoose 'Marketing emphasis lower-order components' to see the scores for individual components of each marketing emphasis construct. There are 8 components for Market orientation, 8 for Marketing capabilities, and 3 for Marketing excellence, for a total of 19 distinct components."
    )
    
    if Selected_tab == "Marketing emphasis lower-order components":
        Select = st.sidebar.selectbox(
            "Select Marketing emphasis higher-order construct", 
            ['Market orientation', 'Marketing capabilities', 'Marketing excellence'], 
            help="Market orientation's 8 components: Customer orientation, Competitor orientation, Interfunctional coordination, Long-term focus, Profit focus, Intelligence generation, Intelligence dissemination, and Responsiveness.  \n\nMarketing capabilities' 8 components: Marketing information management, marketing planning, marketing implementation, pricing, Product development, Channel management, Marketing communication, and Selling.   \n\nMarketing excellence's 3 components: Marketing-ecosystem priority, End-user priority, and Marketing-agility priority."
        )

        if Select == "Market orientation" and st.button("Calculate Market orientation's 8 components"):
            st.write("Note: values of each component range from -1 to 1.")
            for component in Components['Market orientation']:
                st.write(f" - {component}: {round(Components['Market orientation'][component], 2)}")

        elif Select == "Marketing capabilities" and st.button("Calculate Marketing capabilities' 8 components"):
            st.write("Note: values of each component range from -1 to 1.")
            for component in Components['Marketing capabilities']:
                st.write(f" - {component}: {round(Components['Marketing capabilities'][component], 2)}")

        elif Select == "Marketing excellence" and st.button("Calculate Marketing excellence's 3 components"):
            st.write("Note: values of each component range from -1 to 1.")
            for component in Components['Marketing excellence']:
                st.write(f" - {component} priority: {round(Components['Marketing excellence'][component], 2)}")
    
    elif Selected_tab == "Marketing emphasis" and st.button("Calculate Market orientation, Marketing capabilities, and Marketing excellence"):
        st.write("Note: values range from -1 to 1.")
        for construct in Constructs:
            st.write(f" - {construct}: {round(Constructs[construct], 2)}")

elif CEOOrAnalyst == "Financial analyst" and st.button("Calculate analyst's Customer orientation"):
    st.write("Note: values range from -1 to 1.")
    score = round(Components["Market orientation"]["Customer orientation"], 2)
    st.write("Customer orientation", score)