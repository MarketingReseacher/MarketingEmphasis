import streamlit as st
import pandas as pd

st.sidebar.markdown("# File entry instructions")

Sample = pd.read_csv("Sample.csv")[:3]
Sample['gvkey'] = Sample['gvkey'].astype("string")

st.write("An acceptable file meets the following criteria:")
st.write("- The file format is :red[.CSV]")
st.write("- The first row contains the column names")
st.write("- One column is named :red[text], which contains the CEO, Firm, or Analyst's words")
st.write("  \nPlease note that the column names are :red[case-sensitive]. Including columns other than the 'text' column (e.g., gvkey, year) is optional.")
st.write("  \nHere is an example of an acceptable CSV file:")

st.write(Sample[['gvkey', 'year', 'text']].head())
    