# import necessary libraries
import streamlit as st
import requests

# establish API endpoint (adjust the port if needed)
API_URL = "http://localhost:8001/score_headlines"

# initialize session state for headlines
if "headlines" not in st.session_state:
    st.session_state.headlines = [""]

st.title("Sentiment Analysis of Headlines - Krithin Arukala - Assignment 3")
st.subheader("Enter headlines:")

# display editable inputs for each headline
for i, headline in enumerate(st.session_state.headlines):
    col1, col2 = st.columns([5, 1])
    with col1:
        st.session_state.headlines[i] = st.text_input(f"Headline {i+1}", value=headline, key=f"headline_{i}")
    with col2:
        if st.button("❌", key=f"delete_{i}"):
            st.session_state.headlines.pop(i)
            st.experimental_rerun()

# add new headline input
if st.button("Add Headline"):
    st.session_state.headlines.append("")

# submit headlines
if st.button("Analyze Sentiment"):
    if not any(h.strip() for h in st.session_state.headlines):
        st.warning("Please enter at least one headline.")
    else:
        # filter out empty headlines
        non_empty_headlines = [h for h in st.session_state.headlines if h.strip()]
        try:
            response = requests.post(API_URL, json={"headlines": non_empty_headlines})
            result = response.json()

            if "labels" in result:
                st.success("Sentiment results:")
                for headline, label in zip(non_empty_headlines, result["labels"]):
                    st.write(f"**{headline}** ➝ *Sentiment: {label}*")
            else:
                st.error("An error occurred: " + result.get("error", "Unknown error"))
        except Exception as e:
            st.error(f"Failed to connect to the API: {e}")