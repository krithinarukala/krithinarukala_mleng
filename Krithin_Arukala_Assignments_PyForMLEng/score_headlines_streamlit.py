# Krithin Arukala
# Assignment 3
# User Interface for Score Headlines API
'''
This Streamlit app provides a streamlit user interface for users to input headlines,
send them to the sentiment analysis API,
and display the sentiment results.
'''
# import necessary libraries
import streamlit as st
import requests
# establish API endpoint (adjust the port if needed)
url_for_api = "http://localhost:8001/score_headlines"
# initialize session state for headlines
if "headlines" not in st.session_state:
    st.session_state.headlines = [""]
# set page title
st.title("Sentiment Analysis of Headlines - Krithin Arukala - Assignment 3")
# instructional text for users to enter headlines
st.subheader("Enter headlines:")
# display inputs for each headline that the users can edit
for i, headline in enumerate(st.session_state.headlines):
    # layout columns for headline input and delete button
    col1, col2 = st.columns([5, 1])
    # headline input
    with col1:
        # text input for headline
        st.session_state.headlines[i] = st.text_input(f"Headline {i+1}", value=headline, key=f"headline_{i}")
    with col2:
        # delete button for headline
        if st.button("Remove", key=f"delete_{i}"):
            # remove the headline from the list
            st.session_state.headlines.pop(i)
            # rerun the app to reflect changes
            st.experimental_rerun()

# add new headline input button for new headlines
if st.button("Add Another Headline"):
    # append function to add a new empty headline
    st.session_state.headlines.append("")

# submit headlines
if st.button("Analyze Sentiment of Headline(s)"):
    # check if at least one headline is non-empty
    if not any(h.strip() for h in st.session_state.headlines):
        # warning message if no headlines are entered
        st.warning("please enter at least one headline.")
    # if there are headlines to analyze
    else:
        # filter out empty headlines
        headlines_with_values = [h for h in st.session_state.headlines if h.strip()]
        # send headlines to the API
        try:
            # make a POST request to the API with the headlines
            resp = requests.post(url_for_api, json={"headlines": headlines_with_values})
            # parse the JSON resp
            result = resp.json()
            # check if the resp contains sentiment labels
            if "labels" in result:
                # display the sentiment results
                st.success("sentiment results:")
                # for each headline and its corresponding label
                for headline, label in zip(headlines_with_values, result["labels"]):
                    # display the headline with its sentiment label
                    st.write(f"**{headline}** ➝ *headline sentiment: {label}*")
            else:
                # display an error message if the API resp is invalid
                st.error("an error has occurred: " + result.get("error", "Unknown error"))
        # handle exceptions during the API request
        except Exception as e:
            # display an error message if the API request fails
            st.error(f"the system failed to connect to the API, see error: {e}")