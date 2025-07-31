import streamlit as st

st.set_page_config(page_title="My Projects", layout="wide")

st.title("My Projects")
st.write(
    """
    Welcome to my projects page! Here you'll find a showcase of some of the projects I've worked on.
    """
)

projects = [
    {
        "name": "Probabilistic Sequence Models",
        "description": "Hidden Markov Models (HMMs) and the Viterbi algorithm implemented in Python",
        "link": "https://github.com/brea-koenes/probabilistic-sequence-models"
    },
    {
        "name": "N-Gram Language Models",
        "description":"N-gram language models implemented in Python, showcasing the use of n-grams for text prediction and analysis.",
        "link": "https://github.com/brea-koenes/ngram-language-models"
    },
    {
        "name": "Machines Doing the Thinking",
        "description": "Capstone project for Ethics in AI course, exploring the implications of ChatGPT used in coursework.",
        "link": "presentation.mp4"
    }
]

for project in projects:
    st.subheader(project["name"])
    st.write(project["description"])
    if project["link"].endswith(".mp4"):
        st.video(project["link"])
    else:
        st.markdown(f"[View on GitHub]({project['link']})")
    st.markdown("---")
