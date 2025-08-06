# Ensure streamlit is imported
import streamlit as st

# Set up the page configuration
st.set_page_config(page_title="My Projects", layout="wide")

# Set page title and introduction
st.header("My Projects", divider="rainbow")
st.write(
    """
    Welcome to my projects page! Here you'll find a showcase of some of the projects I've worked on.
    """
)

# List projects and links
projects = [
    {
        "name": "Probabilistic Sequence Models",
        "description": "Hidden Markov Models (HMMs) and the Viterbi algorithm implemented in Python. This is work that I did at Eastern University, so the repository is private. To view it, contact me with your email and I will add you as a collaborator.",
        "link": "https://github.com/brea-koenes/probabilistic-sequence-models"
    },
    {
        "name": "N-Gram Language Models",
        "description":"N-gram language models implemented in Python, showcasing the use of n-grams for text prediction and analysis. This is work that I did at Eastern University, so the repository is private. To view it, contact me with your email and I will add you as a collaborator.",
        "link": "https://github.com/brea-koenes/ngram-language-models"
    },
    {
        "name": "Machines Doing the Thinking",
        "description": "Capstone project for Ethics in AI course, exploring the implications of ChatGPT used in coursework.",
        "link": "presentation.mp4"
    }
]

# Display each project
for project in projects:
    st.subheader(project["name"])
    st.write(project["description"])
    if project["link"].endswith(".mp4"):
        st.video(project["link"])
    else:
        st.markdown(f"[View on GitHub]({project['link']})")
    st.markdown("---")
