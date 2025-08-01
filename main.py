import streamlit as st

st.set_page_config(page_title="Product Outage Classifier", layout="centered")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
        .big-font {
            font-size:2.2em !important;
            font-weight:bold;
        }
        .subtitle {
            font-size:1.2em !important;
            color: #555;
        }
        .bio-text {
            font-size: 1em !important;
            color: #222;
            line-height: 1.7;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-font">Welcome to my Product Outage Classifier App! </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Use the sidebar to navigate to my home page, projects, résumé, and app.</div>', unsafe_allow_html=True)

st.header("Biography", divider="rainbow")
st.markdown(
    """
    <div class="bio-text">
    Hi, my name is <b>Brea Koenes</b>.<br><br>
    I'm a professional with 3 years of experience in Corporate Technology at Starbucks, working as a Systems Analyst focused on building reliable systems in cutting-edge cloud platforms.<br><br>
    I received my undergraduate degree in Data Science in 2022, and am graduating with my Master's in Data Science from Eastern University this August 2025.<br><br>
    My career aspirations are to continue working in technology and to pursue my commitment to lifelong learning.<br><br>
    When I'm not working, I'm enjoying hikes in my home state, Washington.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)
