import streamlit as st

# Page setup
st.set_page_config(page_title="Résumé", page_icon="📄")

# Top section with image beside name and info
col1, col2 = st.columns([1, 4])

with col1:
    st.image("headshot.png", width=140)

with col2:
    st.markdown("""
    # Brea Koenes  
    **Systems Analyst | Data Science Student**

    📍 Seattle, WA  
    📧 breakoenes.com  
    🌐 [LinkedIn](https://www.linkedin.com/in/brea-koenes/)
    """)

st.markdown("---")

# Skills section
st.markdown("### 💻 Technical Skills")
st.markdown("""
- **Languages:** Python, SQL, R, C, C++  
- **Libraries:** Pandas, NumPy, TensorFlow, PyTorch, Scikit-learn, HuggingFace  
- **Tools:** AWS, Azure, Incorta, Oracle Analytics Cloud, OBIEE, Tableau, PowerBI
""")

# Experience section
st.markdown("### 💼 Experience")
st.markdown("""
**Starbucks – Business Intelligence Team**  
*Systems Analyst I* • 2023–Present  
- Redesigned 12 legacy data models and 38 dashboards in a cloud data platform, implementing schema
optimizations with PySpark that reduced average query execution time by 27% 
- Engineered new ETL pipelines for regulatory reporting, implementing data validation frameworks that
achieved zero data integrity incidents for EUDR and SOX compliance.
- Built data integration framework connecting disparate supply chain systems, creating unified forecasting
models and improving forecast accuracy by 11%.
- Delivered 80+ technical training sessions on cloud platform adoption, creating best practices and
organizational data literacy that reduced user error-related tickets by 78.1%.

**Starbucks – Business Intelligence Team**  
*Analyst Intern* • 2022–2023  
- Automated an Excel-based data transformation used weekly by the demand planning team. Streamlined
integration through a data pipeline, reducing the process from 4 hours to 2 minutes.
- Upgraded 5 visualizations from Excel charts to dynamic dashboards, delivering fresher data and improved
analytic capabilities.
""")

# Education section
st.markdown("### 🎓 Education")
st.markdown("""
**M.S. in Data Science** *(Complete August 2025)*  
Easter University 

**B.S. Data Science**  
Calvin University
""")

# Projects section
st.markdown("### 🧪 Notable Projects")
st.markdown("""
- **Machines Doing the Thinking** – Presentation on ethical LLM use in coursework 
- **Product Outage Classifier** – NLP model analyzing Starbucks customer feedback
""")
