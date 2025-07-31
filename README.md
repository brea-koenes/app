# Product Outage Classifier & Portfolio

This Streamlit app showcases my résumé, technical projects, and data science portfolio.

## Features

- **Résumé:** View my professional experience, education, and skills.
- **Projects:** Explore my data science and machine learning projects, including links to code and presentations.
- **Home:** Learn more about my background and career goals.
- **Presentation Video:** Watch my capstone project presentation directly in the app.

## File Structure

```
/workspaces/app/
├── main.py                # Main entry point for the Streamlit app
├── headshot.png           # Profile photo for résumé page
├── presentation.mp4       # Project presentation video
├── model.pkl              # (Optional) ML model file
├── pages/
│   ├── home.py            # Home/biography page
│   ├── projects.py        # Projects and portfolio page
│   └── résumé.py          # Résumé page
```

## How to Run Locally

1. **Install dependencies** (if needed):
    ```bash
    pip3 install streamlit
    ```

2. **Start the app**:
    ```bash
    streamlit run main.py
    ```

3. **Navigate** using the sidebar to view all pages.

## Deployment

To deploy on [Streamlit Community Cloud](https://share.streamlit.io/):

1. Push your code to GitHub.
2. Create a new app on Streamlit Community Cloud, pointing to `main.py`.
3. All pages in the `pages/` folder will appear as sidebar tabs automatically.

## Contact

- [LinkedIn](https://www.linkedin.com/in/brea-koenes/)
- [breakoenes.com](mailto:breakoenes.com)

