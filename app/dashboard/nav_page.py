import streamlit as st

pages = {
    "Executive Summary": [
        st.Page("executive_summary.py", title="Executive Summary"),
    ],
    "Main Dashboard": [
        st.Page("main_dashboard.py", title="Main Dashboard"),
    ],
}

pg = st.navigation(pages)
pg.run()