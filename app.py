import streamlit as st
from app.streamlit_tabs import ida, eda, modeling

st.set_page_config(page_title="Olist E-Commerce Data Analysis Dashboard", page_icon=":bar_chart:", layout="wide")



ida_tab, eda_tab, modeling_tab = st.tabs(["IDA", "EDA", "Modeling"])

with ida_tab:
    ida.render_ida_tab()
with eda_tab:
    eda.render_eda_tab()
with modeling_tab:
    modeling.render_modeling_tab()

