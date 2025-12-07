import streamlit as st
import pandas as pd
import altair as alt
from app.utils.preprocessing import load_processed_data
from app.utils.merges import get_highest_selling_cities, get_highest_selling_categories
from app.utils.helpers import set_ax_fig_style
from app.utils.aggregations import get_total_revenue, get_total_orders, get_total_customers, calculate_ARPU
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress


def render_eda_tab():
    """Render the EDA (Exploratory Data Analysis) tab content"""
    st.title("EDA")

    data = load_processed_data()

    df_customer = data['customer']
    df_order = data['order']
    df_geo = data['geo']
    df_order_item = data['order_item']
    df_product = data['product']
    df_order_payment = data['order_payment']
    df_order_review = data['order_review']
    df_seller = data['seller']
    df_product_category = data['product_category']

    selected_year = st.selectbox("Select Year", [2016, 2017, 2018, 2019])

    total_revenue = get_total_revenue(data)
    total_orders = get_total_orders(data, selected_year)
    total_customers = get_total_customers(data, selected_year)
    highest_selling_city = get_highest_selling_cities(data, selected_year)
    highest_selling_category = get_highest_selling_categories(data, selected_year)


    st.title("Sales")


    # KPI Metrics
    with st.container():
        st.markdown("## Sales KPIs")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            with st.container(border=True):
                st.markdown(f"## **Total Revenue:** \n ### `{total_revenue}`")
        with col2:
            with st.container(border=True):
                st.markdown(f"## **Total Orders:** \n ### `{total_orders}`")
        with col3:
            with st.container(border=True):
                st.markdown(f"## **Total Customers:** \n ### `{total_customers}`")
        with col4:
            with st.container(border=True):
                st.markdown(f"## **Highest Selling City:** \n ### `{highest_selling_city}`")
        with col5:
            with st.container(border=True):
                st.markdown(f"## **Highest Selling Category:** \n ### `{highest_selling_category}`")



    st.header("Sales by Region and Product Category")
    # Get unique zips with city, state, lat, lng, and region
    geo_cols = ['zip_code_prefix','region', 'city', 'state', 'latitude', 'longitude']
    unique_zips = (df_geo[geo_cols]
                    .groupby('zip_code_prefix')
                    .first())

    # Merge order and customer data to get zips
    customer_order = df_order.merge(df_customer[['customer_id','zip_code_prefix']], on='customer_id', how='inner')

    # Merge zips and geo location with customer order data
    customer_order_geo = unique_zips.merge(customer_order, on='zip_code_prefix', how='inner')
    
    customer_order_geo_product = customer_order_geo.merge(df_order_item, on='order_id', how='inner')

    # Calculate sales by region and product category
    sales_by_region = (customer_order_geo_product
            .groupby(["category_name", "region"])
            .agg({"price": "sum", "order_id": "count"})
            .reset_index()
            .rename(columns={"price": "sales", "order_id": "order_count"})
            )
    
    sales_by_region = calculate_ARPU(sales_by_region)

    bubble_chart = alt.Chart(sales_by_region).mark_circle(opacity=0.7).encode(
        x=alt.X('sales:Q', title='Total Sales (BRL)').scale(type='log'),
        y=alt.Y('ARPU:Q', title='Average Revenue per Order (ARPU)').scale(type='log'),
        size=alt.Size('order_count:Q', title='Order Count', scale=alt.Scale(range=[30, 1000])),
        color=alt.Color('region:N', title='Region'),
        tooltip=['category_name', 'region', 'sales', 'ARPU', 'order_count']
    ).properties(
        title='Sales vs ARPU by Product Category and Region',
        width=800,
        height=500
    ).interactive()
    rule = alt.Chart(sales_by_region).mark_rule(color='red', strokeWidth=2).encode(
        y=alt.Y('mean(ARPU):Q', title='Average Revenue per Order (ARPU)')
    )
    rule2 = alt.Chart(sales_by_region).mark_rule(color='blue', strokeWidth=2).encode(
        x=alt.X('mean(sales):Q', title='Total Sales (BRL)')
    )

    st.altair_chart(bubble_chart + rule + rule2)

    st.markdown("This chart allows us to visualize product categories that are performing well among different metrics and regions.")
    st.header("Delivery Time by Review Score")

    order_cols = ['order_id', 'delivery_time']
    review_cols = ['order_id', 'review_score']

    source = df_order[order_cols].merge(df_order_review[review_cols], on='order_id', how='left').dropna(axis=0)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax, fig = set_ax_fig_style(title='', xaxis_label='Review Score', yaxis_label='Delivery Time', ax=ax, fig=fig, color='white')

    # Barplot for average delivery time per review score
    sns.regplot(data=source, x="delivery_time", y="review_score")

    res = linregress(source["delivery_time"], source["review_score"])
    
    ax.legend()
    direction = "negatively" if res.rvalue < 0 else "positively" #type: ignore
    r_squared = res.rvalue**2 * 100 #type: ignore


    plt.ylim(0, 10)
    plt.xlim(0, 50)
    plt.text(5, 7, f"p-value: {res.pvalue:.2f}, r: {res.rvalue:.2f}, R^2: {r_squared:.2f}%", color='white') #type: ignore
    st.pyplot(fig)

    st.markdown(f"**Interpretation:** The average delivery time is **{direction} correlated** with the review score " \
        f"(**p < 0.001**), but only **{r_squared:.2f}%** of the variance can be explained. " \
        "This suggests that there is a **weak** relationship between delivery time and review score.")