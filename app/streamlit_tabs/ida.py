import streamlit as st
import pandas as pd
from app.utils import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from app.utils.helpers import set_ax_fig_style
from app.utils.preprocessing import impute_order_delivery, add_delivery_time, rename_columns, convert_to_datetime



def render_ida_tab():
    """Render the IDA (Initial Data Analysis) tab content"""
    data = preprocessing.load_raw_data()

    data = rename_columns(data)
    data = convert_to_datetime(data)
    data = add_delivery_time(data)
    

    df_customer = data['customer']
    df_order = data['order']
    df_geo = data['geo']
    df_order_item = data['order_item']
    df_product = data['product']
    df_order_payment = data['order_payment']
    df_order_review = data['order_review']
    df_seller = data['seller']
    df_product_category = data['product_category']



    st.title("IDA")


    """Tables and Columns with Null Values"""

    nulls_geo = df_geo.loc[:,(df_geo.isna().sum() > 0).values].columns.tolist()
    nulls_orders = df_order.loc[:,(df_order.isna().sum() > 0).values].columns.tolist()
    nulls_order_item = df_order_item.loc[:,(df_order_item.isna().sum() > 0).values].columns.tolist()
    nulls_order_payment = df_order_payment.loc[:,(df_order_payment.isna().sum() > 0).values].columns.tolist()
    nulls_order_review = df_order_review.loc[:,(df_order_review.isna().sum() > 0).values].columns.tolist()  
    nulls_product = df_product.loc[:,(df_product.isna().sum() > 0).values].columns.tolist()
    nulls_seller = df_seller.loc[:,(df_seller.isna().sum() > 0).values].columns.tolist()
    nulls_product_category = df_product_category.loc[:,(df_product_category.isna().sum() > 0).values].columns.tolist()
    nulls_customer = df_customer.loc[:,(df_customer.isna().sum() > 0).values].columns.tolist()


    null_cols = []

    null_cols.append({"Table": "Geolocation", "Columns": nulls_geo})
    null_cols.append({"Table": "Orders", "Columns": nulls_orders})
    null_cols.append({"Table": "Order Item", "Columns": nulls_order_item})
    null_cols.append({"Table": "Order Payment", "Columns": nulls_order_payment})
    null_cols.append({"Table": "Order Review", "Columns": nulls_order_review})
    null_cols.append({"Table": "Product", "Columns": nulls_product})
    null_cols.append({"Table": "Seller", "Columns": nulls_seller})
    null_cols.append({"Table": "Product Category", "Columns": nulls_product_category})
    null_cols.append({"Table": "Customer", "Columns": nulls_customer})


    st.header("Tables and Columns with Null Values")
    null_cols = [i for i in null_cols if i['Columns']]

    st.table(pd.DataFrame(null_cols))


    """Missing Values Bar Chart"""

    fig, ax = plt.subplots(figsize=(10,3))

    ax, fig = set_ax_fig_style(title="Missing values in Orders After Imputation", 
    xaxis_label="Columns", yaxis_label="Rows", ax=ax, fig=fig, color='white')
    sns.heatmap(df_order.isna().transpose(), cmap='viridis', ax=ax, xticklabels=False)
    st.pyplot(fig)

    """Missing values in Delivery with delivered vs not delivered status"""

    st.header("Missing values in Delivery with delivered vs not delivered status")
    missing_mask = df_order['delivered_customer_date'].isna()
    delivered = df_order['order_status'] == 'delivered'
    missing_delivery_delivered = df_order.loc[missing_mask & delivered].copy()
    missing_delivery_not_delivered = df_order.loc[missing_mask & ~delivered].copy()

    missing_delivery_delivered['delivery_status'] = 'delivered'
    missing_delivery_not_delivered['delivery_status'] = 'not delivered'

    missing_delivery = pd.concat([missing_delivery_delivered, missing_delivery_not_delivered])

    chart = alt.Chart(missing_delivery).mark_bar(width=300).encode(
        x=alt.X('delivery_status:N', title='Delivery Status', sort='-y'),
        y=alt.Y('count(order_id):Q', title='Count')
    ).properties(width=750, height=600, title='Count of Null Delivery Dates by Delivery Status')

    st.altair_chart(chart)
    st.write("Missing delivery values with delivered status:",f' {len(missing_delivery_delivered)}')
    st.write("Missing delivery values without delivered status",f' {len(missing_delivery_not_delivered)}')

    """All order statuses"""

    st.header("Count of Null Delivery Dates by Order Status")
    not_delivered = (df_order.loc[missing_mask & ~ delivered]
                        .groupby('order_status')['order_id']
                        .count()
                        .reset_index()
                        .sort_values(by='order_id', ascending=False))

    fig = alt.Chart(not_delivered).mark_bar().encode(
            x=alt.X('order_status:N', title='Order Status', sort='-y'),
            y=alt.Y('order_id:Q', title='Count')
        ).properties(width=750, height=750, title='Count of Null Delivery Dates by Order Status')

    st.altair_chart(fig)

    """Unavailable order status by timestamp"""

    st.header("Unavailable order status by timestamp")
    # Filter for only unavailable orders
    mask = df_order['order_status'] == 'unavailable'

    # Unavailable histogram
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1, fig = set_ax_fig_style(title="Unavailable order status by timestamp", 
    xaxis_label="Timestamp", yaxis_label="Unavailable Order Status", ax=ax1, fig=fig, color='white')

    sns.histplot(df_order.loc[mask], x=df_order.loc[mask, 'purchase_timestamp'], ax=ax1, label='Unavailable', kde=True)
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Unavailable Order Status')
    ax1.legend()

    # Total orders histogram
    ax2 = ax1.twinx()
    ax2, fig = set_ax_fig_style(title="Unavailable order status by timestamp", 
    xaxis_label="Timestamp", yaxis_label="Unavailable Order Status", ax=ax2, fig=fig, color='white')

    sns.histplot(df_order, x='purchase_timestamp', ax=ax2, kde=True, color="red", label='Total Orders')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    st.pyplot(fig)

    st.write("Total orders spike around 2018-01-01, along with unavailable orders. " \
            "This could be due to normal software or human error. We will proceed with imputation to discover more about cancellation behavior.")

    """Impute missing values with 'unavailable' order status by grouping by seller zip code prefix and getting
    median values for customer and carrier delivery dates."""

    st.header("Imputation")
    st.write("We will impute missing delivery dates by filling it with the median delivery time for all orders.")

    data = {}
    data['order'] = df_order
    data = add_delivery_time(data)
    data = impute_order_delivery(data)
    df_order = data['order']

    fig, ax = plt.subplots(figsize=(10,3))

    ax, fig = set_ax_fig_style(title="Missing values in Orders After Imputation", 
    xaxis_label="Columns", yaxis_label="Rows", ax=ax, fig=fig, color='white')

    sns.heatmap(df_order.isna().transpose(), cmap='viridis', ax=ax, xticklabels=False)
    st.pyplot(fig)