import pandas as pd
import altair as alt
import logging
logging.basicConfig(level=logging.INFO)


def get_sales_by_region_category_bubble_chart(df: pd.DataFrame) -> alt.LayerChart:
    """
    Get the bubble chart for ARPU and total sales by Region and Product Category
    Args:
        df: pd.DataFrame - Requires sales, order_count, and ARPU columns, region, and category_name columns
    Returns:
        alt.Chart - Sales vs ARPU by Product Category and Region
    """

    bubble_chart = alt.Chart(df).mark_circle(opacity=0.7).encode(
        x=alt.X('sales:Q', title='Total Sales (BRL)'),
        y=alt.Y('ARPU:Q', title='Average Revenue per Order (ARPU)'),
        size=alt.Size('order_count:Q', title='Order Count', scale=alt.Scale(range=[30, 1000])),
        color=alt.Color('region:N', title='Region'),
        tooltip=['category_name', 'region', 'sales', 'ARPU', 'order_count']
    ).properties(
        title='Sales vs ARPU by Product Category and Region',
        width=800,
        height=500,
    ).interactive()
    rule = alt.Chart(df).mark_rule(color='red', strokeWidth=2).encode(
        y=alt.Y('mean(ARPU):Q', title='Average Revenue per Order (ARPU)')
    )
    rule2 = alt.Chart(df).mark_rule(color='blue', strokeWidth=2).encode(
        x=alt.X('mean(sales):Q', title='Total Sales (BRL)')
    )

    return bubble_chart + rule + rule2

def sales_ARPU_time_chart(df: pd.DataFrame, year: int | list[int] = 2017, title: str = "") -> alt.Chart:
    """
    Above Average Sales and Below Average ARPU
    Args: 
        sales_by_region: pd.DataFrame
            Columns: Product Category, Region, Sales, Orders, ARPU
        df_order: pd.DataFrame
        df_order_item: pd.DataFrame
        df_product: pd.DataFrame
    Returns: 
        alt.Chart - Above Average Sales and Below Average ARPU
    """

    VALID_YEARS = [2016, 2017, 2018, 2019]
    
    if isinstance(year, int) and year not in VALID_YEARS:
        raise ValueError(f"Year {year} not in {VALID_YEARS}")
    if isinstance(year, list):
        if not all(y in VALID_YEARS for y in year):
            raise ValueError(f"All years must be in {VALID_YEARS}")

    df_year = df[df['purchase_month'].dt.year.isin([year])]

    # Group by month and category
    df_year_agg = df_year.groupby(
        ['purchase_month', 'category_name']
            ).size().reset_index(name='order_count')

    chart = alt.Chart(df_year_agg).mark_line(point=True).encode(
        x=alt.X('purchase_month:T', title='Month'),
        y=alt.Y('order_count:Q', title='Number of Orders'),
        color=alt.Color('category_name:N', title='Product Category'),
        tooltip=['purchase_month:T', 'category_name:N', 'order_count:Q']
    ).properties(
        title=title,
        width=700,
        height=400
    ).interactive()

    return chart


def payment_type_pie_chart(df: pd.DataFrame) -> alt.Chart:
    """
    Get the chart for payment type
    Args:
        df: pd.DataFrame - Requires payment_type and payment_value columns
    Returns:
        alt.Chart - Payment type and payment value
    """
    chart = alt.Chart(df).mark_arc().encode(
        theta="sum(payment_value):Q",
        color="payment_type:N"
    )
    return chart

def delivery_time_boxplot_chart(df_order: pd.DataFrame, df_order_review: pd.DataFrame) -> alt.Chart:
    """
    Get the chart for delivery time
    Args:
        df: pd.DataFrame - Requires delivery_time and review_score columns
    Returns:
        alt.Chart - Delivery time and review score
    """
    order_cols = ['order_id', 'delivery_time']
    review_cols = ['order_id', 'review_score']
    source = df_order[order_cols].merge(df_order_review[review_cols], on='order_id', how='left')
    chart = alt.Chart(source).mark_boxplot().encode(
        y=alt.Y('delivery_time:Q', title='Delivery Time', scale=alt.Scale(domain=[1, 50])),
        x=alt.X('review_score:O', title='Review Score', sort=[1,2,3,4,5])
    ).properties(
    title='Delivery Time by Review Score'
    )
    return chart