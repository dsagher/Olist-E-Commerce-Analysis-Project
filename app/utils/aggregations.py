import pandas as pd
from app.utils.helpers import raise_for_invalid_year
def calculate_ARPU(sales_by_region: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the ARPU for each product category and region
    Args:
        sales_by_region: pd.DataFrame
        year: int | list[int] - The year to calculate the ARPU for
    Returns:
        pd.DataFrame - Columns: Product Category, Region, Sales, Orders, ARPU
    """
    sales_by_region = (sales_by_region
            .assign(ARPU=lambda x: round(x["sales"] / x["order_count"], 2))
            .sort_values(by="ARPU", ascending=False))
    return sales_by_region

def get_total_revenue(data: dict[str, pd.DataFrame], year: int = 2017) -> str:
    raise_for_invalid_year(year)
    df_order_item = data['order_item']
    df_order = data['order']
    merged = df_order_item.merge(df_order[['order_id','purchase_year']], on='order_id', how='inner')
    df_year = merged[merged['purchase_year'] == year]
    return f"${df_year['price'].sum().astype(int):,.0f}"

def get_total_orders(data: dict[str, pd.DataFrame], year: int = 2017) -> str:
    raise_for_invalid_year(year)
    df_order = data['order']
    df_year = df_order[df_order['purchase_year'] == year]
    return f"{df_year['order_id'].nunique():,.0f}"

def get_total_customers(data: dict[str, pd.DataFrame], year: int = 2017) -> str:
    raise_for_invalid_year(year)
    df_order = data['order']
    df_customer = data['customer']
    merged = df_customer.merge(df_order[['customer_id','purchase_year']], on='customer_id', how='inner')
    df_year = merged[merged['purchase_year'] == year]
    return f"{df_year['customer_id'].nunique():,.0f}"