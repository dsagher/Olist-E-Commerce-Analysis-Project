import pandas as pd

def calculate_ARPU(sales_by_region: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the ARPU for each product category and region
    Args:
        sales_by_region: pd.DataFrame
    Returns:
        pd.DataFrame - Columns: Product Category, Region, Sales, Orders, ARPU
    """
    sales_by_region = (sales_by_region
            .assign(ARPU=lambda x: round(x["sales"] / x["order_count"], 2))
            .sort_values(by="ARPU", ascending=False))
    return sales_by_region

def get_total_revenue(data: dict[str, pd.DataFrame]) -> str:
    df_order_item = data['order_item']
    return f"${df_order_item['price'].sum().astype(int):,.0f}"

def get_total_orders(data: dict[str, pd.DataFrame]) -> str:
    df_order = data['order']
    return f"{df_order['order_id'].nunique():,.0f}"

def get_total_customers(data: dict[str, pd.DataFrame]) -> str:
    df_customer = data['customer']
    return f"{df_customer['customer_id'].nunique():,.0f}"