import pandas as pd

def get_sales_by_region_category(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge the customer, orders, geo, order item, to get sales by region
    Args:
        data: dict[str, pd.DataFrame]
            Data:
                - customer: pd.DataFrame
                - order: pd.DataFrame
                - geo: pd.DataFrame
                - order_item: pd.DataFrame
    Returns:
        pd.DataFrame - Columns: category_name, region, sales, order_count
    
    Notes:
        - The merging could be done during preprocessing to avoid redundant merging
    """
    
    df_customer = data['customer']
    df_orders = data['order']
    df_geo = data['geo']
    df_order_item = data['order_item']


    # Get unique zips with city, state, lat, lng, and region
    geo_cols = ['zip_code_prefix','region', 'city', 'state', 'latitude', 'longitude']
    unique_zips = (df_geo[geo_cols]
                    .groupby('zip_code_prefix')
                    .first())

    # Merge order and customer data to get zips
    customer_order = df_orders.merge(df_customer[['customer_id','zip_code_prefix']], on='customer_id', how='inner')

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
    
    return sales_by_region

def get_average_sales_ARPU(sales_by_region: pd.DataFrame, 
                                                    data: dict[str, pd.DataFrame],
                                                    sales: bool = True,
                                                    ARPU: bool = False,
                                                    top_n: int = 10) -> pd.DataFrame:
    """
    Get the above average sales and below average ARPU
    Args:
        sales_by_region: pd.DataFrame
        data: dict[str, pd.DataFrame]
            Data:
                - order: pd.DataFrame
                - order_item: pd.DataFrame
                - product: pd.DataFrame
        sales: bool -> True if above average sales, False if below average sales
        ARPU: bool -> True if below average ARPU, False if above average ARPU
    Returns:
        pd.DataFrame - Columns: arpu, region, sales, product_category, order_purchase_month
    
    Notes:
        - The merging may be redundant and could be done in calculating sales_by_region
    """
    
    df_order = data['order']
    df_order_item = data['order_item']
    df_product = data['product']
    # Means
    avg_ARPU = sales_by_region["ARPU"].mean()
    avg_sales = sales_by_region['sales'].mean()

    if sales:
        sales_mask = (sales_by_region['sales'] > avg_sales)
    else:
        sales_mask = (sales_by_region['sales'] < avg_sales)
    if ARPU:
        ARPU_mask = (sales_by_region['ARPU'] < avg_ARPU)
    else:
        ARPU_mask = (sales_by_region['ARPU'] > avg_ARPU)

    mask = sales_mask & ARPU_mask

    # Top 10
    above_avg = sales_by_region.loc[mask].sort_values(by=['sales']).head(top_n)

    # Merge order, order_item, product, and above average to grab top sales
    merge = (df_order
            .merge(df_order_item)
            .merge(df_product)
            .merge(above_avg, left_on="category_name", right_on="category_name")
            [['ARPU','region','sales', "category_name", "purchase_month"]])

    merge['purchase_month'] = pd.to_datetime(merge['purchase_month'])
    return merge

def get_highest_selling_cities(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Get the highest selling cities
    Args:
        data: dict[str, pd.DataFrame]
            Data:
                - order: pd.DataFrame
                - customer: pd.DataFrame
    Returns:
        str - The highest selling cities
    """
    df_order = data['order']
    df_customer = data['customer']

    order_cols = ['order_id', 'customer_id']
    customer_cols = ['customer_id', 'city']

    highest_selling_cities = (df_customer[customer_cols]
                    .merge(df_order[order_cols], on='customer_id', how='inner')
                    .groupby('city').agg({'order_id': 'count'})
                    .sort_values(by='order_id', ascending=False))
    
    return highest_selling_cities

def get_highest_selling_categories(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Get the highest selling categories
    Args:
        data: dict[str, pd.DataFrame]
            Data:
                - order_item: pd.DataFrame - product: pd.DataFrame
    Returns:
        str - The highest selling categories
    """
    df_order_item = data['order_item']
    df_product = data['product']

    highest_selling_categories = (df_order_item
        .merge(df_product['product_id'], on='product_id', how='inner')
        .groupby('category_name')
        .agg({'price': 'sum'})
        .sort_values(by='price', ascending=False))
    return highest_selling_categories