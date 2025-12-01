# Olist E-Commerce Data Analysis Dashboard
*Project for CMSE 830 - Fall 2025*

An interactive Streamlit dashboard for exploring and analyzing Brazilian e-commerce data from Olist. This project provides comprehensive exploratory data analysis (EDA) with visualizations for sales patterns, customer behavior, and regional insights.

### Application Files

- **`app/webapp.py`**: Main Streamlit dashboard that displays KPIs (total revenue, orders, customers, top-selling city/category) and interactive visualizations including bubble charts and time-series analysis of sales and ARPU (Average Revenue Per User).

- **`app/utils/preprocessing.py`**: Handles all data preprocessing including:
  - Loading raw and processed data
  - Column renaming and standardization
  - DateTime conversions
  - Feature engineering (delivery time, date features, product volume)
  - Region mapping for Brazilian states
  - Customer spending categorization

- **`app/utils/merges.py`**: Provides data merging functions:
  - `get_sales_by_region_category()`: Merges customer, order, and product data by region
  - `get_average_sales_ARPU()`: Filters data by sales and ARPU thresholds
  - `get_highest_selling_cities()`: Identifies top-performing cities
  - `get_highest_selling_categories()`: Identifies best-selling product categories

- **`app/utils/aggregations.py`**: Calculates key metrics:
  - ARPU (Average Revenue Per User) calculation
  - Total revenue, orders, and customer counts
  - Formatted string outputs for dashboard KPIs

- **`app/utils/charts.py`**: Generates Altair visualizations:
  - Bubble charts for sales vs ARPU by region and category
  - Time-series line charts for order trends
  - Interactive tooltips and filtering

### Data Files

- **`data/raw/`**: Contains original Olist datasets downloaded from the source, including customer, order, product, seller, payment, review, and geolocation data.

- **`data/processed/`**: Contains cleaned and feature-engineered datasets ready for analysis and visualization.

### Notebooks

- **`notebooks/cleaning.ipynb`**: Documents the data cleaning process and quality checks
- **`notebooks/correlation_analysis.ipynb`**: Statistical analysis and correlation studies
- **`notebooks/EDA.ipynb`**: Exploratory data analysis and initial insights
- **`notebooks/feature_engineering.ipynb`**: Feature creation and transformation experiments

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run app/webapp.py
```

### Processing Raw Data

To regenerate processed datasets from raw data:

```bash
python app/utils/preprocessing.py
```


## Features

- **KPI Dashboard**: View key business metrics at a glance
- **Regional Analysis**: Explore sales patterns across Brazilian regions
- **ARPU Insights**: Identify high-value and low-value customer segments
- **Time-Series Analysis**: Track order trends over time
- **Interactive Visualizations**: Filter and explore data dynamically

## Technologies Used

- **Streamlit**: Interactive web dashboard
- **Pandas**: Data manipulation and analysis
- **Altair**: Declarative statistical visualizations
- **Scikit-learn**: Feature engineering (KBinsDiscretizer)
- **SciPy**: Statistical computations

---