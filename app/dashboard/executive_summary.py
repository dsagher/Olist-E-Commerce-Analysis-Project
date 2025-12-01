import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.preprocessing import load_processed_data_streamlit
from app.utils import aggregations, merges

# Load data for summary metrics
data = load_processed_data_streamlit()
total_revenue = aggregations.get_total_revenue(data)
total_orders = aggregations.get_total_orders(data)
total_customers = aggregations.get_total_customers(data)
highest_selling_city = merges.get_highest_selling_cities(data).head(1).index[0].title()
highest_selling_category = merges.get_highest_selling_categories(data).head(1).index[0].title().replace("_", " & ")

st.title("📊 Executive Summary")
st.markdown("### Olist Brazilian E-Commerce Analysis (2016-2018)")

st.markdown("---")

# Overview Section
st.markdown("## 📖 Project Overview")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    This analysis examines **100,000+ orders** from Olist, Brazil's largest e-commerce platform, 
    spanning multiple marketplaces between 2016-2018. The dataset includes customer demographics, 
    order details, product information, payments, reviews, and geolocation data across Brazil's 
    five geographic regions.
    
    **Objectives:**
    - Identify revenue drivers and high-value customer segments
    - Analyze regional sales patterns and market opportunities
    - Examine the relationship between delivery performance and customer satisfaction
    - Provide data-driven recommendations for business growth
    """)

with col2:
    st.metric("Total Revenue", total_revenue)
    st.metric("Total Orders", total_orders)
    st.metric("Unique Customers", total_customers)

st.markdown("---")

# Key Findings
st.markdown("## 🔍 Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### **1. Regional Market Concentration**
    - **Southeast region dominates** with the majority of revenue and orders
    - **{}** emerges as the highest-selling city
    - Significant growth opportunities exist in underserved North and Northeast regions
    
    ### **2. Product Category Performance**
    - **{}** leads in total sales volume
    - High-value categories show distinct seasonal patterns
    - Cross-regional preferences reveal targeted marketing opportunities
    - Product category mix varies significantly by region
    """.format(highest_selling_city, highest_selling_category))

with col2:
    st.markdown("""
    ### **3. Customer Experience & Delivery**
    - Slight negative correlation between delivery time and review scores
    - Orders with longer delivery times (>10 days) receive lower ratings
    """)

st.markdown("---")

# Business Insights
st.markdown("## 💡 Strategic Insights")

with st.expander("🎯 Market Expansion Opportunities", expanded=True):
    st.markdown("""
    **Regional Growth Strategy:**
    - **North and Northeast regions** represent untapped markets with lower penetration
    - Regional product mix optimization could increase conversion rates
    - Localized marketing campaigns should target underserved areas
    
    **ARPU Analysis:**
    - Significant variation in Average Revenue Per User across regions
    - High-sales/low-ARPU segments indicate opportunity for premium product positioning
    - Category-specific ARPU trends suggest potential for upselling strategies
    """)

with st.expander("🚚 Logistics & Customer Satisfaction"):
    st.markdown("""
    **Delivery Time Impact:**
    - Strong correlation: bigger and more expensive orders take longer to deliver
    - Orders delivered within 10 days show slightly better ratings
    - Delivery time is a key driver of customer satisfaction and repeat purchases
    
    **Recommendations:**
    - Implement delivery time guarantees for high-value orders
    - Implement a delivery time SLA for high-value orders
    """)

with st.expander("📈 Revenue Optimization"):
    st.markdown("""
    **Product Strategy:**
    - Focus inventory investment on high-ARPU categories
    - Seasonal patterns suggest Q4 inventory buildup opportunities (holiday shopping)
    
    **Pricing & Promotions:**
    - Regional pricing strategies based on local ARPU
    """)

st.markdown("---")

# Recommendations
st.markdown("## 🎯 Business Recommendations")

st.markdown("""
### **Short-Term Actions (0-3 months)**
1. **Optimize delivery performance** in low-rating regions
2. **Launch regional marketing campaigns** targeting North/Northeast
3. **Implement A/B testing** for pricing strategies in different regions
4. **Enhance seller network** in underserved areas

### **Medium-Term Initiatives (3-6 months)**
1. **Establish regional distribution centers** to improve delivery times
2. **Develop category-specific strategies** for high-ARPU products
3. **Create customer loyalty programs** focusing on repeat purchases
4. **Analyze and reduce** delivery time variability

### **Long-Term Strategy (6+ months)**
1. **Expand seller partnerships** in emerging regions
2. **Build predictive models** for demand forecasting by region/category
3. **Implement dynamic pricing** based on regional market conditions
4. **Develop category expansion strategy** for underrepresented products
""")

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><i>This analysis is part of the CMSE 830 course project (Fall 2025)</i></p>
    <p><b>Navigate to the Main Dashboard to explore interactive visualizations</b></p>
</div>
""", unsafe_allow_html=True)