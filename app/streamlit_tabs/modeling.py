import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from app.utils.preprocessing import load_processed_data
from app.utils.helpers import set_ax_fig_style
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.dummy import DummyClassifier
import numpy as np

data = load_processed_data()

df_order_item = data['order_item']
df_order = data['order']
df_product = data['product']
df_customer = data['customer']
df_geo = data['geo']
df_seller = data['seller']
df_order_payment = data['order_payment']
df_order_review = data['order_review']
df_product_category = data['product_category']

@st.fragment()
def run_roc_curve(y_test: np.ndarray, y_scores: np.ndarray):
    """Helper function to run ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_dim = list(zip(fpr,tpr, thresholds))
    return roc_dim

@st.fragment()
def run_precision_recall_curve(y_test: np.ndarray, y_scores: np.ndarray):
    """Helper function to run precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    prs_dim = list(zip(precision, recall, thresholds))
    return prs_dim


@st.fragment()
def run_log_reg(df: pd.DataFrame, X: str, y: str):
    """Helper function to loop logistic regression and output results"""

    df = df[[X, y]].dropna() 
    feature = df[[X]].to_numpy()             
    target = df[y].to_numpy()                

    logreg = LogisticRegression()
    logreg.fit(feature, target)

    # Extract coefficients and exponentiate to get non-log odds
    coef = logreg.coef_
    odds_ratio = float(np.exp(coef)[0, 0])

    return logreg.coef_, odds_ratio

def render_modeling_tab():

    st.title("Modeling")
    """=================================================Feature Engineering================================================="""

    """============Define feature and target variables============"""

    # Product dimension columns for analysis - continuous
    prod_dim_cols = ['weight', 'length','height','width']

    # Category column for analysis - multiple binary
    category_col = ['category_name']

    # target variable and order_id for merging order_items and orders
    orders_cols = ['delayed', 'order_id']
    target = 'delayed'

    df_order_item_orders = df_order_item.merge(df_order[orders_cols])

    """============One-hot encode category_name column============"""

    ohe = OneHotEncoder(sparse_output=False)

    encoded = ohe.fit_transform(df_order_item_orders['category_name'].to_numpy().reshape(-1,1))

    category_cols = ohe.get_feature_names_out()

    category_cols = [col.strip('x0_') for col in category_cols]

    """============Create feature df's============"""

    category_df = pd.DataFrame(encoded, columns=category_cols)
    category_df = pd.concat((category_df, df_order_item_orders[target]), axis=1)

    prod_dim_df = pd.DataFrame(df_order_item_orders[prod_dim_cols])
    prod_dim_df = pd.concat((prod_dim_df, df_order_item_orders[target]), axis=1).dropna(axis=0)

    """============Run F-regression on categories============"""

    X_cat = category_df.drop(columns='delayed')
    y_cat = category_df[target]


    f, p = f_regression(X_cat, y_cat)

    # Collect f-statistics, p-values, and column names
    zipped = list(zip(f,p,category_cols))

    # Create DataFrame
    category_f_df = pd.DataFrame(zipped, columns=['F-statistic', 'p-value','feature'])

    # Filter for statistically significant p-values
    mask = category_f_df['p-value'] < .05
    category_below_5 = category_f_df[mask]

    # Sort F-statistics
    top_10_cats = category_below_5.sort_values('F-statistic', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 2))

    ax, fig = set_ax_fig_style("Top 10 Categories by F-Statistic", "feature", "F-statistic", ax, fig)
    sns.barplot(top_10_cats, x='feature', y='F-statistic', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)


    """============Run F-regression on dimensions============"""


    X_dim = prod_dim_df[prod_dim_cols]
    y_dim = prod_dim_df[target]

    f, p = f_regression(X_dim, y_dim)

    zipped = list(zip(f,p, prod_dim_cols))

    # Create DataFrame
    dim_f_df = pd.DataFrame(zipped, columns=['F-statistic', 'p-value','feature'])

    # Filter for statistically significant p-values
    mask = dim_f_df['p-value'] < .05
    top_10_dims = dim_f_df[mask].sort_values('F-statistic', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 2))
    ax, fig = set_ax_fig_style("Product Dimensions by F-Statistic", "feature", "F-statistic", ax, fig)
    sns.barplot(top_10_dims, x='feature', y='F-statistic', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    """=================================================Logistic Regression================================================="""
    st.divider()

    st.title("Logistic Regression Results")

    """============Run logistic regression on top 10 categories and output============"""

    st.header("Logistic Regression Results for Top 10 Categories")
    cat_logreg_results = []
    for feature in top_10_cats['feature']:
        coef, odds_ratio = run_log_reg(category_df, feature, target)
        cat_logreg_results.append((feature, coef[0,0], odds_ratio))
    cat_logreg_results_df = pd.DataFrame(cat_logreg_results, columns=['feature', 'beta', 'odds_ratio']).sort_values(by='odds_ratio', ascending=False)
    st.write(cat_logreg_results_df)

    st.markdown("**Interpretation:**")

    for idx, feature, beta, odds_ratio in cat_logreg_results_df.itertuples():
        direction = 'higher' if odds_ratio > 1 else 'lower'
        percentage = 100 * (odds_ratio - 1)
        st.write(f"- {feature} orders have about {odds_ratio:.2f}x {direction} odds ({percentage:.1f}%) of being delayed than non-{feature} orders.")

    """============Run logistic regression on dimensions and output============"""

    st.header("Logistic Regression on Dimensions")
    prod_logreg_results = []
    for feature in top_10_dims['feature']:
        coef, odds_ratio = run_log_reg(prod_dim_df, feature, target)
        prod_logreg_results.append((feature, coef[0,0], odds_ratio))
    prod_logreg_results_df = pd.DataFrame(prod_logreg_results, columns=['feature', 'beta', 'odds_ratio']).sort_values(by='odds_ratio', ascending=False)
    st.write(prod_logreg_results_df)

    st.markdown("**Interpretation:**")
    for idx, feature, beta, odds_ratio in prod_logreg_results_df.itertuples():
        direction = 'increase' if odds_ratio > 1 else 'decrease'
        percent_change = 100 * (odds_ratio - 1)
        scaled_or = odds_ratio**20
        scaled_percent_change = 100 * (scaled_or - 1)
        st.write(f"- Per cm increase in {feature}, the odds of delay {direction} by {percent_change:.1f}%.")
        st.write(f"Scaled Example: An increase of 20 cm in {feature} would {direction} the odds of delay by {scaled_percent_change:.3f}%.")

    """============Test Baseline Classification Accuracy============"""

    st.header("Baseline Classification Accuracy")
    X_all = pd.concat((X_dim, X_cat), axis=1)
    y = y_cat

    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all,y, train_size=0.8, random_state=0)

    dummy = DummyClassifier(strategy="most_frequent", random_state=0)
    dummy.fit(X_all, y)
    y_dummy = dummy.predict(X_test_all)
    
    st.write(f"Baseline accuracy: {accuracy_score(y_test_all, y_dummy):.2f}")
    fig, ax = plt.subplots(figsize=(5, 3))
    ConfusionMatrixDisplay(confusion_matrix(y_test_all, y_dummy)).plot(ax=ax)
    st.pyplot(fig, use_container_width=False)

    """============Test Classification Strength of Product Dimension============"""

    st.header("Test Classification Strength of Product Dimensions")

    # Split and fit logistic regression model
    X_dim_train, X_dim_test, y_dim_train, y_dim_test = train_test_split(X_dim, y_dim, random_state=42)

    logreg_dim = LogisticRegression()
    logreg_dim.fit(X_dim_train, y_dim_train)

    y_pred = logreg_dim.predict(X_dim_test)
    st.write(f"Classification Accuracy: {accuracy_score(y_dim_test, y_pred):.2f}")

    col1, col2, col3 = st.columns(3)
    y_scores = logreg_dim.predict_proba(X_dim_test)[:,1]

    # Run ROC and precision-recall curves
    roc_dim = run_roc_curve(y_dim_test, y_scores)
    prs_dim = run_precision_recall_curve(y_dim_test, y_scores)

    # Plot confusion matrix, precision-recall curve, and ROC curve
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix(y_dim_test, y_pred)).plot(ax=ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(x='recall', y='precision', data = pd.DataFrame(prs_dim, columns=['precision', 'recall', 'threshold']), ax=ax)
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(x='false positive rate', y='true positive rate', data = pd.DataFrame(roc_dim, columns=['false positive rate', 'true positive rate', 'threshold']), ax=ax)
        st.pyplot(fig)

    st.markdown("**Interpretation:**")
    st.write("The confusion matrix shows the number of true positives, false positives, true negatives, and false negatives.")
    st.write("The precision-recall curve shows the precision and recall at different thresholds.")
    st.write("The ROC curve shows the true positive rate and false positive rate at different thresholds.")

    """============Test Classification Strength of Product Categories============"""

    st.header("Test Classification Strength of Product Categories")

    # Split and fit logistic regression model
    X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(X_cat, y_cat, random_state=42)

    logreg_cat = LogisticRegression()
    logreg_cat.fit(X_cat_train, y_cat_train)

    y_pred = logreg_cat.predict(X_cat_test)
    st.write(f"Classification Accuracy: {accuracy_score(y_cat_test, y_pred):.2f}")

    col1, col2, col3 = st.columns(3)
    y_scores = logreg_cat.predict_proba(X_cat_test)[:,1]

    # Run ROC and precision-recall curves
    roc_cat = run_roc_curve(y_cat_test, y_scores)
    prs_cat = run_precision_recall_curve(y_cat_test, y_scores)

    # Plot confusion matrix, precision-recall curve, and ROC curve
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix(y_cat_test, y_pred)).plot(ax=ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(x='recall', y='precision', data = pd.DataFrame(prs_cat, columns=['precision', 'recall', 'threshold']), ax=ax)
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(x='false positive rate', y='true positive rate', data = pd.DataFrame(roc_cat, columns=['false positive rate', 'true positive rate', 'threshold']), ax=ax)
        st.pyplot(fig)

    st.markdown("**Interpretation:**")
    st.write("The confusion matrix shows the number of true positives, false positives, true negatives, and false negatives.")
    st.write("The precision-recall curve shows the precision and recall at different thresholds.")
    st.write("The ROC curve shows the true positive rate and false positive rate at different thresholds.")
