import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from prophet import Prophet
from prophet.plot import plot_plotly

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Market Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="metric-container"] {
        background-color: #1f1f1f; border: 1px solid #333; padding: 1rem;
        border-radius: 0.5rem; color: white;
    }
    .stButton>button { border-radius: 0.5rem; border: 1px solid #f63366; }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🤖 AI Market Analysis & Forecasting Dashboard")
st.markdown("An advanced tool for analyzing market trends and predicting future outcomes.")

# --- Data Loading and Caching ---
@st.cache_data
def load_data(filepath):
    """Loads and preprocesses data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        for col in ['Launch_Date', 'TransactionDate']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df
    except FileNotFoundError:
        return None

def initialize_session_state():
    """Initializes session state for products and sales data."""
    if 'products_df' not in st.session_state:
        loaded_products = load_data('products.csv')
        st.session_state.products_df = loaded_products if loaded_products is not None else pd.DataFrame()

    if 'sales_df' not in st.session_state:
        loaded_sales = load_data('sales_transactions.csv')
        st.session_state.sales_df = loaded_sales if loaded_sales is not None else pd.DataFrame()

# --- AI Model Functions ---
def run_prophet_forecast(df, date_col, value_col, periods=365):
    """Generic Prophet forecasting function."""
    if df.empty or date_col not in df.columns or value_col not in df.columns or len(df) < 2:
        return None, None
    prophet_df = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# --- Initialize Data ---
initialize_session_state()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    with st.expander("📝 Add New Data"):
        # Simplified forms for demonstration
        with st.form("new_product_form", clear_on_submit=True):
            st.subheader("Add Product")
            st.text_input("Product Name", key="p_name"); st.number_input("Price (USD)", min_value=0.0, key="p_price")
            st.date_input("Launch Date", key="p_date")
            if st.form_submit_button("Add Product"): st.success("Product added (demo).")
        with st.form("new_transaction_form", clear_on_submit=True):
            st.subheader("Add Transaction")
            st.text_input("Customer ID", key="t_customer"); st.selectbox("Product", options=["P-1", "P-2"], key="t_product")
            st.date_input("Transaction Date", key="t_date")
            if st.form_submit_button("Add Transaction"): st.success("Transaction added (demo).")

# --- Main Dashboard ---
tab1, tab2 = st.tabs(["🚀 Product Analysis & Forecasting", "🛒 Customer Behavior & Forecasting"])

with tab1:
    st.header("Product Performance Dashboard")
    df_products = st.session_state.products_df
    
    prod_tab1, prod_tab2, prod_tab3, prod_tab4 = st.tabs(["📊 Overview", "📈 Launch Analysis", "💰 Pricing Analysis", "🔮 Forecasting"])

    if not df_products.empty:
        with prod_tab1:
            st.subheader("Key Product Metrics at a Glance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Products", f"{df_products['Product_ID'].nunique()}")
            col2.metric("Average Price", f"${df_products['Price'].mean():,.2f}")
            col3.metric("Most Recent Launch", f"{df_products['Launch_Date'].max().strftime('%b %d, %Y')}")

        with prod_tab2:
            st.subheader("Annual Product Launch Trends")
            launches_by_year = df_products.set_index('Launch_Date').resample('Y')['Product_ID'].count().reset_index()
            launches_by_year['Launch_Date'] = launches_by_year['Launch_Date'].dt.year.astype(str)
            fig_launches = px.bar(launches_by_year, x='Launch_Date', y='Product_ID', labels={'Launch_Date': 'Year', 'Product_ID': 'Number of Launches'})
            st.plotly_chart(fig_launches, use_container_width=True)

        with prod_tab3:
            st.subheader("Historical Pricing Trends")
            price_by_year = df_products.set_index('Launch_Date')['Price'].resample('Y').mean().reset_index()
            price_by_year['Launch_Date'] = price_by_year['Launch_Date'].dt.year.astype(str)
            fig_price = px.line(price_by_year, x='Launch_Date', y='Price', labels={'Launch_Date': 'Year', 'Price': 'Average Price (USD)'}, markers=True)
            st.plotly_chart(fig_price, use_container_width=True)

        with prod_tab4:
            st.subheader("Predictive Forecasting for Product Pricing")
            forecast_days = st.slider("Select Forecast Period (Days)", 90, 730, 365, key="product_forecast_slider")
            st.info(f"The model is predicting the average price of new products for the next **{forecast_days} days**.")
            
            model, forecast = run_prophet_forecast(df_products, 'Launch_Date', 'Price', periods=forecast_days)
            if forecast is not None:
                fig_forecast = plot_plotly(model, forecast)
                fig_forecast.update_layout(title="Forecast of Average Price for New Products", xaxis_title="Date", yaxis_title="Predicted Avg. Price")
                st.plotly_chart(fig_forecast, use_container_width=True)
            else:
                st.warning("Not enough data for price forecasting. At least two data points are required.")
    else:
        st.warning("`products.csv` is empty or not found. Please add data to begin analysis.")

with tab2:
    st.header("Customer Activity Dashboard")
    df_sales = st.session_state.sales_df

    if not df_sales.empty:
        with st.container():
            st.subheader("Key Customer Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", f"{df_sales['TransactionID'].nunique():,}")
            col2.metric("Unique Customers", f"{df_sales['CustomerID'].nunique():,}")
            col3.metric("Total Revenue", f"${df_sales['TotalPrice'].sum():,.2f}")
            st.markdown("---")

        with st.container(border=True):
            st.subheader("Predictive Forecasting for Customer Transactions")
            forecast_days_customer = st.slider("Select Forecast Period (Days)", 90, 730, 365, key="customer_forecast_slider")
            st.info(f"The model is predicting the number of daily customer transactions for the next **{forecast_days_customer} days**.")
            
            daily_transactions = df_sales.groupby('TransactionDate')['TransactionID'].count().reset_index()
            model, forecast = run_prophet_forecast(daily_transactions, 'TransactionDate', 'TransactionID', periods=forecast_days_customer)

            if forecast is not None:
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Transactions', line=dict(color='#f63366')))
                fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line=dict(color='rgba(246, 51, 102, 0.2)')))
                fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', name='Confidence Interval', line=dict(color='rgba(246, 51, 102, 0.2)')))
                fig_forecast.update_layout(title="Forecast of Daily Customer Transactions", xaxis_title="Date", yaxis_title="Predicted Transactions")
                st.plotly_chart(fig_forecast, use_container_width=True)
            else:
                st.warning("Not enough data for transaction forecasting. At least two transactions on different days are required.")
    else:
        st.warning("`sales_transactions.csv` is empty or not found. Please add data to begin analysis.")