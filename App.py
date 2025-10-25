import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('kmeans_model.pkl')
    return model

k_means = load_model()

# App title and description
st.title("🛍️ Mall Customer Segmentation")
st.write("Predict customer segments based on Annual Income and Spending Score")

# Sidebar for user input
st.sidebar.header("Enter Customer Details")

annual_income = st.sidebar.slider(
    "Annual Income (k$)", 
    min_value=15, 
    max_value=140, 
    value=50, 
    step=1
)

spending_score = st.sidebar.slider(
    "Spending Score (1-100)", 
    min_value=1, 
    max_value=100, 
    value=50, 
    step=1
)

# Predict button
if st.sidebar.button("🔮 Predict Cluster"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'Annual Income (k$)': [annual_income],
        'Spending Score (1-100)': [spending_score]
    })
    
    # Make prediction
    cluster = k_means.predict(input_data)[0]
    
    st.success(f"### Customer belongs to Cluster: **{cluster}**")
    
    # Display cluster characteristics
    cluster_info = {
        0: "💰 High Income, Low Spending - Target for premium products",
        1: "🎯 High Income, High Spending - VIP customers",
        2: "💵 Medium Income, Medium Spending - Regular customers",
        3: "💸 Low Income, High Spending - Budget-conscious spenders",
        4: "🛑 Low Income, Low Spending - Price-sensitive customers"
    }
    
    if cluster < len(cluster_info):
        st.info(cluster_info[cluster])

# Visualization section
st.header("📊 Cluster Visualization")

# Sample data for visualization (you can load your actual data)
fig, ax = plt.subplots(figsize=(10, 6))

# Plot cluster centers
centers = k_means.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], 
           c='red', marker='X', s=300, 
           edgecolors='black', linewidths=2,
           label='Cluster Centers')

# Plot user input
ax.scatter(annual_income, spending_score, 
           c='blue', marker='o', s=200,
           edgecolors='black', linewidths=2,
           label='Your Input')

ax.set_xlabel('Annual Income (k$)', fontsize=12)
ax.set_ylabel('Spending Score (1-100)', fontsize=12)
ax.set_title('K-Means Clustering - Mall Customers', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# Model info
with st.expander("ℹ️ About the Model"):
    st.write("""
    - **Algorithm:** K-Means Clustering
    - **Number of Clusters:** 5
    - **Features:** Annual Income, Spending Score
    - **Use Case:** Customer Segmentation for targeted marketing
    """)
