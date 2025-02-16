import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page configuration for better layout
st.set_page_config(layout="centered", page_title="Customer Segmentation Dashboard")

# Custom Styling
st.markdown("""
    <style>
        .main { max-width: 900px; margin: auto; }
        h1 { color: #2C3E50; font-family: 'Arial', sans-serif; text-align: center; font-size: 36px; font-weight: bold; }
        h2 { color: #E67E22; font-family: 'Arial', sans-serif; font-size: 24px; }
        p, label { font-family: 'Arial', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<h1>Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
st.image('https://clevertap.com/wp-content/uploads/2024/01/RFM-Analysis-for-Customer-Segmentation-A-Comprehensive-Guide.png?w=1024', use_container_width=True)
st.markdown('<p style="text-align:center; color:#34495E; font-size:18px;">Gain actionable insights and tailor marketing strategies for different customer segments.</p>', unsafe_allow_html=True)

# Data Loading
st.markdown('<h2>Dataset Overview</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader('Upload your CSV file', type='csv')

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.write('Data Preview:', df.head())
        
        # Ensure necessary columns exist
        required_columns = {'ORDERDATE', 'CUSTOMERNAME', 'ORDERNUMBER', 'SALES'}
        if not required_columns.issubset(df.columns):
            st.error(f"Missing required columns: {required_columns - set(df.columns)}")
        else:
            # RFM Calculation
            st.markdown('<h2>RFM Analysis</h2>', unsafe_allow_html=True)
            df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
            df = df.dropna(subset=['ORDERDATE'])
            df['Recency'] = (df['ORDERDATE'].max() - df['ORDERDATE']).dt.days
            df['Frequency'] = df.groupby('CUSTOMERNAME')['ORDERNUMBER'].transform('count')
            df['MonetaryValue'] = df.groupby('CUSTOMERNAME')['SALES'].transform('sum')
            rfm_df = df[['CUSTOMERNAME', 'Recency', 'Frequency', 'MonetaryValue']].drop_duplicates().set_index('CUSTOMERNAME')
            st.write('RFM Table:', rfm_df.head())
            
            # Scaling
            scaler = StandardScaler()
            scaled_rfm = scaler.fit_transform(rfm_df)
            
            # KMeans Clustering
            st.markdown('<h2>Customer Segments</h2>', unsafe_allow_html=True)
            num_clusters = st.slider('Select number of clusters:', 2, 5, 4)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            kmeans.fit(scaled_rfm)
            rfm_df['Cluster'] = kmeans.labels_
            
            # Segment Naming
            segment_map = {0: 'departing', 1: 'active', 2: 'inactive', 3: 'new'}
            rfm_df['SegmentName'] = rfm_df['Cluster'].map(segment_map).fillna('Other')
            st.write('Customer Segments:', rfm_df.head())
            
            # Filter and Drill-Down
            st.markdown('<h3>Explore Customer Segments</h3>', unsafe_allow_html=True)
            selected_segment = st.selectbox('Select Segment', rfm_df['SegmentName'].unique())
            st.write(rfm_df[rfm_df['SegmentName'] == selected_segment])
            
            # Visualizations
            st.markdown('<h2>Segment Distribution</h2>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            rfm_df['SegmentName'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#FFA07A', '#20B2AA', '#778899', '#FFDEAD'], ax=ax)
            ax.set_ylabel('')
            st.pyplot(fig)
            
            # Business Recommendations
            st.markdown('<h2>Business Recommendations</h2>', unsafe_allow_html=True)
            st.markdown('''
**Strategic Insights for Each Segment:**
- **Active:** Maintain engagement through loyalty programs.
- **New:** Focus on onboarding campaigns.
- **Inactive:** Re-engagement strategies.
- **Departing:** Win-back campaigns.
            ''')
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning('Please upload a CSV file to proceed.')
