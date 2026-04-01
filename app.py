import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Customer Segmentation", layout="wide")

st.title("🤖 AI-Powered Customer Segmentation System")
st.write("This application leverages Machine Learning to automatically identify distinct customer profiles.")

st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    
    st.sidebar.header("⚙️ Algorithm Settings")
    k = st.sidebar.slider("Number of Segments (K)", 2, 10, 5)
    
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = KMeans(n_clusters=k, init='k-means++', random_state=42)
    df['Cluster'] = model.fit_predict(X_scaled)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📍 Segment Map")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=df, x=features[0], y=features[1], 
            hue='Cluster', palette='viridis', s=150, ax=ax, alpha=0.7
        )
        plt.title(f"Segmentation into {k} Groups")
        st.pyplot(fig)
        
    with col2:
        st.subheader("📝 Automated Interpretation")
        analysis = df.groupby('Cluster')[features].mean()
        
        for i in range(k):
            income = analysis.loc[i, features[0]]
            spending = analysis.loc[i, features[1]]
            
            if income > 70 and spending > 70:
                label = "💎 VIP Customers"
                advice = "Focus on retention and exclusive offers."
            elif income > 70 and spending < 40:
                label = "💰 High-Income Savers"
                advice = "Target with value-based and long-term benefits."
            elif income < 45 and spending > 70:
                label = "🛍️ Active Spenders"
                advice = "Ideal for new arrivals and impulse purchases."
            elif income < 45 and spending < 40:
                label = "🐌 Low-Activity"
                advice = "Requires aggressive promotions to re-engage."
            else:
                label = "📊 Standard Class"
                advice = "Standard loyalty support and regular updates."

            with st.expander(f"Segment {i}: {label}"):
                st.write(f"**Avg Income:** {income:.1f}k$ | **Avg Spending Score:** {spending:.1f}")
                st.info(f"💡 {advice}")

    st.divider()
    st.subheader("📂 Processed Dataset with AI Labels")
    st.dataframe(df.style.background_gradient(subset=['Cluster'], cmap='viridis'))
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name='segmented_customers.csv',
        mime='text/csv',
    )

else:
    st.info("Please upload a CSV file via the sidebar to start the analysis.")
