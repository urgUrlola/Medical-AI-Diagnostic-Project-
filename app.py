# =============================================================================
# AUTO IMMUNE DISEASE DIAGNOSIS DASHBOARD
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Set page config
st.set_page_config(
    page_title="Autoimmune Disease Diagnosis",
    page_icon="🩺",
    layout="wide"
)

# Title and description
st.title("🩺 Autoimmune Digestive Disease Diagnosis Dashboard")
st.markdown("""
This dashboard visualizes our two-dataset approach to autoimmune disease prediction:
1. **Dataset 1**: Gastrointestinal clinical patterns
2. **Dataset 2**: Autoimmune diagnostic markers
""")

# Load your models and data
@st.cache_data
def load_data_and_models():
    """Load datasets and trained models"""
    try:
        # Load Dataset 1 (your gastrointestinal data)
        df1 = pd.read_csv("gastrointestinal_disease_dataset.csv")
        
        # Load Dataset 2 (your autoimmune data)  
        df2 = pd.read_csv("Autoimmune_Disorder_10k_with_All_Disorders.csv")
        
        # Filter to digestive diseases
        digestive = ["Celiac disease", "Crohn's disease", "Ulcerative colitis", 
                    "Autoimmune hepatitis", "Primary biliary cholangitis"]
        df2_filtered = df2[df2["Diagnosis"].isin(digestive)].copy()
        
        # Load trained model
        model = joblib.load("digestive_diagnosis_with_dataset1_insights.joblib")
        
        return df1, df2_filtered, model
    except:
        st.warning("Could not load files. Using sample data for demo.")
        # Create sample data for demo
        df1 = pd.DataFrame({
            'CRP_ESR': np.random.normal(50, 15, 1000),
            'Fecal_Calprotectin': np.random.normal(250, 50, 1000),
            'Age': np.random.randint(18, 85, 1000),
            'Symptom_Complexity': np.random.randint(0, 5, 1000)
        })
        df2 = pd.DataFrame({
            'CRP': np.random.normal(10, 5, 500),
            'ESR': np.random.normal(20, 10, 500),
            'Diagnosis': np.random.choice(['Celiac', "Crohn's", "Ulcerative colitis"], 500)
        })
        return df1, df2, None

# Load data
df1, df2, model = load_data_and_models()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", 
                       ["📊 Dataset Overview", "🔬 Model Insights", 
                        "🎯 Live Prediction", "📈 Performance Metrics"])

# =============================================================================
# PAGE 1: Dataset Overview
# =============================================================================
if page == "📊 Dataset Overview":
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset 1: Gastrointestinal Clinical Data")
        st.metric("Patients", f"{len(df1):,}")
        st.metric("Features", df1.shape[1])
        
        # Show feature distributions
        fig1, ax1 = plt.subplots()
        if 'CRP_ESR' in df1.columns:
            df1['CRP_ESR'].hist(bins=30, ax=ax1, alpha=0.7)
            ax1.set_xlabel("CRP/ESR (Inflammation Marker)")
            ax1.set_ylabel("Count")
            ax1.set_title("Dataset 1: CRP/ESR Distribution")
            st.pyplot(fig1)
    
    with col2:
        st.subheader("Dataset 2: Autoimmune Diagnostic Data")
        st.metric("Patients", f"{len(df2):,}")
        st.metric("Diseases", df2['Diagnosis'].nunique() if 'Diagnosis' in df2.columns else "N/A")
        
        # Show disease distribution
        if 'Diagnosis' in df2.columns:
            fig2, ax2 = plt.subplots()
            df2['Diagnosis'].value_counts().plot(kind='bar', ax=ax2)
            ax2.set_title("Disease Distribution")
            ax2.set_xlabel("Disease")
            ax2.set_ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig2)
    
    # Dataset connection visualization
    st.subheader("Dataset Connection Strategy")
    
    connection_data = pd.DataFrame({
        'Step': ['Dataset 1 Analysis', 'Pattern Discovery', 'Feature Engineering', 'Dataset 2 Enhancement'],
        'Description': [
            'Analyzed clinical patterns in GI data',
            'Found relationships between markers and symptoms',
            'Created new features based on patterns',
            'Improved disease classification accuracy'
        ],
        'Impact': ['Found data quality issues', 'Identified 3 patient clusters', 'Added 4 new features', '+X% accuracy improvement']
    })
    
    st.table(connection_data)

# =============================================================================
# PAGE 2: Model Insights
# =============================================================================
elif page == "🔬 Model Insights":
    st.header("Model Insights & Feature Importance")
    
    # Feature importance visualization
    st.subheader("Feature Importance from Dataset 1 Analysis")
    
    # Simulated feature importance from your analysis
    feature_importance_data = {
        'Feature': ['CRP_ESR', 'Fecal_Calprotectin', 'Genetic_Markers', 'Age', 
                   'Stress_Level', 'BMI', 'Physical_Activity', 'Bowel_Movement_Frequency'],
        'Importance': [0.18, 0.16, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07]
    }
    
    importance_df = pd.DataFrame(feature_importance_data)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', 
                 orientation='h', title="Top Features from Dataset 1 Analysis",
                 color='Importance', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    # Patient clusters from Dataset 1
    st.subheader("Patient Clusters Discovered in Dataset 1")
    
    cluster_data = pd.DataFrame({
        'Cluster': ['Older Patients', 'Inflammatory Profile', 'Genetic Risk'],
        'Size': [11462, 9545, 9553],
        'Avg Age': [73.5, 31.3, 31.6],
        'Avg CRP_ESR': [49.8, 50.4, 50.7],
        'Key Characteristic': ['Age-related issues', 'High inflammation', 'Strong genetic markers']
    })
    
    st.dataframe(cluster_data)
    
    # Show how these patterns influenced Dataset 2
    st.subheader("How Dataset 1 Patterns Enhanced Dataset 2")
    
    enhancement_data = pd.DataFrame({
        'Dataset 1 Pattern': ['CRP_ESR patterns', 'Symptom complexity', 'Genetic markers', 'Patient clustering'],
        'Dataset 2 Feature': ['Inflammation_Risk_Score', 'Symptom_Complexity', 'Autoantibody_Burden', 'Clinical_Profile_Flag'],
        'Impact': ['Better inflammation assessment', 'More symptom context', 'Enhanced risk profiling', 'Patient stratification']
    })
    
    st.table(enhancement_data)

# =============================================================================
# PAGE 3: Live Prediction
# =============================================================================
elif page == "🎯 Live Prediction":
    st.header("Live Disease Prediction")
    
    st.info("""
    This demo shows how our connected model would work in practice. 
    Enter patient values to see the predicted diagnosis.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Clinical Data")
        
        # Input fields for key features
        crp = st.slider("CRP (mg/L)", 0.0, 100.0, 10.0, 0.1)
        esr = st.slider("ESR (mm/hr)", 0, 100, 20)
        age = st.slider("Age", 18, 85, 45)
        
        # Symptom checkboxes
        st.subheader("Symptoms")
        fatigue = st.checkbox("Fatigue")
        joint_pain = st.checkbox("Joint Pain")
        weight_loss = st.checkbox("Weight Loss")
        rash = st.checkbox("Skin Rash")
        
    with col2:
        st.subheader("Autoantibody Levels")
        
        # Autoantibody sliders
        ana = st.select_slider("ANA Titer", options=['Negative', '1:40', '1:80', '1:160', '1:320'])
        anti_dsdna = st.checkbox("Anti-dsDNA Positive")
        rf = st.checkbox("Rheumatoid Factor Positive")
        
        # Additional markers
        wbc = st.slider("WBC Count (10³/µL)", 2.0, 20.0, 7.5, 0.1)
        platelet = st.slider("Platelet Count (10³/µL)", 100, 500, 250)
    
    # Prediction button
    if st.button("🔍 Predict Diagnosis", type="primary"):
        st.success("### Prediction Results")
        
        # Calculate derived features (simulating your model logic)
        inflammation_score = (crp * 0.4) + (esr * 0.3)
        symptom_count = sum([fatigue, joint_pain, weight_loss, rash])
        
        # Mock prediction logic
        diseases = ["Celiac Disease", "Crohn's Disease", "Ulcerative Colitis", 
                   "Autoimmune Hepatitis", "Primary Biliary Cholangitis"]
        
        # Simple scoring system for demo
        scores = {
            "Celiac Disease": (anti_dsdna * 30) + (symptom_count * 10),
            "Crohn's Disease": (inflammation_score * 2) + (weight_loss * 20),
            "Ulcerative Colitis": (inflammation_score * 1.5) + (symptom_count * 15),
            "Autoimmune Hepatitis": (crp * 3) + (ana != 'Negative') * 25,
            "Primary Biliary Cholangitis": (age > 40) * 20 + (fatigue * 15)
        }
        
        predicted_disease = max(scores, key=scores.get)
        confidence = min(95, scores[predicted_disease])
        
        # Display results
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Predicted Disease", predicted_disease)
            st.metric("Confidence", f"{confidence:.1f}%")
        
        with col_b:
            # Show contributing factors
            st.write("**Key Contributing Factors:**")
            if inflammation_score > 20:
                st.write("• High inflammation markers")
            if symptom_count >= 2:
                st.write(f"• Multiple symptoms ({symptom_count})")
            if anti_dsdna:
                st.write("• Anti-dsDNA positive")
        
        # Show probabilities
        st.subheader("Disease Probabilities")
        prob_df = pd.DataFrame({
            'Disease': list(scores.keys()),
            'Probability': [min(100, s) for s in scores.values()]
        }).sort_values('Probability', ascending=False)
        
        fig = px.bar(prob_df, x='Probability', y='Disease', 
                     orientation='h', color='Probability',
                     color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 4: Performance Metrics
# =============================================================================
elif page == "📈 Performance Metrics":
    st.header("Model Performance Metrics")
    
    # Performance comparison
    st.subheader("Accuracy Comparison: With vs Without Dataset 1 Insights")
    
    comparison_data = pd.DataFrame({
        'Model': ['Dataset 2 Only', 'Dataset 1 + Dataset 2'],
        'Accuracy': [0.782, 0.815],  # Replace with your actual numbers
        'Precision': [0.76, 0.79],
        'Recall': [0.78, 0.82],
        'F1-Score': [0.77, 0.80]
    })
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Accuracy", f"{comparison_data.loc[1, 'Accuracy']:.3f}")
    with col2:
        improvement = comparison_data.loc[1, 'Accuracy'] - comparison_data.loc[0, 'Accuracy']
        st.metric("Improvement", f"+{improvement:.3f}", delta=f"+{improvement*100:.1f}%")
    with col3:
        st.metric("Precision", f"{comparison_data.loc[1, 'Precision']:.3f}")
    with col4:
        st.metric("Recall", f"{comparison_data.loc[1, 'Recall']:.3f}")
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Dataset 2 Only',
        x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        y=comparison_data.iloc[0, 1:],
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        name='With Dataset 1 Insights',
        x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        y=comparison_data.iloc[1, 1:],
        marker_color='royalblue'
    ))
    
    fig.update_layout(
        title="Performance Metrics Comparison",
        barmode='group',
        yaxis_title="Score",
        yaxis_range=[0.7, 0.85]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix (Example)")
    
    # Create example confusion matrix
    diseases = ["Celiac", "Crohn's", "Ulcerative Colitis"]
    cm = np.array([[45, 5, 2],
                   [3, 48, 4],
                   [1, 3, 39]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=diseases, yticklabels=diseases,
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Example Confusion Matrix')
    st.pyplot(fig)
