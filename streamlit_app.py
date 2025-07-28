import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Import your TrustScore analyzer
try:
    from trustscore_batch_analyzer import TrustScoreAnalyzer
except ImportError:
    st.error("Please make sure trustscore_batch_analyzer.py is in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="TrustScore Loan Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .high-approval {
        background-color: #1e3a2e;
        border-left: 4px solid #28a745;
        color: #ffffff;
    }
    .moderate-approval {
        background-color: #3d3a1e;
        border-left: 4px solid #ffc107;
        color: #ffffff;
    }
    .low-approval {
        background-color: #3a1e1e;
        border-left: 4px solid #dc3545;
        color: #ffffff;
    }
    .likely-decline {
        background-color: #2d1517;
        border-left: 4px solid #721c24;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained Random Forest model"""
    try:
        model = joblib.load("trustscore_rf_model_5000 final.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'trustscore_rf_model_5000 final.pkl' not found. Please upload the model file.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def initialize_analyzer():
    """Initialize the TrustScore analyzer"""
    return TrustScoreAnalyzer()

def get_feature_columns():
    """Get the expected feature columns for the model"""
    # This should match the columns used during training
    base_features = [
        'income', 'credit_score', 'loan_amount', 'loan_term', 'age', 'monthly_debt'
    ]
    
    # Categorical features (one-hot encoded)
    categorical_features = [
        'employment_parttime', 'employment_selfemployed', 'employment_student', 'employment_unemployed',
        'loan_purpose_education', 'loan_purpose_home', 'loan_purpose_medical', 'loan_purpose_other',
        'gender_male', 'gender_prefer_not',
        'residence_parents', 'residence_rent'
    ]
    
    return base_features + categorical_features

def prepare_input_data(input_dict):
    """Prepare input data for model prediction"""
    # Create base features
    features_df = pd.DataFrame([{
        'income': input_dict['income'],
        'credit_score': input_dict['credit_score'],
        'loan_amount': input_dict['loan_amount'],
        'loan_term': input_dict['loan_term'],
        'age': input_dict['age'],
        'monthly_debt': input_dict['monthly_debt']
    }])
    
    # Add categorical features (one-hot encoded)
    # Employment
    employment_cols = ['employment_parttime', 'employment_selfemployed', 'employment_student', 'employment_unemployed']
    for col in employment_cols:
        features_df[col] = 0
    if input_dict['employment'] == 'parttime':
        features_df['employment_parttime'] = 1
    elif input_dict['employment'] == 'selfemployed':
        features_df['employment_selfemployed'] = 1
    elif input_dict['employment'] == 'student':
        features_df['employment_student'] = 1
    elif input_dict['employment'] == 'unemployed':
        features_df['employment_unemployed'] = 1
    
    # Loan Purpose
    purpose_cols = ['loan_purpose_education', 'loan_purpose_home', 'loan_purpose_medical', 'loan_purpose_other']
    for col in purpose_cols:
        features_df[col] = 0
    if input_dict['loan_purpose'] == 'education':
        features_df['loan_purpose_education'] = 1
    elif input_dict['loan_purpose'] == 'home':
        features_df['loan_purpose_home'] = 1
    elif input_dict['loan_purpose'] == 'medical':
        features_df['loan_purpose_medical'] = 1
    elif input_dict['loan_purpose'] == 'other':
        features_df['loan_purpose_other'] = 1
    
    # Gender
    gender_cols = ['gender_male', 'gender_prefer_not']
    for col in gender_cols:
        features_df[col] = 0
    if input_dict['gender'] == 'male':
        features_df['gender_male'] = 1
    elif input_dict['gender'] == 'prefer_not':
        features_df['gender_prefer_not'] = 1
    
    # Residence
    residence_cols = ['residence_parents', 'residence_rent']
    for col in residence_cols:
        features_df[col] = 0
    if input_dict['residence'] == 'parents':
        features_df['residence_parents'] = 1
    elif input_dict['residence'] == 'rent':
        features_df['residence_rent'] = 1
    
    # Ensure all expected columns are present
    expected_cols = get_feature_columns()
    for col in expected_cols:
        if col not in features_df.columns:
            features_df[col] = 0
    
    return features_df[expected_cols]

def create_prediction_visualization(prediction, probabilities, trustscore_result):
    """Create visualizations for the prediction"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Prediction Confidence', 'TrustScore Components', 'Risk Assessment', 'Financial Ratios'),
        specs=[[{"type": "indicator"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "bar"}]]
    )
    
    # 1. Prediction Confidence Gauge
    max_prob = max(probabilities) * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=max_prob,
            domain={'x': [0, 1], 'y': [0.2, 1]},
            title={
                'text': f"{prediction.replace('_', ' ').title()}<br><span style='font-size:0.8em;color:gray'>Confidence %</span>",
                'font': {'size': 16}
            },
            number={'font': {'size': 40}},
            gauge={
                'axis': {
                    'range': [None, 100],
                    'tickwidth': 1,
                    'tickcolor': "darkblue"
                },
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=1
    )
    
    # 2. TrustScore Components
    components = ['Overall', 'Fairness', 'Transparency', 'Privacy', 'Credit Risk']
    scores = [
        trustscore_result['overall_score'],
        trustscore_result['fairness_score'],
        trustscore_result['transparency_score'],
        trustscore_result['privacy_score'],
        trustscore_result['credit_risk_score']
    ]
    
    fig.add_trace(
        go.Bar(x=components, y=scores, name='Scores', marker_color='lightblue'),
        row=1, col=2
    )
    
    # 3. Risk Category Pie
    risk_categories = ['Assessed Risk', 'Remaining Risk']
    risk_values = [trustscore_result['overall_score'], 100 - trustscore_result['overall_score']]
    
    fig.add_trace(
        go.Pie(labels=risk_categories, values=risk_values, name="Risk"),
        row=2, col=1
    )
    
    # 4. Financial Ratios
    ratios = ['Debt-to-Income', 'Total Debt Ratio', 'Loan-to-Income']
    ratio_values = [
        trustscore_result['debt_to_income_ratio'],
        trustscore_result['total_debt_ratio'],
        trustscore_result['loan_to_income_ratio']
    ]
    
    fig.add_trace(
        go.Bar(x=ratios, y=ratio_values, name='Ratios (%)', marker_color='orange'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Prediction Analysis Dashboard")
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 TrustScore Loan Assessment System</h1>', unsafe_allow_html=True)
    
    # Load model and analyzer
    model = load_model()
    analyzer = initialize_analyzer()
    
    if model is None:
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("📝 Loan Application Details")
    
    # Personal Information
    st.sidebar.subheader("Personal Information")
    age = st.sidebar.slider("Age", 18, 80, 35)
    gender = st.sidebar.selectbox("Gender", ["female", "male", "prefer_not"])
    employment = st.sidebar.selectbox("Employment Status", 
                                     ["fulltime", "selfemployed", "parttime", "unemployed", "student"])
    residence = st.sidebar.selectbox("Residence", ["own", "rent", "parents"])
    
    # Financial Information
    st.sidebar.subheader("Financial Information")
    income = st.sidebar.number_input("Annual Income ($)", 15000, 200000, 50000, step=1000)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    monthly_debt = st.sidebar.number_input("Monthly Debt Payments ($)", 0, 5000, 800, step=50)
    
    # Loan Information
    st.sidebar.subheader("Loan Details")
    loan_amount = st.sidebar.number_input("Loan Amount ($)", 1000, 100000, 25000, step=1000)
    loan_term = st.sidebar.selectbox("Loan Term (years)", [1, 2, 3, 5, 7, 10, 15, 20, 30])
    loan_purpose = st.sidebar.selectbox("Loan Purpose", ["car", "home", "education", "medical", "other"])
    
    # Prediction button
    if st.sidebar.button("🔍 Analyze Application", type="primary"):
        
        # Prepare input data
        input_data = {
            'age': age,
            'gender': gender,
            'employment': employment,
            'residence': residence,
            'income': income,
            'credit_score': credit_score,
            'monthly_debt': monthly_debt,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'loan_purpose': loan_purpose
        }
        
        # Get TrustScore analysis
        trustscore_result = analyzer.calculate_trustscore_single(input_data)
        
        # Prepare data for ML model
        ml_input = prepare_input_data(input_data)
        
        # Make prediction
        prediction = model.predict(ml_input)[0]
        probabilities = model.predict_proba(ml_input)[0]
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Prediction Result
            decision_class = prediction.lower().replace('_', '-')
            st.markdown(f"""
            <div class="prediction-box {decision_class}">
                <h2>🎯 Prediction: {prediction.replace('_', ' ').title()}</h2>
                <h3>📊 TrustScore: {trustscore_result['overall_score']}/100</h3>
                <p><strong>Risk Category:</strong> {trustscore_result['risk_category']}</p>
                <p><strong>Monthly Payment:</strong> ${trustscore_result['monthly_payment']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualization
            viz_fig = create_prediction_visualization(prediction, probabilities, trustscore_result)
            st.plotly_chart(viz_fig, use_container_width=True)
        
        with col2:
            # Key Metrics
            st.subheader("📈 Key Metrics")
            
            # Score breakdown
            st.metric("Overall Score", f"{trustscore_result['overall_score']}/100")
            st.metric("Credit Risk Score", f"{trustscore_result['credit_risk_score']:.1f}/100")
            st.metric("Fairness Score", f"{trustscore_result['fairness_score']}/100")
            st.metric("Transparency Score", f"{trustscore_result['transparency_score']}/100")
            st.metric("Privacy Score", f"{trustscore_result['privacy_score']}/100")
            
            # Financial ratios
            st.subheader("💰 Financial Ratios")
            st.metric("Debt-to-Income", f"{trustscore_result['debt_to_income_ratio']:.1f}%")
            st.metric("Total Debt Ratio", f"{trustscore_result['total_debt_ratio']:.1f}%")
            st.metric("Loan-to-Income", f"{trustscore_result['loan_to_income_ratio']:.1f}%")
            
            # Model confidence
            st.subheader("🤖 Model Confidence")
            class_names = model.classes_
            for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                st.metric(class_name.replace('_', ' ').title(), f"{prob*100:.1f}%")
        
        # Detailed Analysis
        st.subheader("📋 Detailed Analysis")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.write("**Positive Factors:**")
            positive_factors = []
            
            if trustscore_result['credit_risk_score'] > 70:
                positive_factors.append("✅ Strong credit risk profile")
            if trustscore_result['fairness_score'] > 80:
                positive_factors.append("✅ High fairness score")
            if trustscore_result['debt_to_income_ratio'] < 30:
                positive_factors.append("✅ Low debt-to-income ratio")
            if income > 60000:
                positive_factors.append("✅ Strong income level")
            if employment == 'fulltime':
                positive_factors.append("✅ Stable employment")
            
            for factor in positive_factors:
                st.write(factor)
        
        with analysis_col2:
            st.write("**Areas of Concern:**")
            concerns = []
            
            if trustscore_result['credit_risk_score'] < 50:
                concerns.append("⚠️ Low credit risk score")
            if trustscore_result['total_debt_ratio'] > 43:
                concerns.append("⚠️ High total debt ratio")
            if credit_score < 600:
                concerns.append("⚠️ Low credit score")
            if employment in ['unemployed', 'parttime']:
                concerns.append("⚠️ Employment stability concerns")
            if trustscore_result['loan_to_income_ratio'] > 50:
                concerns.append("⚠️ High loan-to-income ratio")
            
            if not concerns:
                concerns.append("✅ No major concerns identified")
            
            for concern in concerns:
                st.write(concern)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if prediction in ['HIGH_APPROVAL', 'MODERATE_APPROVAL']:
            st.success("**Recommended Action:** Approve with standard terms")
            if trustscore_result['overall_score'] < 80:
                st.info("💡 Consider offering financial literacy resources to improve long-term success")
        elif prediction == 'LOW_APPROVAL':
            st.warning("**Recommended Action:** Approve with modified terms or require additional documentation")
            st.info("💡 Suggestions: Lower loan amount, longer term, or co-signer requirement")
        else:
            st.error("**Recommended Action:** Decline application")
            st.info("💡 Provide feedback on areas for improvement before reapplication")
    
    # Batch Processing Tab
    st.subheader("📊 Batch Processing")
    
    uploaded_file = st.file_uploader("Upload CSV file for batch processing", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.write(f"📁 Loaded {len(df)} records")
            st.write("Preview:", df.head())
            
            if st.button("🚀 Process Batch"):
                with st.spinner("Processing batch..."):
                    # Run TrustScore analysis
                    results_df = analyzer.analyze_batch(df)
                    
                    # Display summary
                    summary = analyzer.generate_summary_report(results_df)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", summary['total_records'])
                    with col2:
                        st.metric("Approval Rate", f"{summary['approval_rate']:.1f}%")
                    with col3:
                        st.metric("High Risk", summary['high_risk_count'])
                    with col4:
                        st.metric("Avg Score", f"{summary['score_statistics']['overall_score']['mean']:.1f}")
                    
                    # Decision distribution chart
                    decision_df = pd.DataFrame(list(summary['decision_distribution'].items()), 
                                             columns=['Decision', 'Count'])
                    fig = px.pie(decision_df, values='Count', names='Decision', 
                               title='Decision Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="trustscore_batch_results.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏦 TrustScore Loan Assessment System | Built with Streamlit</p>
        <p>⚖️ Promoting Fair and Transparent Lending Decisions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()