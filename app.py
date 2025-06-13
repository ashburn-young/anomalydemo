"""
Retail Anomaly Detection Dashboard with Human-in-the-Loop Verification
A comprehensive solution for detecting anomalies in retail data using Azure AI services.
"""

# Force OpenAI version compatibility
import os
import sys
import subprocess
import importlib

# Check OpenAI version and downgrade if needed
try:
    import openai
    with open('debug_env.log', 'a') as f:
        f.write(f'APP_START_OPENAI_VERSION: {openai.__version__}\n')
    # If using OpenAI >= 1.0.0, downgrade to 0.28.1
    if openai.__version__.startswith('1.') or openai.__version__.startswith('2.'):
        print("Detected incompatible OpenAI version, downgrading to 0.28.1...")
        try:
            with open('debug_env.log', 'a') as f:
                f.write(f'STREAMLIT_DOWNGRADING_OPENAI: from {openai.__version__} to 0.28.1\n')
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openai==0.28.1", "--force-reinstall"])
            # Reload openai module
            if 'openai' in sys.modules:
                del sys.modules['openai']
            import openai
            with open('debug_env.log', 'a') as f:
                f.write(f'STREAMLIT_AFTER_DOWNGRADE: {openai.__version__}\n')
        except Exception as e:
            with open('debug_env.log', 'a') as f:
                f.write(f'STREAMLIT_DOWNGRADE_ERROR: {str(e)}\n')
except Exception as e:
    with open('debug_env.log', 'a') as f:
        f.write(f'APP_START_IMPORT_ERROR: {str(e)}\n')

# Now import the rest of the required modules
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Import our custom modules
from azure_services import AzureOpenAIService, AzureStorageService
from data_processor import RetailDataProcessor

# Configure Streamlit page
st.set_page_config(
    page_title="Retail Anomaly Detection",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'anomaly_results' not in st.session_state:
    st.session_state.anomaly_results = None
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []

# Apply dark mode styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1f2937;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .anomaly-card {
        background: #1f2937;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ef4444;
        margin: 0.5rem 0;
    }
    .success-card {
        background: #1f2937;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #10b981;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1e40af 0%, #1e3a8a 100%);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛒 Kroger Anomaly Detection Dashboard</h1>
        <p>Intelligent solution for detecting anomalies in retail data with AI-powered analysis and human verification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize services
    openai_service = None
    storage_service = None
    data_processor = None
    
    try:
        data_processor = RetailDataProcessor()
        st.success("✅ Data processor initialized successfully")
    except Exception as e:
        st.error(f"❌ Failed to initialize data processor: {str(e)}")
        st.stop()
    
    try:
        openai_service = AzureOpenAIService()
        # Enhanced debug logging
        with open('debug_env.log', 'a') as f:
            f.write(f'STREAMLIT_INIT_OPENAI_VERSION: {openai.__version__}\n')
            
        if openai_service.client:
            st.success("✅ Azure OpenAI service initialized successfully")
            with open('debug_env.log', 'a') as f:
                f.write(f'STREAMLIT_CLIENT_SUCCESS: True\n')
        else:
            st.warning("⚠️ Azure OpenAI service unavailable - statistical analysis only")
            with open('debug_env.log', 'a') as f:
                f.write(f'STREAMLIT_CLIENT_SUCCESS: False\n')
    except Exception as e:
        st.warning(f"⚠️ Azure OpenAI service unavailable: {str(e)}")
        with open('debug_env.log', 'a') as f:
            f.write(f'STREAMLIT_CLIENT_ERROR: {str(e)}\n')
        openai_service = None
    
    # Debug log after OpenAI service initialization
    if openai_service:
        with open('debug_env.log', 'a') as f:
            f.write(f'STREAMLIT_CLIENT: {"set" if openai_service.client else "None"}, DEPLOYMENT: {getattr(openai_service, "deployment_name", None)}\n')
    
    try:
        storage_service = AzureStorageService()
        st.success("✅ Azure Storage service initialized successfully")
    except Exception as e:
        st.warning(f"⚠️ Azure Storage service unavailable: {str(e)}")
        storage_service = None
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.header("🔧 Control Panel")
        
        # Upload section
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your retail dataset",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Supported formats: CSV, Excel, JSON"
        )
        
        # Analysis options
        st.subheader("⚙️ Analysis Options")
        enable_ai_analysis = st.checkbox("Enable AI Analysis", value=True, help="Use Azure OpenAI for intelligent anomaly detection")
        enable_statistical = st.checkbox("Enable Statistical Analysis", value=True, help="Use traditional statistical methods")
        
        # Feedback section
        st.subheader("📊 Session Stats")
        if st.session_state.feedback_history:
            approved_count = sum(1 for feedback in st.session_state.feedback_history if feedback.get('approved', False))
            total_feedback = len(st.session_state.feedback_history)
            st.metric("Feedback Given", total_feedback)
            st.metric("Anomalies Approved", approved_count)
            st.metric("Accuracy Rate", f"{(approved_count/total_feedback)*100:.1f}%" if total_feedback > 0 else "N/A")
    
    # Main content area
    if uploaded_file is not None:
        # Process uploaded data
        with st.spinner("Loading and validating data..."):
            df, validation_messages = data_processor.load_and_validate_data(uploaded_file)
        
        if not df.empty:
            # Display validation messages
            if validation_messages:
                with st.expander("📋 Data Validation Results", expanded=False):
                    for msg in validation_messages:
                        if "Warning" in msg:
                            st.warning(msg)
                        elif "Error" in msg:
                            st.error(msg)
                        else:
                            st.info(msg)
            
            # Data preprocessing
            with st.spinner("Preprocessing data..."):
                processed_df, preprocessing_summary = data_processor.preprocess_data(df)
                st.session_state.processed_data = processed_df
            
            # Display dataset overview
            display_dataset_overview(processed_df, preprocessing_summary)
            
            # Run anomaly detection
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🔍 Run Anomaly Detection", type="primary", use_container_width=True):
                    run_anomaly_detection(processed_df, data_processor, openai_service, enable_ai_analysis, enable_statistical)
            
            with col2:
                if st.button("💾 Save to Azure Storage", use_container_width=True):
                    if storage_service and storage_service.upload_processed_data(processed_df, f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"):
                        st.success("Data saved to Azure Storage successfully!")
                    else:
                        st.warning("Could not save to Azure Storage (storage service may not be configured)")
            
            # Display anomaly results if available
            if st.session_state.anomaly_results:
                display_anomaly_results(st.session_state.anomaly_results, processed_df, storage_service)
    
    else:
        # Welcome screen
        display_welcome_screen()

def display_welcome_screen():
    """Display welcome screen with instructions"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h2>Welcome to Kroger's Anomaly Detection</h2>
            <p>Upload your retail dataset to get started with AI-powered anomaly detection.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Upload your data** using the file uploader in the sidebar
        2. **Review validation results** to ensure data quality
        3. **Run anomaly detection** to identify unusual patterns
        4. **Provide feedback** to improve the AI model
        
        ### 📊 Supported Data Types
        
        - **Sales Transactions**: Revenue, quantity, pricing data
        - **Inventory Records**: Stock levels, movement patterns
        - **Customer Behavior**: Purchase patterns, frequency analysis
        - **Product Performance**: Sales trends, category analysis
        
        ### 🔧 Analysis Methods
        
        - **AI-Powered Analysis**: Azure OpenAI GPT-4o for contextual understanding
        - **Statistical Methods**: Z-score, IQR, Isolation Forest
        - **Human Verification**: Interactive feedback loop for continuous improvement
        """)

def display_dataset_overview(df: pd.DataFrame, preprocessing_summary: dict):
    """Display dataset overview and preprocessing results"""
    st.subheader("📊 Dataset Overview")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Records</h4>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Columns</h4>
            <h2>{len(df.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Numeric Columns</h4>
            <h2>{numeric_cols}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h4>Missing Data</h4>
            <h2>{missing_percentage:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Data preview and visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📈 Data Distribution")
        
        # Create distribution plot for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for distribution", numeric_cols)
            
            fig = px.histogram(
                df, 
                x=selected_col, 
                title=f"Distribution of {selected_col}",
                template="plotly_dark"
            )
            fig.update_layout(height=350, width=400)
            st.plotly_chart(fig, use_container_width=False)
    
    # Preprocessing summary
    if preprocessing_summary and 'transformations_applied' in preprocessing_summary:
        with st.expander("🔧 Preprocessing Summary", expanded=False):
            st.write(f"**Original Shape**: {preprocessing_summary.get('original_shape', 'N/A')}")
            st.write(f"**Final Shape**: {preprocessing_summary.get('final_shape', 'N/A')}")
            
            if preprocessing_summary['transformations_applied']:
                st.write("**Transformations Applied**:")
                for transformation in preprocessing_summary['transformations_applied']:
                    st.write(f"- {transformation}")

def run_anomaly_detection(df: pd.DataFrame, data_processor: RetailDataProcessor, 
                         openai_service: AzureOpenAIService, enable_ai: bool, enable_statistical: bool):
    """Run comprehensive anomaly detection"""
    
    results = {
        'ai_analysis': None,
        'statistical_analysis': None,
        'timestamp': datetime.now().isoformat()
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Statistical analysis
        if enable_statistical:
            status_text.text("Running statistical anomaly detection...")
            progress_bar.progress(25)
            
            statistical_results = data_processor.detect_statistical_anomalies(df)
            results['statistical_analysis'] = statistical_results
            
            st.success(f"Statistical analysis completed: {len(statistical_results.get('statistical_anomalies', []))} anomaly types detected")
        
        # AI analysis
        if enable_ai and openai_service and openai_service.client:
            status_text.text("Running AI-powered anomaly detection...")
            progress_bar.progress(50)
            
            data_summary = data_processor.get_data_summary(df)
            ai_results = openai_service.analyze_retail_data_for_anomalies(data_summary, df)
            results['ai_analysis'] = ai_results
            
            anomaly_count = len(ai_results.get('anomalies_detected', []))
            st.success(f"AI analysis completed: {anomaly_count} anomalies detected")
        elif enable_ai and not openai_service:
            st.warning("AI analysis requested but OpenAI service is not available")
        elif enable_ai and openai_service and not openai_service.client:
            st.warning("AI analysis requested but OpenAI client is not initialized")
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Store results in session state
        st.session_state.anomaly_results = results
        
    except Exception as e:
        st.error(f"Error during anomaly detection: {str(e)}")
        logger.error(f"Anomaly detection error: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

def display_anomaly_results(results: dict, df: pd.DataFrame, storage_service: AzureStorageService):
    """Display anomaly detection results with human feedback interface"""
    
    st.subheader("🔍 Anomaly Detection Results")
    
    # Create tabs for different analysis types
    tabs = []
    if results.get('ai_analysis'):
        tabs.append("🤖 AI Analysis")
    if results.get('statistical_analysis'):
        tabs.append("📊 Statistical Analysis")
    
    if tabs:
        tab_objects = st.tabs(tabs)
        
        # AI Analysis Tab
        if results.get('ai_analysis') and "🤖 AI Analysis" in tabs:
            with tab_objects[0 if "🤖 AI Analysis" in tabs else None]:
                display_ai_analysis_results(results['ai_analysis'], storage_service)
        
        # Statistical Analysis Tab
        if results.get('statistical_analysis') and "📊 Statistical Analysis" in tabs:
            tab_index = 1 if "🤖 AI Analysis" in tabs and "📊 Statistical Analysis" in tabs else 0
            with tab_objects[tab_index]:
                display_statistical_analysis_results(results['statistical_analysis'])
    
    # Visualization section
    st.subheader("📈 Data Visualization")
    create_anomaly_visualizations(df, results)

def display_ai_analysis_results(ai_results: dict, storage_service: AzureStorageService):
    """Display AI analysis results with human feedback"""
    
    # Summary
    if ai_results.get('summary'):
        st.markdown(f"""
        <div class="success-card">
            <h4>AI Analysis Summary</h4>
            <p>{ai_results['summary']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detected anomalies
    anomalies = ai_results.get('anomalies_detected', [])
    
    if anomalies:
        st.subheader("🚨 Detected Anomalies")
        
        for i, anomaly in enumerate(anomalies):
            with st.expander(f"Anomaly {i+1}: {anomaly.get('anomaly_type', 'Unknown')}", expanded=True):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description**: {anomaly.get('description', 'No description')}")
                    st.write(f"**Confidence Score**: {anomaly.get('confidence_score', 0):.2f}")
                    st.write(f"**Severity**: {anomaly.get('severity', 'Unknown')}")
                    
                    if anomaly.get('potential_causes'):
                        st.write("**Potential Causes**:")
                        for cause in anomaly['potential_causes']:
                            st.write(f"- {cause}")
                
                with col2:
                    st.write("**Human Verification**")
                    
                    feedback_key = f"feedback_{i}"
                    
                    # Feedback buttons
                    col_approve, col_reject = st.columns(2)
                    
                    with col_approve:
                        if st.button("✅ Approve", key=f"approve_{i}", use_container_width=True):
                            save_feedback(anomaly, True, storage_service, i)
                    
                    with col_reject:
                        if st.button("❌ Reject", key=f"reject_{i}", use_container_width=True):
                            save_feedback(anomaly, False, storage_service, i)
                    
                    # Additional feedback
                    feedback_notes = st.text_area(
                        "Additional Notes", 
                        key=f"notes_{i}",
                        placeholder="Optional: Add your reasoning or additional context..."
                    )
    
    # Recommendations
    if ai_results.get('recommendations'):
        st.subheader("💡 Recommendations")
        for rec in ai_results['recommendations']:
            st.write(f"- {rec}")

def display_statistical_analysis_results(statistical_results: dict):
    """Display statistical analysis results"""
    
    anomalies = statistical_results.get('statistical_anomalies', [])
    
    if anomalies:
        for anomaly in anomalies:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"{anomaly['method']} Anomalies",
                    anomaly['anomaly_count']
                )
            
            with col2:
                st.metric(
                    "Percentage",
                    f"{anomaly['percentage']:.2f}%"
                )
            
            with col3:
                st.info(anomaly['description'])
    
    # Summary statistics
    summary_stats = statistical_results.get('summary_stats', {})
    if summary_stats:
        with st.expander("📊 Analysis Summary", expanded=False):
            st.json(summary_stats)

def save_feedback(anomaly: dict, approved: bool, storage_service: AzureStorageService, anomaly_id: int):
    """Save user feedback"""
    
    feedback = {
        'anomaly_id': anomaly_id,
        'anomaly_type': anomaly.get('anomaly_type'),
        'approved': approved,
        'confidence_score': anomaly.get('confidence_score'),
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.get('session_id', 'unknown')
    }
    
    # Add to session history
    st.session_state.feedback_history.append(feedback)
    
    # Save to Azure Storage
    if storage_service.save_feedback(feedback):
        status = "approved" if approved else "rejected"
        st.success(f"Feedback saved: Anomaly {status}")
    else:
        st.warning("Feedback saved locally (Azure Storage not available)")

def create_anomaly_visualizations(df: pd.DataFrame, results: dict):
    """Create visualizations for anomaly detection results"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot
            x_col = st.selectbox("X-axis", numeric_cols, key="viz_x")
            y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="viz_y")
            
            fig = px.scatter(
                df, 
                x=x_col, 
                y=y_col,
                title=f"{x_col} vs {y_col}",
                template="plotly_dark"
            )
            
            # Add anomaly indicators if available
            # This is a simplified version - in a full implementation, 
            # you would mark actual detected anomalies
            
            fig.update_layout(height=350, width=450)
            st.plotly_chart(fig, use_container_width=False)
        
        with col2:
            # Box plot for outlier visualization
            selected_col = st.selectbox("Column for outlier analysis", numeric_cols, key="box_col")
            
            fig = px.box(
                df, 
                y=selected_col,
                title=f"Outlier Analysis: {selected_col}",
                template="plotly_dark"
            )
            
            fig.update_layout(height=350, width=450)
            st.plotly_chart(fig, use_container_width=False)
    
    # Correlation heatmap
    if len(numeric_cols) > 2:
        st.subheader("🔥 Correlation Heatmap")
        
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Matrix",
            template="plotly_dark",
            color_continuous_scale="RdBu"
        )
        
        fig.update_layout(height=400, width=600)
        st.plotly_chart(fig, use_container_width=False)

if __name__ == "__main__":
    main()
