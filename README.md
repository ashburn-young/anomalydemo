# Retail Anomaly Detection with Human-in-the-Loop Verification

A comprehensive solution for detecting anomalies in retail data using Azure AI services, featuring intelligent analysis with Azure OpenAI GPT-4o and human verification capabilities.

## 🎯 Overview

This solution provides:
- **AI-Powered Analysis**: Uses Azure OpenAI GPT-4o for contextual anomaly detection
- **Statistical Methods**: Traditional statistical analysis (Z-score, IQR, Isolation Forest)
- **Human Verification**: Interactive feedback loop for continuous model improvement
- **Data Storage**: Azure Blob Storage integration for data persistence
- **Interactive Dashboard**: Streamlit-based dark mode UI with visualizations

## 🏗️ Architecture

1. **Data Ingestion**: Support for CSV, Excel, and JSON files
2. **Data Preprocessing**: Automated cleaning and feature engineering
3. **Anomaly Detection**: Dual approach using AI and statistical methods
4. **Human Feedback**: Interactive UI for validating detected anomalies
5. **Storage & Learning**: Azure Storage for data persistence and feedback collection

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Azure OpenAI account with GPT-4o deployment
- Azure Storage account (optional, for data persistence)

### Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env`:
   ```
   AZURE_OPENAI_ENDPOINT=https://oaimodels.openai.azure.com/
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
   AZURE_OPENAI_API_VERSION=2025-01-01-preview
   AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string_here
   ```

### Generate Sample Data

Create a sample retail dataset for testing:

```bash
python generate_sample_data.py
```

This generates `sample_retail_data.csv` with realistic retail transactions including injected anomalies.

### Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Usage Guide

### 1. Data Upload
- Use the sidebar file uploader to upload your retail dataset
- Supported formats: CSV, Excel (.xlsx, .xls), JSON
- The system will automatically validate and preprocess your data

### 2. Data Analysis
- Review the dataset overview with key metrics and visualizations
- Enable/disable AI analysis and statistical analysis in the sidebar
- Click "Run Anomaly Detection" to start the analysis

### 3. Review Results
- **AI Analysis Tab**: View contextual anomalies detected by GPT-4o
- **Statistical Analysis Tab**: See traditional statistical anomaly detection results
- Each anomaly includes confidence scores, descriptions, and potential causes

### 4. Human Verification
- For each detected anomaly, use the ✅ Approve or ❌ Reject buttons
- Add optional notes explaining your reasoning
- Feedback is stored for continuous model improvement

### 5. Data Visualization
- Interactive charts for exploring data patterns
- Correlation heatmaps for understanding feature relationships
- Outlier analysis with box plots and scatter plots

## 🔍 Anomaly Types Detected

### AI-Powered Detection
- **Sales Pattern Anomalies**: Unusual transaction volumes or patterns
- **Pricing Irregularities**: Unexpected price changes or errors
- **Customer Behavior**: Suspicious purchasing patterns
- **Inventory Issues**: Unusual stock movements
- **Data Quality**: Inconsistent or erroneous data entries

### Statistical Detection
- **Z-Score Outliers**: Values more than 3 standard deviations from mean
- **IQR Outliers**: Values outside 1.5 * IQR from quartiles
- **Isolation Forest**: Machine learning-based outlier detection

## 📁 File Structure

```
krogerdemo1/
├── app.py                    # Main Streamlit application
├── azure_services.py        # Azure OpenAI and Storage integration
├── data_processor.py        # Data preprocessing and statistical analysis
├── generate_sample_data.py  # Sample data generation utility
├── requirements.txt         # Python dependencies
├── .env                     # Environment configuration
└── README.md               # This file
```

## 🔧 Configuration

### Azure OpenAI Setup
1. Create an Azure OpenAI resource
2. Deploy the GPT-4o model
3. Update the `.env` file with your endpoint and API key

### Azure Storage Setup (Optional)
1. Create an Azure Storage account
2. Get the connection string
3. Update the `.env` file with your connection string

## 🎯 Retail-Specific Features

### Data Types Supported
- **Sales Transactions**: Revenue, quantity, pricing data
- **Inventory Records**: Stock levels, movement patterns  
- **Customer Behavior**: Purchase patterns, frequency analysis
- **Product Performance**: Sales trends, category analysis

### Contextual Understanding
The AI model is specifically prompted to understand:
- Seasonal retail patterns
- Promotional impacts
- Business cycle variations
- Industry-specific anomalies

## 📈 Analytics & Feedback

### Session Statistics
- Track approval/rejection rates
- Monitor analysis accuracy
- View feedback history

### Continuous Improvement
- User feedback is stored in Azure Storage
- Feedback data can be used to retrain models
- Adaptive thresholds based on human validation

## 🚨 Common Anomalies Examples

1. **Price Anomalies**: Items sold at 10x normal price (data entry error)
2. **Quantity Anomalies**: Bulk purchases of 100+ items
3. **Discount Anomalies**: Discounts over 80% (potential fraud)
4. **Temporal Anomalies**: Transactions at 3 AM (security issue)
5. **Customer Anomalies**: Single customer making 10 high-value purchases in one day

## 🔒 Security & Best Practices

- Environment variables for sensitive configuration
- Azure Managed Identity support (recommended for production)
- Proper error handling and logging
- Data validation and sanitization
- Secure Azure Storage integration

## 🐛 Troubleshooting

### Common Issues

1. **Azure OpenAI Connection Failed**
   - Verify your API key and endpoint
   - Check the deployment name matches your Azure setup
   - Ensure you have sufficient quota

2. **File Upload Issues**
   - Ensure file format is supported (CSV, Excel, JSON)
   - Check file size limits
   - Verify data structure and encoding

3. **No Anomalies Detected**
   - Try adjusting contamination parameters
   - Ensure dataset has numeric columns
   - Check data quality and preprocessing results

### Logging
The application includes comprehensive logging. Check the console output for detailed error messages and processing information.

## 🤝 Contributing

This is a demonstration solution. For production use, consider:
- Enhanced error handling
- Authentication and authorization
- Scalability improvements
- Advanced ML model integration
- Real-time streaming data support

## 📞 Support

For issues or questions related to Azure services:
- Azure OpenAI documentation
- Azure Storage documentation
- Azure Support portal

---

**Built with Azure AI Services for Intelligent Retail Analytics** 🛒✨
