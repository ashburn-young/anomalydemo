"""
Azure Services Integration Module for Retail Anomaly Detection
Handles Azure OpenAI and Azure Storage operations with proper error handling and security.
"""

import os
import logging
import sys
from typing import Dict, List, Any, Optional
from azure.storage.blob import BlobServiceClient
import json
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureOpenAIService:
    """Azure OpenAI service for anomaly detection analysis"""
    
    def __init__(self):
        """Initialize Azure OpenAI client with proper error handling"""
        try:
            # Get environment variables
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            
            # DEBUG: Print and flush environment variable values
            print('DEBUG: AZURE_OPENAI_API_KEY:', api_key[:8] if api_key else 'None', flush=True)
            print('DEBUG: AZURE_OPENAI_API_VERSION:', api_version, flush=True)
            print('DEBUG: AZURE_OPENAI_ENDPOINT:', azure_endpoint, flush=True)
            # Also log to a file for debugging
            with open('debug_env.log', 'a') as f:
                f.write(f"API_KEY: {api_key[:8] if api_key else 'None'}\n")
                f.write(f"API_VERSION: {api_version}\n")
                f.write(f"ENDPOINT: {azure_endpoint}\n")
            
            # Validate required environment variables
            if not api_key:
                logger.warning("AZURE_OPENAI_API_KEY environment variable not found")
                self.client = None
                self.deployment_name = None
                return
                
            if not azure_endpoint:
                logger.warning("AZURE_OPENAI_ENDPOINT environment variable not found")
                self.client = None
                self.deployment_name = None
                return
            
            # Clean up the endpoint URL to ensure it's properly formatted
            if not azure_endpoint.startswith('https://'):
                azure_endpoint = f"https://{azure_endpoint}"
            if not azure_endpoint.endswith('/'):
                azure_endpoint = f"{azure_endpoint}/"
                
            # Log OpenAI version
            with open('debug_env.log', 'a') as f:
                f.write(f'OPENAI_VERSION: {openai.__version__}\n')
                
            # Force classic Azure OpenAI setup regardless of the OpenAI version
            try:
                # Configure OpenAI with Azure credentials
                openai.api_type = "azure"
                openai.api_base = azure_endpoint.rstrip("/")
                openai.api_key = api_key
                openai.api_version = api_version
                
                # Store the openai module as our client
                self.client = openai
                
                # Add compatibility layer for different API versions
                # For newer versions of OpenAI SDK that don't have ChatCompletion
                if not hasattr(openai, 'ChatCompletion'):
                    print('DEBUG: Adding ChatCompletion compatibility layer', flush=True)
                    with open('debug_env.log', 'a') as f:
                        f.write('ADDING_COMPATIBILITY: Creating ChatCompletion class\n')
                    
                    # Create a compatibility class that uses the new API style but works with old code
                    class ChatCompletionCompat:
                        @staticmethod
                        def create(**kwargs):
                            # Convert parameters for new API if needed
                            if 'request_timeout' in kwargs:
                                kwargs['timeout'] = kwargs.pop('request_timeout')
                            if hasattr(openai, 'chat') and hasattr(openai.chat, 'completions'):
                                return openai.chat.completions.create(**kwargs)
                            else:
                                raise ValueError("Neither ChatCompletion nor chat.completions API available")
                    
                    # Add the compatibility class to the openai module
                    openai.ChatCompletion = ChatCompletionCompat
                    
                print('DEBUG: OpenAI client configured with Azure credentials', flush=True)
            except Exception as e:
                print(f'DEBUG: AzureOpenAI client creation failed: {e}', flush=True)
                with open('debug_env.log', 'a') as f:
                    f.write(f'CLIENT_CREATION_ERROR: {e}\n')
                self.client = None

            # Set deployment name for Azure OpenAI
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
            print('DEBUG: AZURE_OPENAI_DEPLOYMENT_NAME:', deployment_name, flush=True)
            with open('debug_env.log', 'a') as f:
                f.write(f"DEPLOYMENT_NAME: {deployment_name}\n")
            self.deployment_name = deployment_name
            
            # Test the connection with a simple call to surface errors immediately
            try:
                # Use the client we've configured for a simple API call
                if self.client and hasattr(self.client, 'ChatCompletion'):
                    test_response = self.client.ChatCompletion.create(
                        engine=self.deployment_name,
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5
                    )
                    print('DEBUG: Test API call succeeded', flush=True)
                    with open('debug_env.log', 'a') as f:
                        f.write('TEST_API_CALL: success\n')
                else:
                    error_msg = "OpenAI client not properly configured - ChatCompletion not available"
                    print(f'DEBUG: {error_msg}', flush=True)
                    with open('debug_env.log', 'a') as f:
                        f.write(f'TEST_API_CALL_ERROR: {error_msg}\n')
                    self.client = None  # Reset client as it's not usable
            except Exception as e:
                print(f'DEBUG: Test API call failed: {e}', flush=True)
                with open('debug_env.log', 'a') as f:
                    f.write(f'TEST_API_CALL_ERROR: {e}\n')
                self.client = None
                self.deployment_name = None
            
            logger.info("Azure OpenAI client initialized successfully")
            
            # Final debug log for client status
            print(f'DEBUG: AzureOpenAIService __init__ complete. client is {"set" if self.client else "None"}', flush=True)
            with open('debug_env.log', 'a') as f:
                f.write(f'CLIENT_FINAL: {"set" if self.client else "None"}\n')
                # Log if ChatCompletion is available
                if self.client:
                    f.write(f'HAS_CHAT_COMPLETION: {hasattr(self.client, "ChatCompletion")}\n')
                    f.write(f'CLIENT_TYPE: {type(self.client).__name__}\n')
        
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            # Record the error for debugging
            with open('debug_env.log', 'a') as f:
                f.write(f'INIT_ERROR: {str(e)}\n')
            # Don't raise the exception, allow the app to continue without AI analysis
            self.client = None
            self.deployment_name = None
    
    def analyze_retail_data_for_anomalies(self, data_summary: str, data_sample: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze retail data for anomalies using Azure OpenAI
        
        Args:
            data_summary: Summary statistics of the dataset
            data_sample: Sample of the actual data for context
            
        Returns:
            Dictionary containing anomaly analysis results
        """
        # Check if client is available
        if not self.client:
            logger.warning("Azure OpenAI client not available, returning mock analysis")
            return {
                "anomalies_detected": [
                    {
                        "anomaly_type": "Service Unavailable",
                        "description": "Azure OpenAI service is not available. Please check your configuration and network connectivity.",
                        "confidence_score": 0.0,
                        "potential_causes": ["Service configuration issue", "Network connectivity problem"],
                        "affected_records": 0,
                        "severity": "low"
                    }
                ],
                "summary": "Azure OpenAI analysis unavailable - please check service configuration",
                "recommendations": ["Verify Azure OpenAI credentials", "Check network connectivity", "Use statistical analysis instead"]
            }
        
        try:
            # Create a comprehensive prompt for retail anomaly detection
            prompt = self._create_anomaly_detection_prompt(data_summary, data_sample)
            
            # Make sure the client has ChatCompletion available
            if not hasattr(self.client, 'ChatCompletion'):
                error_msg = "ChatCompletion API not available - cannot perform analysis"
                logger.error(error_msg)
                with open('debug_env.log', 'a') as f:
                    f.write(f'ANALYZE_ERROR: {error_msg}\n')
                return {
                    "anomalies_detected": [],
                    "summary": error_msg,
                    "recommendations": ["Check OpenAI version and API compatibility"]
                }
                
            try:
                # Use the client's ChatCompletion interface for API call
                response = self.client.ChatCompletion.create(
                    engine=self.deployment_name,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a retail data anomaly detection assistant. Your job is to analyze uploaded datasets and identify unusual patterns in sales or inventory data. You highlight transactions or stock movements that deviate from expected trends. 
                            
                            Provide your response as a structured JSON with the following format:
                            {
                                "anomalies_detected": [
                                    {
                                        "anomaly_type": "string",
                                        "description": "string",
                                        "confidence_score": float,
                                        "potential_causes": ["cause1", "cause2"],
                                        "affected_records": int,
                                        "severity": "low|medium|high"
                                    }
                                ],
                                "summary": "string",
                                "recommendations": ["rec1", "rec2"]
                            }"""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=2000,
                    request_timeout=30
                )
                
                # Log successful API call
                with open('debug_env.log', 'a') as f:
                    f.write('ANALYZE_API_CALL: success\n')
                
                # Parse the response - OpenAI 0.28.1 specific format
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                        content = response.choices[0].message.content
                    else:
                        # Fallback to dictionary access
                        content = response.choices[0].get('message', {}).get('content', str(response))
                else:
                    content = str(response)
                
                result = self._parse_openai_response(content)
                logger.info(f"Successfully analyzed data, found {len(result.get('anomalies_detected', []))} potential anomalies")
                return result
                
            except Exception as e:
                print(f"DEBUG: OpenAI API call failed: {e}", flush=True)
                with open('debug_env.log', 'a') as f:
                    f.write(f"OPENAI_API_ERROR: {e}\n")
                return {
                    "anomalies_detected": [],
                    "summary": f"OpenAI API call failed: {e}",
                    "recommendations": ["Check deployment name, model, API version, and Azure OpenAI resource configuration."]
                }
            
        except Exception as e:
            logger.error(f"Error analyzing data with Azure OpenAI: {str(e)}")
            return {
                "anomalies_detected": [
                    {
                        "anomaly_type": "Analysis Error",
                        "description": f"Error occurred during AI analysis: {str(e)}",
                        "confidence_score": 0.0,
                        "potential_causes": ["Service timeout", "API error", "Network issue"],
                        "affected_records": 0,
                        "severity": "low"
                    }
                ],
                "summary": f"AI analysis failed: {str(e)}",
                "recommendations": ["Try again later", "Check service status", "Use statistical analysis"]
            }
    
    def _create_anomaly_detection_prompt(self, data_summary: str, data_sample: pd.DataFrame) -> str:
        """Create a comprehensive prompt for anomaly detection"""
        sample_data = data_sample.head(10).to_string()
        
        prompt = f"""
        Analyze this retail dataset for anomalies and unusual patterns:

        Dataset Summary:
        {data_summary}

        Sample Data (first 10 rows):
        {sample_data}

        Please identify:
        1. Sales transactions that deviate significantly from normal patterns
        2. Unusual inventory movements or stock levels
        3. Suspicious pricing or discount patterns
        4. Abnormal customer behavior indicators
        5. Data quality issues (missing values, inconsistent formats, outliers)

        For each anomaly, provide:
        - Type of anomaly
        - Detailed description
        - Confidence score (0.0 to 1.0)
        - Potential causes
        - Number of affected records
        - Severity level

        Focus on retail-specific patterns like seasonal trends, promotional impacts, and typical business cycles.
        """
        return prompt
    
    def _parse_openai_response(self, response_content: str) -> Dict[str, Any]:
        """Parse OpenAI response, handling both JSON and text formats"""
        if not response_content:
            logger.warning("Empty response content from OpenAI")
            return {
                "anomalies_detected": [
                    {
                        "anomaly_type": "Empty Response",
                        "description": "The AI model returned an empty response",
                        "confidence_score": 0.0,
                        "potential_causes": ["API timeout", "Model error", "Token limit reached"],
                        "affected_records": 0,
                        "severity": "low"
                    }
                ],
                "summary": "AI analysis returned empty response",
                "recommendations": ["Try again with a smaller dataset", "Check API configuration"]
            }
            
        try:
            # Try to parse as JSON first - look for the first { character to handle any prefixes
            json_start = response_content.find('{')
            if json_start >= 0:
                try:
                    return json.loads(response_content[json_start:])
                except json.JSONDecodeError:
                    # If the entire string isn't valid JSON, try to extract JSON from it
                    import re
                    json_pattern = r'({.*})'
                    match = re.search(json_pattern, response_content, re.DOTALL)
                    if match:
                        try:
                            return json.loads(match.group(1))
                        except json.JSONDecodeError:
                            pass
            
            # If not JSON, create a structured response from text
            return {
                "anomalies_detected": [
                    {
                        "anomaly_type": "General Analysis",
                        "description": response_content,
                        "confidence_score": 0.7,
                        "potential_causes": ["Various factors identified in analysis"],
                        "affected_records": 0,
                        "severity": "medium"
                    }
                ],
                "summary": "Analysis completed - see detailed description",
                "recommendations": ["Review the detailed analysis and validate findings"]
            }
        except Exception as e:
            logger.warning(f"Error parsing OpenAI response: {str(e)}")
            return {
                "anomalies_detected": [
                    {
                        "anomaly_type": "Analysis Result",
                        "description": response_content[:500] + "..." if len(response_content) > 500 else response_content,
                        "confidence_score": 0.5,
                        "potential_causes": ["Analysis provided in text format"],
                        "affected_records": 0,
                        "severity": "medium"
                    }
                ],
                "summary": f"Text-based analysis (parsing error: {str(e)})",
                "recommendations": ["Review the analysis output"]
            }

class AzureStorageService:
    """Azure Blob Storage service for data persistence"""
    
    def __init__(self):
        """Initialize Azure Storage client"""
        try:
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                raise ValueError("Azure Storage connection string not found")
                
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            self.container_name = "retail-anomaly-data"
            self._ensure_container_exists()
            logger.info("Azure Storage client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Storage client: {str(e)}")
            self.blob_service_client = None
    
    def _ensure_container_exists(self):
        """Ensure the storage container exists"""
        try:
            self.blob_service_client.create_container(self.container_name)
        except Exception:
            # Container might already exist
            pass
    
    def upload_processed_data(self, data: pd.DataFrame, filename: str) -> bool:
        """Upload processed data to Azure Storage"""
        if not self.blob_service_client:
            logger.warning("Azure Storage not available, skipping upload")
            return False
            
        try:
            # Convert DataFrame to CSV string
            csv_data = data.to_csv(index=False)
            
            # Upload to blob storage
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=f"processed/{filename}"
            )
            blob_client.upload_blob(csv_data, overwrite=True)
            
            logger.info(f"Successfully uploaded {filename} to Azure Storage")
            return True
        except Exception as e:
            logger.error(f"Failed to upload data to Azure Storage: {str(e)}")
            return False
    
    def save_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Save user feedback to Azure Storage for model improvement"""
        if not self.blob_service_client:
            logger.warning("Azure Storage not available, skipping feedback save")
            return False
            
        try:
            feedback_json = json.dumps(feedback_data, indent=2)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feedback/feedback_{timestamp}.json"
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=filename
            )
            blob_client.upload_blob(feedback_json, overwrite=True)
            
            logger.info(f"Successfully saved feedback to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save feedback: {str(e)}")
            return False