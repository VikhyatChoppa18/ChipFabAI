"""
ChipFabAI Data Processor
Batch processing service for historical manufacturing data analysis
"""

import os
import json
import logging
from typing import List, Dict
import pandas as pd
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_historical_data(input_file: str, output_file: str):
    """
    Process historical semiconductor manufacturing data
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output JSON file
    """
    try:
        logger.info(f"Processing historical data from {input_file}")
        
        # Load data
        df = pd.read_csv(input_file)
        
        # Data cleaning and preprocessing
        df_clean = clean_data(df)
        
        # Feature engineering
        df_features = engineer_features(df_clean)
        
        # Calculate statistics
        stats = calculate_statistics(df_features)
        
        # Generate insights
        insights = generate_insights(df_features, stats)
        
        # Save results - convert timestamps to strings for JSON serialization
        results = {
            "processed_at": datetime.now().isoformat(),
            "records_processed": len(df),
            "statistics": stats,
            "insights": insights,
            "features": df_features.to_dict(orient="records")
        }
        
        # Convert any datetime objects to strings for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert datetime objects to strings"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'dtype') and 'datetime' in str(obj.dtype):
                return obj.astype(str).tolist()
            else:
                return obj
        
        results = convert_to_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Processed {len(df)} records, saved to {output_file}")
        return results
        
    except Exception as e:
        logger.error(f"Data processing error: {str(e)}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate data"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Removing rows with null values in columns essential for analysis
    # These columns are required for accurate process analysis
    critical_columns = ['temperature', 'pressure', 'etch_time', 'yield']
    df = df.dropna(subset=critical_columns)
    
    # Validate ranges
    df = df[(df['temperature'] >= 0) & (df['temperature'] <= 500)]
    df = df[(df['pressure'] >= 0) & (df['pressure'] <= 10)]
    df = df[(df['yield'] >= 0) & (df['yield'] <= 100)]
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features from raw process parameters for ML model training
    These derived features capture relationships between process parameters and yield
    based on semiconductor manufacturing domain knowledge
    """
    # Creating derived features that capture physical relationships in manufacturing
    # temp_pressure_ratio captures the relationship between temperature and pressure
    df['temp_pressure_ratio'] = df['temperature'] / (df['pressure'] + 1e-6)
    # process_efficiency measures yield per unit time
    df['process_efficiency'] = df['yield'] / df['etch_time']
    # normalized_gas_flow normalizes gas flow by chamber pressure
    df['normalized_gas_flow'] = df['gas_flow'] / df['chamber_pressure']
    
    # Time-based features
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    return df


def calculate_statistics(df: pd.DataFrame) -> Dict:
    """Calculate statistical summaries"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    stats = {
        "mean": df[numeric_columns].mean().to_dict(),
        "median": df[numeric_columns].median().to_dict(),
        "std": df[numeric_columns].std().to_dict(),
        "min": df[numeric_columns].min().to_dict(),
        "max": df[numeric_columns].max().to_dict(),
        "correlation_with_yield": df[numeric_columns].corr()['yield'].to_dict() if 'yield' in df.columns else {}
    }
    
    return stats


def generate_insights(df: pd.DataFrame, stats: Dict) -> List[str]:
    """Generate insights from processed data"""
    insights = []
    
    # Yield analysis
    if 'yield' in df.columns:
        avg_yield = df['yield'].mean()
        if avg_yield < 85:
            insights.append(f"Average yield ({avg_yield:.2f}%) is below optimal threshold (85%)")
        else:
            insights.append(f"Average yield ({avg_yield:.2f}%) is within optimal range")
    
    # Parameter correlations
    if 'yield' in stats.get('correlation_with_yield', {}):
        correlations = stats['correlation_with_yield']
        top_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for param, corr in top_correlations:
            if param != 'yield':
                insights.append(f"{param} shows strong correlation ({corr:.2f}) with yield")
    
    # Process stability
    if 'temperature' in df.columns:
        temp_std = df['temperature'].std()
        if temp_std > 10:
            insights.append(f"Temperature variation is high (std={temp_std:.2f}°C), affecting process stability")
    
    return insights


if __name__ == "__main__":
    # Cloud Run Job entry point
    input_file = os.getenv("INPUT_FILE", "/tmp/input.csv")
    output_file = os.getenv("OUTPUT_FILE", "/tmp/output.json")
    
    logger.info("Starting data processing job...")
    results = process_historical_data(input_file, output_file)
    logger.info("Data processing completed successfully")

