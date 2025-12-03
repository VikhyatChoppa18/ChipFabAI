"""
Data Processor - batch processing service for analyzing historical manufacturing data
This runs as a Cloud Run Job - you give it a CSV file, it processes it and outputs insights
I use this for analyzing large datasets offline without blocking the real-time prediction service
"""

import os
import json
import logging
from typing import List, Dict
import pandas as pd
from datetime import datetime
import numpy as np

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_historical_data(input_file: str, output_file: str):
    """
    Main processing function - takes a CSV file and produces insights
    I clean the data, engineer features, calculate stats, and generate insights
    The output is JSON so it's easy to consume from other services
    """
    try:
        logger.info(f"Processing historical data from {input_file}")
        
        # Load the CSV file
        df = pd.read_csv(input_file)
        
        # Clean and validate the data
        df_clean = clean_data(df)
        
        # Create derived features that might be useful for analysis
        df_features = engineer_features(df_clean)
        
        # Calculate summary statistics
        stats = calculate_statistics(df_features)
        
        # Generate human-readable insights from the data
        insights = generate_insights(df_features, stats)
        
        # Build the output structure
        results = {
            "processed_at": datetime.now().isoformat(),
            "records_processed": len(df),
            "statistics": stats,
            "insights": insights,
            "features": df_features.to_dict(orient="records")
        }
        
        # JSON can't serialize datetime objects, so I convert them to strings
        def convert_to_serializable(obj):
            """Recursively convert datetime objects to ISO format strings"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'dtype') and 'datetime' in str(obj.dtype):
                # Pandas datetime series
                return obj.astype(str).tolist()
            else:
                return obj
        
        results = convert_to_serializable(results)
        
        # Write the results to a JSON file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Processed {len(df)} records, saved to {output_file}")
        return results
        
    except Exception as e:
        logger.error(f"Data processing error: {str(e)}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the input data
    I remove duplicates, drop rows with missing critical fields, and filter out invalid values
    This ensures the analysis is based on good quality data
    """
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Drop rows where critical columns are missing
    # Can't analyze a process if we don't know the temperature, pressure, etc.
    critical_columns = ['temperature', 'pressure', 'etch_time', 'yield']
    df = df.dropna(subset=critical_columns)
    
    # Filter out values that are clearly invalid
    # Temperature should be 0-500°C, pressure 0-10 Torr, yield 0-100%
    df = df[(df['temperature'] >= 0) & (df['temperature'] <= 500)]
    df = df[(df['pressure'] >= 0) & (df['pressure'] <= 10)]
    df = df[(df['yield'] >= 0) & (df['yield'] <= 100)]
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from the raw process parameters
    These features capture relationships that might be useful for analysis
    For example, temp/pressure ratio can be more informative than either alone
    """
    # Create ratio features that capture relationships between parameters
    # I add a tiny epsilon to avoid division by zero
    df['temp_pressure_ratio'] = df['temperature'] / (df['pressure'] + 1e-6)
    # Efficiency metric - how much yield per second of processing time
    df['process_efficiency'] = df['yield'] / df['etch_time']
    # Normalize gas flow by chamber pressure - gives a better sense of actual flow rate
    df['normalized_gas_flow'] = df['gas_flow'] / df['chamber_pressure']
    
    # Extract time-based features if we have timestamps
    # Hour of day and day of week can reveal patterns (e.g., lower yield on night shifts)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    return df


def calculate_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics for all numeric columns
    I also compute correlations with yield - this tells us which parameters
    are most strongly related to good/bad outcomes
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    stats = {
        "mean": df[numeric_columns].mean().to_dict(),
        "median": df[numeric_columns].median().to_dict(),
        "std": df[numeric_columns].std().to_dict(),
        "min": df[numeric_columns].min().to_dict(),
        "max": df[numeric_columns].max().to_dict(),
        # Correlation with yield is super useful - shows which params matter most
        "correlation_with_yield": df[numeric_columns].corr()['yield'].to_dict() if 'yield' in df.columns else {}
    }
    
    return stats


def generate_insights(df: pd.DataFrame, stats: Dict) -> List[str]:
    """
    Generate human-readable insights from the processed data
    I look at yield trends, parameter correlations, and process stability
    These insights help engineers understand what's happening in their fab
    """
    insights = []
    
    # Check overall yield performance
    if 'yield' in df.columns:
        avg_yield = df['yield'].mean()
        if avg_yield < 85:
            insights.append(f"Average yield ({avg_yield:.2f}%) is below optimal threshold (85%)")
        else:
            insights.append(f"Average yield ({avg_yield:.2f}%) is within optimal range")
    
    # Find which parameters correlate most strongly with yield
    # High correlation means that parameter has a big impact on outcomes
    if 'yield' in stats.get('correlation_with_yield', {}):
        correlations = stats['correlation_with_yield']
        # Get top 3 correlations (positive or negative)
        top_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for param, corr in top_correlations:
            if param != 'yield':
                insights.append(f"{param} shows strong correlation ({corr:.2f}) with yield")
    
    # Check process stability - high variation in temperature suggests control issues
    if 'temperature' in df.columns:
        temp_std = df['temperature'].std()
        if temp_std > 10:
            insights.append(f"Temperature variation is high (std={temp_std:.2f}°C), affecting process stability")
    
    return insights


if __name__ == "__main__":
    """
    Entry point for Cloud Run Job
    Reads input/output file paths from environment variables
    This makes it easy to configure without changing code
    """
    input_file = os.getenv("INPUT_FILE", "/tmp/input.csv")
    output_file = os.getenv("OUTPUT_FILE", "/tmp/output.json")
    
    logger.info("Starting data processing job...")
    results = process_historical_data(input_file, output_file)
    logger.info("Data processing completed successfully")

