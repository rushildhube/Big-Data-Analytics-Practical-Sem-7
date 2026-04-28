"""
Comprehensive EDA for Capital Bikeshare Time Series Data
Dataset: Capital Bikeshare Trip Data (March 2026)
Purpose: Exploratory Data Analysis for Time Series Analysis
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Fix Unicode encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================================================
# 1. DATA LOADING AND BASIC INFO
# ============================================================================

def load_data(filepath):
    """Load and display basic dataset information"""
    print("\n" + "="*80)
    print("1. DATA LOADING AND BASIC INFORMATION")
    print("="*80)
    
    df = pd.read_csv(filepath)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Total Records: {df.shape[0]:,}")
    print(f"Total Features: {df.shape[1]}")
    
    print("\n--- Column Names and Data Types ---")
    print(df.dtypes)
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Basic Statistics ---")
    print(df.describe())
    
    return df


# ============================================================================
# 2. DATETIME CONVERSION AND TIME SERIES PREPARATION
# ============================================================================

def prepare_time_series(df):
    """Convert datetime columns and extract time features"""
    print("\n" + "="*80)
    print("2. DATETIME CONVERSION AND TIME SERIES PREPARATION")
    print("="*80)
    
    # Convert to datetime
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    
    # Calculate ride duration in minutes
    df['ride_duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    
    # Extract time features for time series analysis
    df['year'] = df['started_at'].dt.year
    df['month'] = df['started_at'].dt.month
    df['day'] = df['started_at'].dt.day
    df['hour'] = df['started_at'].dt.hour
    df['dayofweek'] = df['started_at'].dt.dayofweek  # Monday=0, Sunday=6
    df['dayname'] = df['started_at'].dt.day_name()
    df['week'] = df['started_at'].dt.isocalendar().week
    df['quarter'] = df['started_at'].dt.quarter
    df['date'] = df['started_at'].dt.date
    
    print("\n✓ DateTime columns converted")
    print(f"  - started_at: {df['started_at'].dtype}")
    print(f"  - ended_at: {df['ended_at'].dtype}")
    
    print("\n--- Time Range of Data ---")
    print(f"Start Date: {df['started_at'].min()}")
    print(f"End Date: {df['started_at'].max()}")
    print(f"Duration: {(df['started_at'].max() - df['started_at'].min()).days} days")
    
    print("\n--- Ride Duration Statistics (in minutes) ---")
    print(f"Mean: {df['ride_duration_min'].mean():.2f} min")
    print(f"Median: {df['ride_duration_min'].median():.2f} min")
    print(f"Std Dev: {df['ride_duration_min'].std():.2f} min")
    print(f"Min: {df['ride_duration_min'].min():.2f} min")
    print(f"Max: {df['ride_duration_min'].max():.2f} min")
    
    return df


# ============================================================================
# 3. MISSING VALUES AND DATA QUALITY
# ============================================================================

def check_missing_values(df):
    """Analyze missing values and data quality"""
    print("\n" + "="*80)
    print("3. MISSING VALUES AND DATA QUALITY ANALYSIS")
    print("="*80)
    
    missing_data = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    print("\n--- Missing Values ---")
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_data.values,
        'Percentage': missing_percent.values
    }).sort_values('Missing_Count', ascending=False)
    
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    if missing_df['Missing_Count'].sum() == 0:
        print("✓ No missing values detected!")
    
    print("\n--- Duplicate Records ---")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    print("\n--- Ride ID Uniqueness ---")
    print(f"Total records: {len(df)}")
    print(f"Unique ride_ids: {df['ride_id'].nunique()}")
    print(f"Duplicate ride_ids: {len(df) - df['ride_id'].nunique()}")
    
    # Check for invalid ride durations
    invalid_rides = (df['ride_duration_min'] < 0).sum()
    print(f"\n--- Invalid Ride Durations (negative) ---")
    print(f"Count: {invalid_rides}")
    
    return missing_df


# ============================================================================
# 4. UNIVARIATE ANALYSIS
# ============================================================================

def univariate_analysis(df):
    """Analyze individual variables"""
    print("\n" + "="*80)
    print("4. UNIVARIATE ANALYSIS")
    print("="*80)
    
    print("\n--- Rideable Type Distribution ---")
    print(df['rideable_type'].value_counts())
    print(f"\nPercentage:")
    print(df['rideable_type'].value_counts(normalize=True) * 100)
    
    print("\n--- User Type Distribution (Member vs Casual) ---")
    print(df['member_casual'].value_counts())
    print(f"\nPercentage:")
    print(df['member_casual'].value_counts(normalize=True) * 100)
    
    print("\n--- Start Station Analysis ---")
    print(f"Total unique start stations: {df['start_station_name'].nunique()}")
    print(f"Missing start station names: {df['start_station_name'].isnull().sum()}")
    print("\nTop 10 Start Stations:")
    print(df['start_station_name'].value_counts().head(10))
    
    print("\n--- End Station Analysis ---")
    print(f"Total unique end stations: {df['end_station_name'].nunique()}")
    print(f"Missing end station names: {df['end_station_name'].isnull().sum()}")
    print("\nTop 10 End Stations:")
    print(df['end_station_name'].value_counts().head(10))
    
    print("\n--- Geographical Distribution ---")
    print(f"Start Latitude - Min: {df['start_lat'].min()}, Max: {df['start_lat'].max()}")
    print(f"Start Longitude - Min: {df['start_lng'].min()}, Max: {df['start_lng'].max()}")
    print(f"End Latitude - Min: {df['end_lat'].min()}, Max: {df['end_lat'].max()}")
    print(f"End Longitude - Min: {df['end_lng'].min()}, Max: {df['end_lng'].max()}")


# ============================================================================
# 5. TIME SERIES TEMPORAL PATTERNS
# ============================================================================

def temporal_patterns(df):
    """Analyze temporal patterns for time series"""
    print("\n" + "="*80)
    print("5. TEMPORAL PATTERNS ANALYSIS")
    print("="*80)
    
    print("\n--- Daily Distribution ---")
    daily_rides = df.groupby('date').size()
    print(f"Mean daily rides: {daily_rides.mean():.2f}")
    print(f"Median daily rides: {daily_rides.median():.2f}")
    print(f"Std Dev: {daily_rides.std():.2f}")
    print(f"Min daily rides: {daily_rides.min()}")
    print(f"Max daily rides: {daily_rides.max()}")
    
    print("\n--- Hourly Distribution ---")
    hourly_rides = df.groupby('hour').size()
    print("Rides per hour:")
    print(hourly_rides)
    
    print("\n--- Day of Week Distribution ---")
    dow_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
               4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    dow_rides = df.groupby('dayofweek').size()
    for day, count in dow_rides.items():
        print(f"{dow_map[day]}: {count:,} rides")
    
    print("\n--- Weekly Trend ---")
    weekly_rides = df.groupby('week').size()
    print(f"Mean weekly rides: {weekly_rides.mean():.2f}")
    print(f"Rides per week (first 5):")
    print(weekly_rides.head())
    
    print("\n--- Peak Hours ---")
    peak_hour = hourly_rides.idxmax()
    low_hour = hourly_rides.idxmin()
    print(f"Peak hour: {peak_hour}:00 with {hourly_rides[peak_hour]:,} rides")
    print(f"Lowest hour: {low_hour}:00 with {hourly_rides[low_hour]:,} rides")


# ============================================================================
# 6. USER BEHAVIOR PATTERNS
# ============================================================================

def user_behavior_analysis(df):
    """Analyze user behavior patterns"""
    print("\n" + "="*80)
    print("6. USER BEHAVIOR ANALYSIS")
    print("="*80)
    
    print("\n--- Member vs Casual Ride Duration ---")
    user_duration = df.groupby('member_casual')['ride_duration_min'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ])
    print(user_duration)
    
    print("\n--- Member vs Casual by Rideable Type ---")
    user_bike = pd.crosstab(df['member_casual'], df['rideable_type'])
    print(user_bike)
    
    print("\n--- Hourly Patterns by User Type ---")
    hourly_user = df.groupby(['hour', 'member_casual']).size().unstack(fill_value=0)
    print("Top 3 peak hours by user type:")
    print(hourly_user.head(3))
    
    print("\n--- Day of Week Patterns by User Type ---")
    dow_user = df.groupby(['dayname', 'member_casual']).size().unstack(fill_value=0)
    print(dow_user)


# ============================================================================
# 7. BIKE TYPE ANALYSIS
# ============================================================================

def bike_type_analysis(df):
    """Analyze bike type usage patterns"""
    print("\n" + "="*80)
    print("7. BIKE TYPE ANALYSIS")
    print("="*80)
    
    print("\n--- Bike Type Distribution ---")
    bike_dist = df['rideable_type'].value_counts()
    print(bike_dist)
    
    print("\n--- Average Ride Duration by Bike Type ---")
    bike_duration = df.groupby('rideable_type')['ride_duration_min'].agg([
        'count', 'mean', 'median', 'std'
    ])
    print(bike_duration)
    
    print("\n--- Hourly Usage by Bike Type ---")
    hourly_bike = df.groupby(['hour', 'rideable_type']).size().unstack(fill_value=0)
    print(hourly_bike.head(10))


# ============================================================================
# 8. OUTLIER DETECTION
# ============================================================================

def outlier_detection(df):
    """Identify potential outliers"""
    print("\n" + "="*80)
    print("8. OUTLIER DETECTION")
    print("="*80)
    
    # Using IQR method for ride duration
    Q1 = df['ride_duration_min'].quantile(0.25)
    Q3 = df['ride_duration_min'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print("\n--- Ride Duration Outliers (IQR Method) ---")
    print(f"Q1 (25th percentile): {Q1:.2f} min")
    print(f"Q3 (75th percentile): {Q3:.2f} min")
    print(f"IQR: {IQR:.2f} min")
    print(f"Lower Bound: {lower_bound:.2f} min")
    print(f"Upper Bound: {upper_bound:.2f} min")
    
    outliers = df[(df['ride_duration_min'] < lower_bound) | (df['ride_duration_min'] > upper_bound)]
    print(f"\nOutlier records: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    print("\n--- Extreme Values ---")
    print(f"Rides > 2 hours: {(df['ride_duration_min'] > 120).sum()}")
    print(f"Rides > 3 hours: {(df['ride_duration_min'] > 180).sum()}")
    print(f"Rides < 1 minute: {(df['ride_duration_min'] < 1).sum()}")
    
    # Geographical outliers
    print("\n--- Geographical Outliers (extreme coordinates) ---")
    start_lat_outliers = df[(df['start_lat'] < 38.5) | (df['start_lat'] > 39.0)]
    print(f"Records with unusual start latitude: {len(start_lat_outliers)}")


# ============================================================================
# 9. CORRELATIONS AND RELATIONSHIPS
# ============================================================================

def correlation_analysis(df):
    """Analyze correlations between numerical variables"""
    print("\n" + "="*80)
    print("9. CORRELATION AND RELATIONSHIP ANALYSIS")
    print("="*80)
    
    # Select numerical columns
    numerical_df = df[['ride_duration_min', 'hour', 'dayofweek', 'start_lat', 
                       'start_lng', 'end_lat', 'end_lng']].copy()
    
    print("\n--- Correlation Matrix (Numerical Variables) ---")
    correlation = numerical_df.corr()
    print(correlation)
    
    print("\n--- Distance Analysis ---")
    # Simple Euclidean distance calculation
    df['distance'] = np.sqrt((df['end_lat'] - df['start_lat'])**2 + 
                             (df['end_lng'] - df['start_lng'])**2)
    
    print(f"Average distance (lat/long degrees): {df['distance'].mean():.4f}")
    print(f"Distance std dev: {df['distance'].std():.4f}")
    print(f"Max distance: {df['distance'].max():.4f}")
    
    # Correlation between distance and duration
    distance_duration_corr = df[['distance', 'ride_duration_min']].corr().iloc[0, 1]
    print(f"\nCorrelation between distance and ride duration: {distance_duration_corr:.4f}")


# ============================================================================
# 10. SUMMARY STATISTICS FOR TIME SERIES
# ============================================================================

def summary_for_timeseries(df):
    """Generate summary statistics optimized for time series analysis"""
    print("\n" + "="*80)
    print("10. SUMMARY STATISTICS FOR TIME SERIES ANALYSIS")
    print("="*80)
    
    print("\n--- Dataset Overview ---")
    print(f"Total Rides: {len(df):,}")
    print(f"Time Period: {df['started_at'].min()} to {df['started_at'].max()}")
    print(f"Days Covered: {(df['started_at'].max() - df['started_at'].min()).days} days")
    
    print("\n--- Daily Aggregates ---")
    daily_stats = df.groupby('date').agg({
        'ride_id': 'count',
        'ride_duration_min': ['mean', 'std'],
        'distance': 'mean'
    }).round(2)
    daily_stats.columns = ['count', 'avg_duration', 'std_duration', 'avg_distance']
    print("\nFirst 10 days:")
    print(daily_stats.head(10))
    
    print("\n--- Hourly Aggregates (Average across all days) ---")
    hourly_stats = df.groupby('hour').agg({
        'ride_id': 'count',
        'ride_duration_min': ['mean', 'std'],
        'member_casual': lambda x: (x == 'member').sum()
    }).round(2)
    hourly_stats.columns = ['total_rides', 'avg_duration', 'std_duration', 'member_rides']
    print(hourly_stats)
    
    print("\n--- Peak Traffic Times ---")
    hourly_count = df.groupby('hour').size().sort_values(ascending=False)
    print("Top 5 peak hours:")
    for hour, count in hourly_count.head(5).items():
        print(f"  {hour:02d}:00 - {count:,} rides")


# ============================================================================
# 11. VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(df, output_dir=None):
    """Create visualizations for time series analysis"""
    import os
    
    print("\n" + "="*80)
    print("11. CREATING VISUALIZATIONS")
    print("="*80)
    
    # Determine output directory based on current working directory
    if output_dir is None:
        if os.path.basename(os.getcwd()) == 'App':
            output_dir = '../Visualizations'
        else:
            output_dir = './Mini Project/Visualizations'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Hourly trend
    fig, ax = plt.subplots(figsize=(14, 6))
    hourly_rides = df.groupby('hour').size()
    ax.plot(hourly_rides.index, hourly_rides.values, marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Number of Rides', fontsize=12)
    ax.set_title('Hourly Ride Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_hourly_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_hourly_distribution.png")
    
    # Figure 2: Daily trend
    fig, ax = plt.subplots(figsize=(14, 6))
    daily_rides = df.groupby('date').size()
    ax.plot(daily_rides.index, daily_rides.values, linewidth=2, color='#2E86AB')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Rides', fontsize=12)
    ax.set_title('Daily Ride Trend', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_daily_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_daily_trend.png")
    
    # Figure 3: Day of week comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    dow_rides = df.groupby('dayofweek').size()
    days = [dow_map[i] for i in range(7)]
    ax.bar(days, [dow_rides.get(i, 0) for i in range(7)], color='#A23B72')
    ax.set_ylabel('Number of Rides', fontsize=12)
    ax.set_title('Rides by Day of Week', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_day_of_week.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_day_of_week.png")
    
    # Figure 4: Ride duration distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df[df['ride_duration_min'] <= 60]['ride_duration_min'], bins=50, color='#F18F01', edgecolor='black')
    ax.set_xlabel('Ride Duration (minutes)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Ride Duration Distribution (0-60 minutes)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_ride_duration_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_ride_duration_dist.png")
    
    # Figure 5: User type comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    user_type = df['member_casual'].value_counts()
    colors = ['#2E86AB', '#A23B72']
    ax.pie(user_type.values, labels=user_type.index, autopct='%1.1f%%', 
           colors=colors, startangle=90, textprops={'fontsize': 12})
    ax.set_title('Member vs Casual User Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_user_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 05_user_distribution.png")
    
    # Figure 6: Bike type distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    bike_type = df['rideable_type'].value_counts()
    colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(bike_type)]
    ax.pie(bike_type.values, labels=bike_type.index, autopct='%1.1f%%', 
           colors=colors, startangle=90, textprops={'fontsize': 12})
    ax.set_title('Bike Type Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_bike_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 06_bike_type.png")
    
    # Figure 7: Heatmap - Hour vs Day of week
    fig, ax = plt.subplots(figsize=(14, 8))
    dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    heatmap_data = df.groupby(['hour', 'dayofweek']).size().unstack(fill_value=0)
    heatmap_data.columns = [dow_map[i] for i in heatmap_data.columns]
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Number of Rides'})
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Hour of Day', fontsize=12)
    ax.set_title('Ride Frequency Heatmap: Hour vs Day of Week', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_heatmap_hour_dow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 07_heatmap_hour_dow.png")
    
    # Figure 8: Member vs Casual by hour
    fig, ax = plt.subplots(figsize=(14, 6))
    hourly_user = df.groupby(['hour', 'member_casual']).size().unstack(fill_value=0)
    hourly_user.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72'], width=0.8)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Number of Rides', fontsize=12)
    ax.set_title('Hourly Distribution: Member vs Casual', fontsize=14, fontweight='bold')
    ax.legend(title='User Type', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_member_vs_casual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 08_member_vs_casual.png")
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    import os
    
    # Determine the correct path based on current working directory
    if os.path.basename(os.getcwd()) == 'App':
        # Running from App folder
        data_path = '../Dataset/202603-capitalbikeshare-tripdata.csv'
    else:
        # Running from BDA LAB root or other location
        data_path = './Mini Project/Dataset/202603-capitalbikeshare-tripdata.csv'
    
    # Also try alternative paths if the primary one doesn't exist
    alternative_paths = [
        '../Dataset/202603-capitalbikeshare-tripdata.csv',
        './Mini Project/Dataset/202603-capitalbikeshare-tripdata.csv',
        'd:/BDA LAB/Mini Project/Dataset/202603-capitalbikeshare-tripdata.csv',
        'D:\\BDA LAB\\Mini Project\\Dataset\\202603-capitalbikeshare-tripdata.csv',
    ]
    
    if not os.path.exists(data_path):
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                data_path = alt_path
                break
    
    print("\n")
    print("=" * 80)
    print("CAPITAL BIKESHARE TIME SERIES EDA - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print(f"Working Directory: {os.getcwd()}")
    print(f"Data Path: {data_path}")
    
    try:
        # Load data
        df = load_data(data_path)
        
        # Prepare time series data
        df = prepare_time_series(df)
        
        # Check data quality
        check_missing_values(df)
        
        # Univariate analysis
        univariate_analysis(df)
        
        # Temporal patterns
        temporal_patterns(df)
        
        # User behavior
        user_behavior_analysis(df)
        
        # Bike type analysis
        bike_type_analysis(df)
        
        # Outlier detection
        outlier_detection(df)
        
        # Correlation analysis
        correlation_analysis(df)
        
        # Time series summary
        summary_for_timeseries(df)
        
        # Create visualizations
        create_visualizations(df)
        
        print("\n" + "=" * 80)
        print("EDA ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNext Steps for Time Series Analysis:")
        print("  1. Decompose time series (trend, seasonality, residuals)")
        print("  2. Check for stationarity (ADF test)")
        print("  3. ACF/PACF analysis for ARIMA parameters")
        print("  4. Build forecasting models (ARIMA, Prophet, etc.)")
        print("  5. Anomaly detection on time series")
        print("  6. Clustering analysis by temporal patterns")
        print("\n")
        
        # Save processed data for further analysis - in the same directory as the original dataset
        data_dir = os.path.dirname(os.path.abspath(data_path))
        output_csv_path = os.path.join(data_dir, 'capitalbikeshare_processed.csv')
        os.makedirs(data_dir, exist_ok=True)
        
        df.to_csv(output_csv_path, index=False)
        print(f"✓ Processed dataset saved: {output_csv_path}")
        
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}")
        print("Please ensure the CSV file is in the correct location.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()