import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew

# Load and clean the dataset
def load_and_clean_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Drop duplicates if any
    data = data.drop_duplicates()
    
    # Handle any potential missing values by filling them with the mean (for numeric) or mode (for categorical)
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
            data[column].fillna(data[column].mean(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)
    
    return data

# Load and clean the data
file_path = 'bestsellers with categories.csv'
data = load_and_clean_data(file_path)

# Statistical summaries
def display_statistics(data):
   
    # Summary statistics
    print("Summary Statistics:\n", data.describe())

    # Select only numeric columns for correlation calculation
    numeric_data = data.select_dtypes(include=['number'])
    
    # Correlation matrix
    correlation_matrix = numeric_data.corr()
    print("\nCorrelation Matrix:\n", correlation_matrix)

    # Skewness and Kurtosis
    skewness = numeric_data.apply(lambda x: skew(x.dropna()))
    kurt = numeric_data.apply(lambda x: kurtosis(x.dropna()))
    print("\nSkewness:\n", skewness)
    print("\nKurtosis:\n", kurt)

# Display statistical information
display_statistics(data)

# Visualization functions

def plot_histogram():
    """
    Plots a histogram of User Ratings with KDE, including the mean line and annotation.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data['User Rating'], kde=True)
    plt.title('Histogram of User Ratings', fontsize=14, fontweight='bold')
    plt.xlabel('User Rating', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    
    # Mean annotation
    mean_rating = data['User Rating'].mean()
    plt.axvline(mean_rating, color='red', linestyle='--', label=f'Mean: {mean_rating:.2f}')
    plt.legend()
    plt.show()

def plot_genre_bar_chart():
    """
    Plots a bar chart showing the distribution of Genres with count annotations.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(data['Genre'])
    plt.title('Distribution of Genres', fontsize=14, fontweight='bold')
    plt.xlabel('Genre', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    
    # Count annotations
    for index, value in enumerate(data['Genre'].value_counts()):
        plt.text(index, value + 5, str(value), ha='center', fontweight='bold')
    plt.show()

def plot_reviews_vs_year():
    """
    Plots a scatter plot of Reviews over Years with annotation for the maximum review count.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Year', y='Reviews', data=data)
    plt.title('Scatter Plot of Reviews over Years', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Reviews', fontsize=12, fontweight='bold')
    
    # Max reviews annotation
    max_reviews = data['Reviews'].max()
    max_reviews_year = data[data['Reviews'] == max_reviews]['Year'].values[0]
    plt.annotate(f'Max Reviews: {max_reviews}', xy=(max_reviews_year, max_reviews),
                 xytext=(max_reviews_year, max_reviews + 10000),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontweight='bold')
    plt.show()

def plot_price_by_genre():
    """
    Plots a box plot of Price by Genre, with median price annotations for each genre.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Genre', y='Price', data=data)
    plt.title('Box Plot of Price by Genre', fontsize=14, fontweight='bold')
    plt.xlabel('Genre', fontsize=12, fontweight='bold')
    plt.ylabel('Price', fontsize=12, fontweight='bold')
    
    # Median price annotations
    for genre in data['Genre'].unique():
        median_price = data[data['Genre'] == genre]['Price'].median()
        plt.text(data['Genre'].unique().tolist().index(genre), median_price + 2,
                 f'Median: {median_price}', ha='center', fontweight='bold')
    plt.show()

def plot_heatmap():
    """
    Plots a heatmap of the correlation matrix for numeric features.
    """
    plt.figure(figsize=(10, 8))
    numeric_data = data.select_dtypes(include=['number'])
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'fontweight': 'bold'})
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.show()

# Generate the plots
plot_histogram()
plot_genre_bar_chart()
plot_reviews_vs_year()
plot_price_by_genre()
plot_heatmap()