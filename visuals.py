import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

def plot_value_distribution(df):
    """
    Create a histogram to visualize the distribution of property values in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing property value data to be plotted.

    Returns:
        None

    Note:
        - The function creates a histogram to visualize the distribution of property values.
        - It sets appropriate labels and titles for the plot.
        - The x-axis is formatted to display property values in millions (e.g., $1.0M).
        - The top and right spines of the plot are removed for a cleaner appearance.
    """
    # Create a histogram of property values
    plt.figure(figsize=(10, 6))
    plt.hist(df['value'], bins=491, color='#1f77b4', edgecolor='black')
    plt.xlabel('Property Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Property Values')

    # Set x-axis limits for property size
    plt.xlim(0, 3000000)
    
    # Set y-axis limits for property value
    plt.ylim(0, 6000)
    
    # Format the x-axis to display property values in millions
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.1f}M'.format(x / 1e6)))
    
    # Remove the top and right spines for a cleaner appearance
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Show the plot
    plt.show()

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def area_vs_value_plt(df):
    """
    Create a scatter plot to visualize the relationship between property size and property value.

    Parameters:
        df (pd.DataFrame): The DataFrame containing property size and property value data to be plotted.

    Returns:
        None

    Note:
        - The function creates a scatter plot to show how property size (in square feet) relates to property value.
        - It sets appropriate labels and titles for the plot.
        - The x-axis is limited to property sizes up to 12,000 square feet, and the y-axis is limited to property values
          up to $4.5 million.
        - The y-axis values are formatted to display property values in millions (e.g., $1.0M).
        - The top and right spines of the plot are removed for a cleaner appearance.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(df['area'], df['value'], alpha=0.5)
    plt.xlabel('Property Size (Square Feet)')
    plt.ylabel('Property Value')
    plt.title('Property Size vs. Property Value')
    plt.grid(False)

    # Set x-axis limits for property size
    plt.xlim(0, 12000)
    
    # Set y-axis limits for property value
    plt.ylim(0, 4500000)
    
    # Format the y-axis to display property values in millions
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.1f}M'.format(x / 1e6)))
    
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.0f}K'.format(x / 1e3)))
    
    # Remove the top and right spines for a cleaner appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Show the plot
    plt.show()


def area_vs_value_trend_plt(df):
    """
    Create a scatter plot with a trend line to visualize the relationship between property size and property value.

    Parameters:
        df (pd.DataFrame): The DataFrame containing property size and property value data to be plotted.

    Returns:
        None

    Note:
        - The function creates a scatter plot to show how property size (in square feet) relates to property value.
        - It sets appropriate labels and titles for the plot.
        - The x-axis is limited to property sizes up to 12,000 square feet, and the y-axis is limited to property values
          up to $4.5 million.
        - The y-axis values are formatted to display property values in millions (e.g., $1.0M).
        - The top and right spines of the plot are removed for a cleaner appearance.
    """
    plt.figure(figsize=(8, 6))
    
    # Scatter plot
    plt.scatter(df['area'], df['value'], alpha=0.5, label='Data Points')
    
    # Calculate the linear regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['area'], df['value'])
    x_values = np.linspace(0, 12000, 100)
    y_values = intercept + slope * x_values
    
    # Plot the trend line
    plt.plot(x_values, y_values, color='red', label='Trend Line')
    
    plt.xlabel('Property Size (Square Feet)')
    plt.ylabel('Property Value')
    plt.title('Property Size vs. Property Value')
    plt.grid(False)

    # Set x-axis limits for property size
    plt.xlim(0, 12000)
    
    # Set y-axis limits for property value
    plt.ylim(0, 4500000)
    
    # Format the y-axis to display property values in millions
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.1f}M'.format(x / 1e6)))
    
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.0f}K'.format(x / 1e3)))
    
    # Remove the top and right spines for a cleaner appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Show legend
    #plt.legend()
    
    # Show the plot
    plt.show()

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def age_vs_value_plt(df):
#def age_vs_value_bar_chart(df):
    """
    Create a bar chart to visualize the relationship between property age (by decade) and property value.

    Parameters:
        df (pd.DataFrame): The DataFrame containing property age and property value data to be plotted.

    Returns:
        None

    Note:
        - The function creates a bar chart to show how the age of properties (by decade) relates to property value.
        - It sets appropriate labels and titles for the plot.
        - The x-axis represents property decades, and the y-axis represents property values.
        - The y-axis values are formatted to display property values in millions (e.g., $1.0M).
        - The top and right spines of the plot are removed for a cleaner appearance.
    """
    df_copy = df.copy()
    
    # Create a new column 'decade' by binning 'year' into decades
    df_copy['decade'] = (df_copy['year'] // 10) * 10
    
    # Calculate the mean property value for each decade
    mean_values = df_copy.groupby('decade')['value'].mean()
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=mean_values.index, y=mean_values.values, color='#1f77b4')
    plt.xlabel('Decade Built')
    plt.title('Average Property Value by Property Age')
    plt.xticks(rotation=45)
    plt.grid(False)
    
    # Remove the y-axis
    ax.get_yaxis().set_visible(False)
    
    # Display property values on top of the bars
    for i, value in enumerate(mean_values):
        ax.text(i, value, f'${value / 1e6:.1f}M', ha='center', va='bottom', fontsize=10)
    
    # Remove the top and right spines for a cleaner appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Show the plot
    plt.show()
# -----------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------

def bedr_vs_value_plt(df):
    """
    Create a bar plot to visualize the relationship between the number of bedrooms and property value.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the number of bedrooms and property value data to be plotted.

    Returns:
        None

    Note:
        - The function creates a bar plot to show how the number of bedrooms relates to property value.
        - It sets appropriate labels and titles for the plot.
        - The x-axis represents the number of bedrooms, and property values are displayed on top of the bars.
        - The y-axis is removed.
        - The top and right spines of the plot are removed for a cleaner appearance.
    """
    plt.figure(figsize=(10, 6))

    # Filter out values above 10 million for homes with 14+ bedrooms
    df = df[(df['bedrooms'] <= 13)]

    ax = sns.barplot(x='bedrooms', y='value', data=df, color='#1f77b4', errorbar=None)
    plt.xlabel('Number of Bedrooms')
    plt.title('Average Property Value by Bedrooms')
    
    # Remove the y-axis
    ax.get_yaxis().set_visible(False)
    
    # Display property values on top of the bars
    for i, value in enumerate(df.groupby('bedrooms')['value'].mean()):
        ax.text(i, value, f'${value / 1e6:.1f}M', ha='center', va='bottom', fontsize=12)
    
    # Remove the top and right spines for a cleaner appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def bathr_vs_value_plt(df):
    """
    Create a bar plot to visualize the relationship between the number of bathrooms and property value.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the number of bathrooms and property value data to be plotted.

    Returns:
        None

    Note:
        - The function creates a bar plot to show how the number of bathrooms relates to property value.
        - It sets appropriate labels and titles for the plot.
        - The x-axis represents the number of bathrooms, and property values are displayed on top of the bars.
        - The y-axis is removed.
        - The top and right spines of the plot are removed for a cleaner appearance.
        - Property values above 10 million for homes with 14+ bathrooms are filtered out.
    """
    plt.figure(figsize=(10, 6))

    # Filter out values above 10 million for homes with 14+ bathrooms
    df = df[(df['bedrooms'] <= 13)]

    # Bin the number of bedrooms by whole numbers
    df.loc[:, 'bathrooms'] = np.ceil(df['bathrooms']).astype(int)

    ax = sns.barplot(x='bathrooms', y='value', data=df, color='#1f77b4', errorbar=None)
    plt.xlabel('Number of Bathrooms')
    plt.title('Average Property Value by Number of Bathrooms')
    
    # Remove the y-axis
    ax.get_yaxis().set_visible(False)
    
    # Display property values on top of the bars
    for i, value in enumerate(df.groupby('bathrooms')['value'].mean()):
        ax.text(i, value, f'${value / 1e6:.1f}M', ha='center', va='bottom', fontsize=10)
    
    # Remove the top and right spines for a cleaner appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def county_vs_value_plt(df):
    """
    Create a bar chart to visualize the relationship between county and property value.

    Parameters:
        df (pd.DataFrame): The DataFrame containing county and property value data to be plotted.

    Returns:
        None

    Note:
        - The function creates a bar chart to show how property values vary across different counties.
        - It sets appropriate labels and titles for the plot.
        - The x-axis represents counties, and the y-axis represents property values.
        - The y-axis values are formatted to display property values in millions (e.g., $1.0M).
        - The top and right spines of the plot are removed for a cleaner appearance.
    """
    # Define the desired order of counties
    plt.figure(figsize=(10, 6))

    # Calculate the mean property value for each county
    mean_values = df.groupby('county')['value'].mean().reset_index()

    ax = sns.barplot(x='county', y='value', data=mean_values, color='#1f77b4')
    plt.xlabel('County')
    plt.ylabel('Average Property Value')
    plt.title('Average Property Value by County (Bar Chart)')

    # Remove the y-axis
    ax.get_yaxis().set_visible(False)
    # Display property values on top of the bars
    for i, value in enumerate(df.groupby('county')['value'].mean()):
        ax.text(i, value, f'${value / 1e3:.0f}K', ha='center', va='bottom', fontsize=12)
    
    # Remove the top and right spines for a cleaner appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()