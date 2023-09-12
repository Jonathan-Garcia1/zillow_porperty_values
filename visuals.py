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
    plt.ylim(0, 8000)
    
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
    
    # Remove the top and right spines for a cleaner appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Show the plot
    plt.show()

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def age_vs_value_plt(df):
    """
    Create a box plot to visualize the relationship between property age (by decade) and property value.

    Parameters:
        df (pd.DataFrame): The DataFrame containing property age and property value data to be plotted.

    Returns:
        None

    Note:
        - The function creates a box plot to show how the age of properties (by decade) relates to property value.
        - It sets appropriate labels and titles for the plot.
        - The x-axis represents property decades, and the y-axis represents property values.
        - Outliers are not shown on the plot (showfliers=False).
        - The y-axis values are formatted to display property values in millions (e.g., $1.0M).
        - The top and right spines of the plot are removed for a cleaner appearance.
    """
    df_copy = df.copy()
    
    # Create a new column 'decade' by binning 'year' into decades
    df_copy['decade'] = (df_copy['year'] // 10) * 10
    
    # Create the plot using the 'decade' column
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='decade', y='value', data=df_copy, showfliers=False, color='#1f77b4')
    plt.xlabel('Decade Built')
    plt.ylabel('Property Value')
    plt.title('Property Age vs. Property Value')
    plt.xticks(rotation=45)
    plt.grid(False)
    
    # Set y-axis limits for property value
    plt.ylim(0, 3000000)
    
    # Set x-axis limits for decades
    plt.xlim(0.5, 14.5)

    # Format the y-axis labels to display property values in millions
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.1f}M'.format(x / 1e6)))
    
    # Remove the top and right spines for a cleaner appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Show the plot
    plt.show()

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

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
        - The x-axis represents the number of bedrooms, and the y-axis represents property values.
        - The y-axis values are formatted to display property values in millions (e.g., $1M).
        - The top and right spines of the plot are removed for a cleaner appearance.
    """
    plt.figure(figsize=(10, 6))

    sns.barplot(x='bedrooms', y='value', data=df, color='#1f77b4') 
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Property Value')
    plt.title('Number of Bedrooms vs. Property Value')
    
    # Set y-axis limits for property value
    plt.ylim(0, 10000000)
    
    # Set x-axis limits for bedrooms
    plt.xlim(-0.5, 11.5)
    
    # Format the y-axis labels to display property values in millions
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.0f}M'.format(x / 1e6)))
    
    # Remove the top and right spines for a cleaner appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
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
        - The x-axis represents the number of bathrooms, and the y-axis represents property values.
        - The y-axis values are formatted to display property values in millions (e.g., $1M).
        - The top and right spines of the plot are removed for a cleaner appearance.
    """
    plt.figure(figsize=(10, 6))

    sns.barplot(x='bathrooms', y='value', data=df, color='#1f77b4') 
    plt.xlabel('Number of Bathrooms')
    plt.ylabel('Property Value')
    plt.title('Number of Bathrooms vs. Property Value')
    
    # Set y-axis limits for property value
    plt.ylim(0, 20000000)
    
    # Set x-axis limits for bathrooms
    plt.xlim(-0.5, 18.5)
    
    # Format the y-axis labels to display property values in millions
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.0f}M'.format(x / 1e6)))
    
    # Remove the top and right spines for a cleaner appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def county_vs_value_plt(df):
    """
    Create a box plot to visualize the relationship between county and property value.

    Parameters:
        df (pd.DataFrame): The DataFrame containing county and property value data to be plotted.

    Returns:
        None

    Note:
        - The function creates a box plot to show how property values vary across different counties.
        - It sets appropriate labels and titles for the plot.
        - The x-axis represents counties, and the y-axis represents property values.
        - Outliers are not shown on the plot (showfliers=False).
        - The y-axis values are formatted to display property values in millions (e.g., $1.0M).
        - The top and right spines of the plot are removed for a cleaner appearance.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='county', y='value', data=df, showfliers=False)  # Use sns.boxplot for box plot
    plt.xlabel('County')
    plt.ylabel('Property Value')
    plt.title('County vs. Property Value (Box Plot)')
    plt.xticks(rotation=45)
    plt.grid(False)

    # Format the y-axis tick labels as dollars with millions
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.1f}M'.format(x / 1e6)))
    
    # Remove the top and right spines for a cleaner appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()

