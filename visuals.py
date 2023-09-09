import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

def plot_value_distribution(df):

    # Create a histogram of property values
    plt.figure(figsize=(10, 6))
    plt.hist(df['value'], bins=491, color='#1f77b4', edgecolor='black')
    plt.xlabel('Property Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Property Values')
    #plt.grid(True)

    plt.xlim(0, 3000000)  # xlim for property size
    plt.ylim(0, 8000)  # ylim for property value
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.1f}M'.format(x / 1e6)))  # Corrected the y-axis

    # Remove the top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

def area_vs_value_plt(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['area'], df['value'], alpha=0.5)
    plt.xlabel('Property Size (Square Feet)')
    plt.ylabel('Property Value')
    plt.title('Property Size vs. Property Value')
    plt.grid(False)

    plt.xlim(0, 12000)  # xlim for property size
    plt.ylim(0, 4500000)  # ylim for property value

    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.1f}M'.format(x / 1e6)))  # Corrected the y-axis 
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()

def age_vs_value_plt(df):
    # Create a new column 'decade' by binning 'year' into decades
    df['decade'] = (df['year'] // 10) * 10

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='decade', y='value', data=df, showfliers = False, color= '#1f77b4')
    plt.xlabel('Decade Built')
    plt.ylabel('Property Value')
    plt.title('Property Age vs. Property Value')
    plt.xticks(rotation=45)
    plt.grid(False)
    plt.ylim(0, 3000000)  # ylim for property value
    plt.xlim(0.5, 14.5)  # xlim for bedrooms

    # Format the y-axis labels to display in millions (M) or thousands (K)
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.1f}M'.format(x / 1e6)))   
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.show()


def bedr_vs_value_plt(df):
        
    plt.figure(figsize=(10, 6))

    sns.barplot(x='bedrooms', y='value', data=df, color='#1f77b4') 
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Property Value')
    plt.title('Number of Bedrooms vs. Property Value')
    plt.ylim(0, 10000000)  # ylim for property value
    plt.xlim(-0.5, 11.5)  # xlim for bedrooms
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.0f}M'.format(x / 1e6)))  # Corrected the y-axis formatter
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def bathr_vs_value_plt(df):
    plt.figure(figsize=(10, 6))

    sns.barplot(x='bathrooms', y='value', data=df, color='#1f77b4') 
    plt.xlabel('Number of Bathrooms')
    plt.ylabel('Property Value')
    plt.title('Number of Bathrooms vs. Property Value')
    plt.ylim(0, 20000000)  # ylim for property value
    plt.xlim(-0.5, 18.5)  # xlim for bathrooms

    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.0f}M'.format(x / 1e6)))  # Corrected the y-axis formatter
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def county_vs_value_plt(df):

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='county', y='value', data=df, showfliers = False)  # Use sns.boxplot for box plot
    plt.xlabel('County')
    plt.ylabel('Property Value')
    plt.title('County vs. Property Value (Box Plot)')
    plt.xticks(rotation=45)
    plt.grid(False)

    # Format the y-axis tick labels as dollars with millions
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.1f}M'.format(x / 1e6)))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()
