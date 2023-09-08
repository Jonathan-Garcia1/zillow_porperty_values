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

def area_vs_value(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['area'], df['value'], alpha=0.5)
    plt.xlabel('Property Size (Square Feet)')
    plt.ylabel('Property Value')
    plt.title('Property Size vs. Property Value')
    plt.grid(True)

    plt.xlim(0, 12000)  # xlim for property size
    plt.ylim(0, 4500000)  # ylim for property value

    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '${:.1f}M'.format(x / 1e6)))  # Corrected the y-axis 
    plt.show()
