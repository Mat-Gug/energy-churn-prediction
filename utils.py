import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

def load_and_convert_datetime(file_path, date_cols, date_format='%Y-%m-%d'):
    """
    Load a CSV file into a DataFrame and convert specified columns to datetime.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    - date_cols (list): List of column names to convert to datetime.
    - date_format (str): The format of the datetime values (default is '%Y-%m-%d').
    
    Returns:
    - pd.DataFrame: A DataFrame with the specified columns converted to datetime.
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert specified columns to datetime
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
    
    return df

def compute_and_sort_statistics(df, numeric_cols, by='zero_counts', ascending=False):
    """
    Compute skewness, zero counts, and cardinality for specified numeric columns 
    in a DataFrame and return a sorted DataFrame based on the specified column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric columns to analyze.
    by (str): Column to sort by in the result. Default is 'cardinality'.
    ascending (bool): Sort order; True for ascending, False for descending. Default is True.

    Returns:
    pd.DataFrame: A DataFrame with skewness, zero counts, and cardinality, sorted by the specified column.
    """
    # Compute skewness, zero counts, and cardinality
    skewness = df[numeric_cols].skew()
    zero_counts = (df[numeric_cols] == 0).sum()
    cardinality = df[numeric_cols].nunique()

    # Combine statistics into a DataFrame
    combined_df = pd.DataFrame({
        'skewness': skewness,
        'zero_counts': zero_counts,
        'cardinality': cardinality
    })

    # Sort the DataFrame based on the specified column
    sorted_combined_df = combined_df.sort_values(by=by, ascending=ascending)

    return sorted_combined_df

def plot_numeric_distributions(df, numeric_cols, n_cols=3):
    num_plots = len(numeric_cols)
    num_rows = (num_plots // n_cols) + (1 if num_plots % n_cols else 0)  # Calculate number of rows (round up)
    fig, axes = plt.subplots(num_rows, n_cols, figsize=(5 * n_cols, num_rows * 5), layout="constrained")

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Plot histograms for numeric variables
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=25, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].tick_params(axis='x', rotation=45)

    # Remove unused subplots (if any)
    for k in range(num_plots, len(axes)):
        fig.delaxes(axes[k])

    # Adjust layout and show the plot
    #plt.tight_layout()
    plt.show()


def plot_bar_charts(df, categorical_cols, ncols=2, orientation='vertical', sort_index=True):
    """
    Plot bar charts for the specified categorical columns in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    categorical_cols (list): List of categorical column names to plot.
    ncols (int): Number of columns in the subplot grid. Default is 2.
    orientation (str): Bar chart orientation ('vertical' or 'horizontal'). Default is 'vertical'.

    Returns:
    None
    """
    # Define the number of plots and the grid layout
    num_plots = len(categorical_cols)
    num_rows = (num_plots // ncols) + (1 if num_plots % ncols else 0)
    
    # Create the figure and axes for the subplots
    fig, axes = plt.subplots(num_rows, ncols, figsize=(5 * ncols, num_rows * 5), layout="constrained")
    axes = axes.flatten()  # Flatten axes for easier iteration

    # Plot bar charts
    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        if sort_index:
            value_counts = df[col].value_counts().sort_index()  # Get sorted value counts
        else:
            value_counts = df[col].value_counts()
        percentages = (value_counts / value_counts.sum()) * 100

        # Plot horizontal or vertical bars based on the orientation
        if orientation == 'vertical':
            bars = value_counts.plot(kind='bar', color='salmon', edgecolor='black', ax=ax)
            max_height = max(value_counts)  # Maximum height of bars
            ax.set_ylim(0, max_height * 1.1)  # Add 15% space above the bars
            # Annotate bars with count and percentages
            for j, (count, pct) in enumerate(zip(value_counts, percentages)):
                ax.text(j, count + 0.5, f"{count}\n({pct:.2f}%)", ha='center', va='bottom', fontsize=9, color='black')
        elif orientation == 'horizontal':
            bars = value_counts.plot(kind='barh', color='skyblue', edgecolor='black', ax=ax)
            max_width = max(value_counts)  # Maximum width of bars
            ax.set_xlim(0, max_width * 1.15)  # Add 15% space to the right of the bars
            # Annotate bars with count
            for bar in bars.patches:
                width = bar.get_width()
                if width > 0:
                    ax.text(
                        width,  # x position (width of the bar)
                        bar.get_y() + bar.get_height() / 2,  # y position (middle of the bar)
                        f'{int(width)}',  # Frequency count
                        ha='left', va='center', fontsize=10, color='black'
                    )

        # Customize the chart
        ax.set_title(f'Frequency of {col}')
        if orientation == 'vertical':
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.grid(axis='y', alpha=0.75)
        elif orientation == 'horizontal':
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Value')
            # Apply custom labels based on integer or float
            labels = value_counts.index
            custom_labels = [int(label) if int(label) == float(label) else round(label, 2) for label in labels]  # Format labels
            ax.set_yticklabels(custom_labels)  # Apply formatted labels
            ax.grid(axis='x', alpha=0.75)

    # Hide unused subplots if the number of columns doesn't fill the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Show the plot
    #plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, numeric_cols, cmap='Blues', threshold=0.9, figsize=(15, 12)):
    """
    Plot the correlation matrix for the specified numeric columns in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names to include in the correlation matrix.

    Returns:
    None
    """
    # Compute the correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Select upper triangle of the correlation matrix to avoid duplicates
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Filter pairs with correlation greater than the threshold
    high_corr_pairs = upper_triangle.stack().reset_index()
    high_corr_pairs.columns = ['variable_1', 'variable_2', 'correlation']
    high_corr_pairs = high_corr_pairs[high_corr_pairs['correlation'].abs() > threshold]

    # Create a mask for values less than or equal to threshold
    mask = corr_matrix.abs() <= threshold
    #annotated_matrix = corr_matrix.where(~mask)  # Replace masked values with NaN

    # Plot the correlation matrix
    plt.figure(figsize=figsize)  # Adjust the figure size as needed
    sns.heatmap(abs(corr_matrix),
                annot=corr_matrix,
                cmap=cmap, 
                fmt='.2f', 
                linewidths=0.5, 
                cbar=True,
                mask=mask)

    # Customize the plot
    plt.title('Correlation Matrix of Numeric Variables')
    #plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()

    return high_corr_pairs

def plot_price_trends(df, price_cols, date_col, id_col, start_date='2015-01-01', end_date='2015-12-31', n_sample=5):
    """
    Plot the price trends across all IDs for each price variable using subplots.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing price and date columns.
    price_cols (list): List of price columns to plot.
    date_col (str): The name of the column containing the date.
    id_col (str): The name of the column containing the IDs.
    start_date (str): The start date for the period to consider (default is January 1, 2015).
    end_date (str): The end date for the period to consider (default is December 31, 2015).
    n_sample (int): The number of IDs to randomly sample and plot (default is 5).

    Returns:
    None
    """
    # Set up the subplots (2 columns)
    num_plots = len(price_cols)
    num_rows = (num_plots // 2) + (1 if num_plots % 2 else 0)  # Calculate the number of rows (rounding up)
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5), layout="constrained")

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Randomly sample `n_sample` IDs and plot their individual trends (dashed lines)
    # Filter out IDs with exactly 12 rows (no missing months) before sampling them
    #valid_ids = df.groupby(id_col).filter(lambda x: len(x) == 12)[id_col].unique()
    #valid_ids = df.groupby(id_col).filter(lambda x: len(x) < 12)[id_col].unique()
    #df_valid = df[df[id_col].isin(valid_ids)]
    sampled_ids = np.random.choice(df[id_col].unique(), size=n_sample, replace=False)

    # Loop through each price column and create a separate plot
    for i, price_col in enumerate(price_cols):
        # Compute the monthly average for the price column
        monthly_avg = df.groupby(df[date_col].dt.to_period('M'))[price_col].mean()

        # Plot the monthly average trend on the subplot (solid line)
        ax = axes[i]
        ax.plot(monthly_avg.index.astype(str), monthly_avg, label=f'Average {price_col}', linewidth=2, color='blue')

        for sampled_id in sampled_ids:
            # Filter the data for the specific ID
            id_data = df[df[id_col] == sampled_id]

            # Compute the monthly average price for this ID
            monthly_id_avg = id_data.groupby(id_data[date_col].dt.to_period('M'))[price_col].mean()

            # Plot the individual trend on the same subplot (dashed line)
            ax.plot(monthly_id_avg.index.astype(str), monthly_id_avg, linestyle='--', color='gray', alpha=0.6)

        # Customize each subplot
        ax.set_title(f'{price_col} Trend (2015)', fontsize=14)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

    # Show the plot
    #plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_histograms_for_sparse_cols(df, numeric_cols, zero_threshold=0.3):
    """
    Plot histograms for numeric columns with a significant number of zeros and high cardinality.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_cols (list): List of numeric column names to analyze.
    zero_threshold (float): Proportion threshold for considering a column as having "many zeros." Default is 0.3.

    Returns:
    None
    """
    # Compute zero counts and cardinality
    zero_counts = (df[numeric_cols] == 0).sum()

    # Identify columns with many zeros and sufficient cardinality
    cols_with_many_zeros = [
        col for col in zero_counts[zero_counts > zero_threshold * len(df)].index 
    ]

    # Number of subplots
    num_plots = len(cols_with_many_zeros)
    if num_plots == 0:
        print("No columns meet the criteria for plotting.")
        return

    num_rows = (num_plots // 3) + (1 if num_plots % 3 else 0)
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5), layout="constrained")

    # Flatten axes for easier iteration
    axes = axes.flatten()

    for i, col in enumerate(cols_with_many_zeros):
        # Filter out zeros
        col_no_zeros = df[col][df[col] != 0]
        col_skewness = col_no_zeros.skew()
        print(f'Skewness for column {col} (excluding zeros): {col_skewness:.3f}')
        print('---------------------------------')

        # Plot histogram for non-zero values
        axes[i].hist(col_no_zeros.dropna(), bins=25, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution of {col} (non-zero values)')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].tick_params(axis='x', rotation=45)

    # Remove unused subplots, if any
    for k in range(num_plots, len(axes)):
        fig.delaxes(axes[k])

    # Show the plot
    #plt.tight_layout()
    plt.show()

def apply_log_transformation_and_impute(df, cols, skewness_threshold=1.0, target='churn'):
    """
    Apply log transformation to skewed numeric columns and impute missing values in a way that minimizes the 
    change in Point-Biserial correlation with the target variable. The skewness is computed based on non-zero values.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    numeric_cols (list): A list of numeric column names to process.
    skewness_threshold (float): The skewness threshold above which the log transformation is applied.
    target (str): The target variable (e.g., 'churn').

    Returns:
    pd.DataFrame: The dataframe with transformed columns and imputed values.
    """
    skewness = df[cols].apply(lambda col: col[col != 0].skew(), axis=0)
    skewed_cols = [col for col in skewness.index if abs(skewness[col]) > skewness_threshold]
    transformed_cols = []

    for col in cols:
        flag_col = f'is_zero_{col}'
        df[flag_col] = (df[col] == 0).astype(int)  # Flag for zero vs. non-zero

        if (col in skewed_cols) and (skewness[col]<0):
            col_name = f"{col}_square"
            #df[col_name] = df[col].apply(lambda x: x**2)
            df[col_name] = np.where(df[col] != 0, df[col]**2, np.nan)
        elif (col in skewed_cols) and (skewness[col]>0):
            col_name = f"{col}_log"
            #df[col_name] = df[col].apply(lambda x: np.log1p(x))
            df[col_name] = np.where(df[col] != 0, np.log1p(df[col]), np.nan)
        else:
            col_name = f"{col}_distr"
            df[col_name] = np.where(df[col] != 0, df[col], np.nan)

        transformed_cols.append(col_name)

        # Calculate the original Point-Biserial correlation before imputation
        valid_rows = df[[col_name, target]].dropna()  # Drop rows with NaN in either column
        original_corr, _ = pointbiserialr(valid_rows[col_name], valid_rows[target].map({'No': 0, 'Yes': 1}))
        print(f'Original Point-Biserial Correlation for {col_name}: {original_corr:.3f}')

        # Step 5: Impute missing values and choose the value that minimizes the change in correlation
        best_imputation = None
        best_corr_diff = float('inf')
        candidate_values = np.linspace(df[col_name].min(), df[col_name].max(), 100)  # Imputation candidates

        # Try imputing with each candidate value
        for value in candidate_values:
            temp_df = df[[col_name, target]].copy()
            temp_df[col_name] = temp_df[col_name].fillna(value)

            # Calculate the Point-Biserial correlation after imputation
            new_corr, _ = pointbiserialr(temp_df[col_name], temp_df[target].map({'No': 0, 'Yes': 1}))

            # Calculate the absolute difference between the original and new correlation
            corr_diff = abs(original_corr - new_corr)

            # Track the imputation that results in the smallest difference
            if corr_diff < best_corr_diff:
                best_corr_diff = corr_diff
                best_imputation = value

        # Delete the temp_df after using it
        del temp_df
        
        # Impute the missing values with the best value
        df[col_name] = df[col_name].fillna(best_imputation)
        print(f'Best imputation for {col}: {best_imputation}')
        print(f'Point-Biserial Correlation difference: {best_corr_diff}')
    
    return df, skewness, transformed_cols


def engineer_saz_features(df, cols, skewness_threshold=1.0):
    """
    Engineer Spike-At-Zero (SAZ) features by applying transformations to skewed numeric columns and creating indicators 
    for zero values. Skewness is computed based on non-zero values.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    cols (list): A list of numeric column names to process.
    skewness_threshold (float, optional): The absolute skewness threshold above which a transformation 
                                          is applied. Defaults to 1.0.

    Returns:
    pd.DataFrame: The dataframe with transformed columns and imputed values.
    list: A list of new column names generated by the transformations.

    Transformations:
    - If a column is **positively skewed** (skewness > skewness_threshold), apply a **log transformation**.
    - If a column is **negatively skewed** (skewness < -skewness_threshold), apply a **square transformation**.
    - Otherwise, retain the original distribution but impute missing values.
    - Create a binary indicator column (`is_zero_<col>`) marking whether the original value was zero.

    Notes:
    - Log and square transformations are applied only to non-zero values.
    - The zero indicator column is stored as an object type with values 'Yes' and 'No'.
    """
    skewness = df[cols].apply(lambda col: col[col != 0].skew(), axis=0)
    skewed_cols = [col for col in skewness.index if abs(skewness[col]) > skewness_threshold]
    transformed_cols = []

    for col in cols:
        flag_col = f'is_zero_{col}'
        df[flag_col] = (df[col] == 0).astype(int).astype('object')  # Flag for zero vs. non-zero
        df.loc[:, flag_col] = df[flag_col].map({0: 'No', 1: 'Yes'})
        
        if (col in skewed_cols) and (skewness[col]<0):
            col_name = f"{col}_square"
            #df[col_name] = df[col].apply(lambda x: x**2)
            df[col_name] = np.where(df[col] != 0, df[col]**2, np.nan)
        elif (col in skewed_cols) and (skewness[col]>0):
            col_name = f"{col}_log"
            #df[col_name] = df[col].apply(lambda x: np.log1p(x))
            df[col_name] = np.where(df[col] != 0, np.log1p(df[col]), np.nan)
        else:
            col_name = f"{col}_distr"
            df[col_name] = np.where(df[col] != 0, df[col], np.nan)

        transformed_cols.append(col_name)
    
    return df, transformed_cols

def remove_underrepresented_categories(df, columns, threshold=0.01):
    """
    Remove rows belonging to categories that appear in less than the specified threshold (e.g., 1%) of total observations.
    
    Parameters:
    - df: The input DataFrame.
    - columns: List of column names to process.
    - threshold: The minimum proportion (between 0 and 1) to keep categories.
    
    Prints out the removed categories and which columns they belong to.
    """
    for column in columns:
        # Get the category counts and their proportions
        category_counts = df[column].value_counts(normalize=True)
        
        # Find categories under the threshold
        underrepresented_categories = category_counts[category_counts < threshold].index
        
        if len(underrepresented_categories) > 0:
            # Print the removed categories and the column they belong to
            print(f"\nRemoved categories from '{column}':")
            for category in underrepresented_categories:
                print(f"  - {category}")
                
            # Remove rows that belong to underrepresented categories
            df = df[~df[column].isin(underrepresented_categories)]
    
    return df


def process_graph(data):
    """
    Processes a table of arcs to iteratively remove variables based on the criteria described.
    
    Parameters:
    - data (pd.DataFrame): A DataFrame with columns ['variable_1', 'variable_2', 'correlation'].
    
    Returns:
    - remaining_variables (list): List of remaining variables after processing.
    - remaining_data (pd.DataFrame): Remaining rows of the DataFrame after processing.
    """
    # Create the initial graph from the data
    def create_graph(df):
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(row['variable_1'], row['variable_2'], weight=row['correlation'])
        return G

    def get_strongest_arc(node, graph):
        """Get the strongest arc for a given node in the graph."""
        edges = graph[node]
        return max(edges.values(), key=lambda x: x['weight'])['weight']
    
    def get_strongest_arcs(node, graph, n=2):
        """
        Get the strongest arcs (up to `n`) for a given node in the graph.
        
        Returns:
        - List of the strongest weights in descending order.
        """
        edges = graph[node]
        weights = sorted([attr['weight'] for attr in edges.values()], reverse=True)
        return weights[:n]  # Return up to `n` strongest weights

    # Initialize the graph
    G = create_graph(data)
    print('Initial graph:')
    print(G)
    variables_to_remove = []
    
    while True:
        # Count the number of connected components
        components = list(nx.connected_components(G))
        print('The graph is made of the following components:')
        for i, component in enumerate(components):
            print(f"{i+1}) {component}")
        if len(components) == len(G.nodes):  # No arcs remain
            print('There are no remaining arcs, we are done!')
            break

        # Process each connected component
        for i, component in enumerate(components):
            print(f"Let's see if there's something to do for the component n. {i+1}:")
            print(G.subgraph(component))
            
            if len(component) <= 1:
                print('I found a component with only one node, nothing to do with it :)')
                print('-------------------------------------')
                continue  # Skip components with only one node
            
            # Create a subgraph for the current component
            
            # Count arcs for each node in the component
            node_degrees = {node: len(G[node]) for node in component}
            max_degree = max(node_degrees.values())
            
            # Find nodes with the maximum degree
            candidates = [node for node, degree in node_degrees.items() if degree == max_degree]
            print("The candidate nodes to remove (highest number of arcs) are the following:")
            print(candidates)

            if len(candidates) > 1:
                # Compare the strongest and second strongest arcs for each candidate
                candidates.sort(
                    key=lambda node: (
                        -get_strongest_arcs(node, G)[0],  # Sort by first strongest arc (descending)
                        -get_strongest_arcs(node, G)[1] if len(get_strongest_arcs(node, G)) > 1 else -float('inf'),  # Then by second strongest arc (descending)
                        node  # Finally, by node name (ascending)
                    )
                )
                print(candidates)

            # Remove the node with the highest priority
            node_to_remove = candidates[0]
            print(f"Let's remove node {node_to_remove}")
            
            G.remove_node(node_to_remove)
            print(f'Graph after {node_to_remove} removal')
            print(G)
            print('-------------------------------------')

            variables_to_remove.append(node_to_remove)
    
    print('Final disconnected graph:')
    print(G)
    
    return variables_to_remove