import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

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

def plot_numeric_distributions(df, numeric_cols):
    num_plots = len(numeric_cols)
    num_rows = (num_plots // 3) + (1 if num_plots % 3 else 0)  # Calculate number of rows (round up)
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5), constrained_layout=True)

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
    plt.tight_layout()
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
    fig, axes = plt.subplots(num_rows, ncols, figsize=(5 * ncols, num_rows * 5), constrained_layout=True)
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
            print('Unique values: ', labels)
            custom_labels = [int(label) if int(label) == float(label) else round(label, 2) for label in labels]  # Format labels
            print('Custom labels: ', custom_labels)
            ax.set_yticklabels(custom_labels)  # Apply formatted labels
            ax.grid(axis='x', alpha=0.75)

    # Hide unused subplots if the number of columns doesn't fill the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Display the plot
    plt.show()


def plot_correlation_matrix(df, numeric_cols, cmap='Blues'):
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

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    sns.heatmap(abs(corr_matrix), annot=corr_matrix, cmap=cmap, fmt='.2f', linewidths=0.5, cbar=True)

    # Customize the plot
    plt.title('Correlation Matrix of Numeric Variables')
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()


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
    price_cols = [col for col in price_cols if col in df.columns]

    # Set up the subplots (2 columns)
    num_plots = len(price_cols)
    num_rows = (num_plots // 2) + (1 if num_plots % 2 else 0)  # Calculate the number of rows (rounding up)
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5), constrained_layout=True)

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Randomly sample `n_sample` IDs and plot their individual trends (dashed lines)
    # Filter out IDs with exactly 12 rows (no missing months) before sampling them
    #valid_ids = df.groupby(id_col).filter(lambda x: len(x) == 12)[id_col].unique()
    valid_ids = df.groupby(id_col).filter(lambda x: len(x) < 12)[id_col].unique()
    df_valid = df[df[id_col].isin(valid_ids)]
    sampled_ids = np.random.choice(df_valid[id_col].unique(), size=n_sample, replace=False)

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
        ax.grid(True)

    # Show the plot
    plt.tight_layout()
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
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5), constrained_layout=True)

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

    plt.show()

def apply_log_transformation_and_impute(df, numeric_cols, skewness_threshold=1.0, target='churn'):
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
    # Step 1: Identify columns with skewness above the threshold for log transformation (consider only non-zero values)
    skewness = df[numeric_cols].apply(lambda col: col[col != 0].skew(), axis=0)
    cols_with_many_zeros = [col for col in skewness.index if abs(skewness[col]) > skewness_threshold]

    for col in cols_with_many_zeros:
        # Step 2: Create a flag column for zero values
        flag_col = f'is_zero_{col}'
        df[flag_col] = (df[col] == 0).astype(int)  # Flag for zero vs. non-zero

        # Step 3: Log-transform non-zero values or set 0s to missing (for special columns like 'forecast_price_energy_peak')
        if col != 'forecast_price_energy_peak':
            new_col_name_log = f'{col}_distr_log_transformed'
            df[new_col_name_log] = np.where(df[col] != 0, np.log1p(df[col]), np.nan)  # log(x + 1) for non-zero values
        else:
            new_col_name_log = f'{col}_distr'
            df[new_col_name_log] = np.where(df[col] != 0, df[col], np.nan)

        # Step 4: Calculate the original Point-Biserial correlation before imputation
        valid_rows = df[[new_col_name_log, target]].dropna()  # Drop rows with NaN in either column
        original_corr, _ = pointbiserialr(valid_rows[new_col_name_log], valid_rows[target].map({'No': 0, 'Yes': 1}))
        print(f'Original Point-Biserial Correlation for {new_col_name_log}: {original_corr:.3f}')

        # Step 5: Impute missing values and choose the value that minimizes the change in correlation
        best_imputation = None
        best_corr_diff = float('inf')
        candidate_values = np.linspace(df[new_col_name_log].min(), df[new_col_name_log].max(), 100)  # Imputation candidates

        # Try imputing with each candidate value
        for value in candidate_values:
            temp_df = df[[new_col_name_log, target]].copy()
            temp_df[new_col_name_log] = temp_df[new_col_name_log].fillna(value)

            # Calculate the Point-Biserial correlation after imputation
            new_corr, _ = pointbiserialr(temp_df[new_col_name_log], temp_df[target].map({'No': 0, 'Yes': 1}))

            # Calculate the absolute difference between the original and new correlation
            corr_diff = abs(original_corr - new_corr)

            # Track the imputation that results in the smallest difference
            if corr_diff < best_corr_diff:
                best_corr_diff = corr_diff
                best_imputation = value

        # Delete the temp_df after using it
        del temp_df
        
        # Step 6: Impute the missing values with the best value
        df[new_col_name_log] = df[new_col_name_log].fillna(best_imputation)
        print(f'Best imputation for {col}: {best_imputation}')
        print(f'Point-Biserial Correlation difference: {best_corr_diff}')
    
    return df, skewness