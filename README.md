# 🔥 Predicting customer churn in the energy sector with SAS Viya Workbench

Customer churn is a major challenge for energy providers, directly impacting revenue and long-term business stability. This project aims to develop a churn prediction model to help energy companies proactively identify at-risk customers and implement targeted retention strategies. The dataset we used can be found [here](https://www.kaggle.com/datasets/erolmasimov/powerco). It is a Kaggle dataset that provides insights into Small and Medium Enterprises (SMEs) served by the PowerCO energy supplier.

The goal is to build and evaluate machine learning models using both **[scikit-learn](https://scikit-learn.org/stable/index.html)** and the **[SAS Python API](https://go.documentation.sas.com/doc/en/workbenchcdc/v_001/vwbpymlpg/titlepage.htm)** available in **[SAS Viya Workbench](https://go.documentation.sas.com/doc/en/workbenchcdc/v_001/workbenchwlcm/home.htm)**, which enables users to leverage SAS analytics within Python workflows. The best-performing model is selected using a hyperparameter tuning procedure, and a final cutoff optimization step is performed to maximize business impact from the retention campaign.

# 📌 Project Overview

In this project, we:

- Load and preprocess customer and price data.
- Perform feature engineering and exploratory data analysis (EDA).
- Train and evaluate machine learning models to predict the churn likelihood of each customer.
- Compare model performance using AUC score and cross-validation.
- Optimize model hyperparameters using [Optuna](https://optuna.org/).
- Determine the optimal probability cutoff for maximizing retention campaign effectiveness.

# 📂 Repository Structure
- `data` folder: It contains raw and processed datasets used and created throughout the project.
- Three Jupyter Notebooks guiding the full churn prediction pipeline (data cleaning, data preparation, model training/evaluation, and cutoff optimization).
- `utils.py` and `model_building_utils.py`: Python utility scripts for preprocessing, data visualization, data preparation, and model training.
- `requirements.txt`: It lists the required Python packages for this project.

# 🔧 Setup Instructions

To ensure the project works properly, we recommend creating a dedicated storage location in SAS Viya Workbench to separate the different Python dependencies required for various projects. To do so, follow the steps outlined in the [documentation](https://go.documentation.sas.com/doc/en/workbenchcdc/v_001/workbenchgs/p1oa82y6hbc2vfn1po8y224qxwx9.htm). Next, when creating the workbench for the project (click [here](https://go.documentation.sas.com/doc/en/workbenchcdc/v_001/workbenchgs/n1fvzggnvda7v1n19e171jg5ug7i.htm) for the procedure), select the storage location you just created. Once the workbench has been set up, you can open it in your IDE of choice (Visual Studio Code, Jupyter Notebook, or Jupyter Lab), then clone this repository by running the following command in the terminal:
```
git clone https://github.com/Mat-Gug/energy-churn-prediction.git
``` 
Finally, navigate to the repository's root folder and install the required Python dependencies:
```
cd /path/to/repository
pip install -r requirements_short.txt
```

# :notebook_with_decorative_cover: Notebooks

## Churn Prediction - Part 1: Data Cleaning

This notebook focuses on loading and preprocessing the dataset to ensure data consistency and quality before model training. The key steps include:

1. **Date Handling**: Converting date columns to datetime format and ensuring valid date sequences.
2. **Feature Mapping**: Replacing hashed categorical values with more human-readable labels.
3. **Data Cleaning**: This involves several data quality checks.
    - Removing duplicate rows.
    - Checking for negative values in numerical columns.
    - Ensuring data consistency for features like `has_gas` and `cons_gas_12m`.
    - Identifying and handling invalid date sequences.
    - Verifying margin constraints for energy pricing.
    All these potential redundant or inconsistent rows are saved to separate csv files for further investigation, if needed.
4. **Missing Value Analysis**: Reporting and saving columns with missing values.
5. **Saving Cleaned Data**: Exporting processed data for further analysis.

This notebook establishes a clean dataset, preparing it for feature engineering and model building in the next steps.

## Churn Prediction - Part 2: Data Preparation

This notebook processes and prepares client and price data for churn prediction analysis by handling missing values, feature engineering, and data transformations. The key steps are:

1. **Data Exploration for the cleaned `price_data` table**: This involves counting unique IDs, examining distinct values in the `price_date` variable, and plotting the number of price records per client. 
2. **Handling Missing Data**: From the previous step, clients with incomplete price records (fewer than 12 months of data) are identified. Therefore, these missing values are imputed using median monthly prices.
3. **Feature Engineering for Price Data**: Summary statistics (mean, median, min, max, standard deviation, max difference between consecutive months) are computed for each client’s price variables. This allows us to then merge the price data with client data.
4. **Feature Engineering for Client Data and saving a first version of the data**: Before dropping date columns, relevant features are extracted (e.g., whether a product was modified, contract age at last modification, days remaining until contract end). After doing that, a first version of the dataset is saved to the `merged_data_no_tfm.csv` file.
5. **Data Transformations**:
    - Some low cardinality numeric variables (`nb_prod_act`, `num_years_antig`, `forecast_discount_energy`) are converted into categorical variables, grouping rare values.
    - Highly skewed numerical variables are transformed using square or log transformations.
    - Numerical columns with many zero values are transformed creating:
        - a flag indicator variable telling whether the variable is in the spike (0 value) or in the distribution (non-zero value).
        - a variable created from the values of the original one in the distribution and where the observations that have the value at the spike are set to missing. A log or square transformatio) is then performed on the non-missing values of this variable in case of highe skewness), while missing values will be imputed during the Model Building phase.
7.** Correlation Analysis and Feature Selectio**: After plotting a correlation matrix for both the non-transformed and transformed version of the dataset, redundant variables are removed from the two tables to avoid multicollinearity, overfitting and improve model interpretability..
8.** Saving Processed Dat**:-Tth non-transformed and thee transformed datases are savedt totwo separatea CSV filse forthe next part of the projects.

## Churn Prediction - Part 3: Model Training, Evaluation, an Cutoff Optimization

This notebook focuses on training and evaluating multiple machine learning models for churn prediction using both `sasviya` and `scikit-learn libraries`. It also includes hyperparameter optimization using **Optuna** to find the best-performing model. The key steps are:
1. **Model Training and Evaluation (Non-Transformed Data)**:  Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting model are traineds using both `sasviya` and `scikit-learn`, implemening cross-validation for model evaluation. Then, the trained models are compareds based on AUC score for a first evaluation and comparison of the two libraries.
2. **Hyperparameter Optimization with Optuna (Non-Transformed Data)**: An Optuna study is run with multiple trials to maximize model performance based on AUC score on the validation set, implementing also in this case cross-validation. Finally, the best-performing model is selected and re-trained on the full training set.
3. **Model Training and Hyperparameter Optimization (Transformed Data)**: The model training and evaluation process is repeated for the transformed data, in order to understand which data leads to the best performances
4. **Model Selection and Cutoff optimization**: After choosing the best model from the previous steps, a cutoff optimization for a retention campaign is performed, consisting of the following steps:
    - Choose the discount rate and the expected retention efficiency of the discount.
    - Determine the optimal probability threshold to maximize annual profit by balancing prevented churn losses against the cost of offering discounts.

To sum up, the key features of this last notebook are:
- Comparison of `sasviya` and `scikit-learn` models.
- Automated hyperparameter tuning with Optuna 
- Cross-validation for model evaluation.
- Cutoff optimization for identifying customers to target with a retention campaign.