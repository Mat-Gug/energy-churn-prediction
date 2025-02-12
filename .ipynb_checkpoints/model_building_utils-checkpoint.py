import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# SAS models
from sasviya.ml.linear_model import LogisticRegression as SASLogisticRegression
from sasviya.ml.tree import DecisionTreeClassifier as SASDecisionTreeClassifier
from sasviya.ml.tree import DecisionTreeRegressor as SASDecisionTreeRegressor
from sasviya.ml.tree import ForestClassifier as SASForestClassifier
from sasviya.ml.tree import GradientBoostingClassifier as SASGradientBoostingClassifier

# scikit-learn classes and models
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as SKGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from imblearn.under_sampling import RandomUnderSampler


def split_data_and_generate_column_lists(df, target):
    """
    Analyzes and splits the input dataframe into training and testing sets.
    Also analyzes the distribution of the target variable and handles missing data.

    Parameters:
    df (pandas.DataFrame): The input dataset containing features and the target variable.
    target (str): The name of the target variable.

    Returns:
    tuple: A tuple containing:
        - X_train (pandas.DataFrame): The training feature set.
        - X_test (pandas.DataFrame): The test feature set.
        - y_train (pandas.Series): The training target set.
        - y_test (pandas.Series): The test target set.
        - cat_cols (list): List of categorical columns in the dataset.
        - cat_cols_with_two_values (list): List of categorical columns with exactly two unique values.
        - cat_cols_with_more_values (list): List of categorical columns with more than two unique values.
        - numeric_cols (list): List of numeric columns in the dataset.
    """
    # Separate features (X) and target variable (y)
    X = df.drop(['id', target], axis=1)
    y = df[target]

    # Display the distribution of the target variable
    for index in y.value_counts().index:
        print(f"- Number of rows with {target}='{index}': {y.value_counts()[index]} ({y.value_counts(normalize=True)[index]*100:.2f} %)")

    # print('---------------------------------')
    
    # Split the data into training and testing sets (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Identify categorical and numeric columns
    cat_cols = X.select_dtypes(exclude='number').columns
    numeric_cols = X.select_dtypes(include='number').columns

    # Categorize categorical columns into those with two unique values and those with more
    cat_cols_with_two_values = [col for col in cat_cols if X[col].nunique() == 2]
    cat_cols_with_more_values = [col for col in cat_cols if X[col].nunique() > 2]

    # Display value counts for categorical columns with two unique values
    # for col in cat_cols_with_two_values:
    #     print(f"Value counts for column {col}:")
    #     print(X[col].value_counts())
    #     print('---------------------------------')

    # Return the split data (X_train, X_test, y_train, y_test) and some useful columns info
    return X_train, X_test, y_train, y_test, cat_cols, cat_cols_with_two_values, cat_cols_with_more_values, numeric_cols

# Custom Transformer for Missing Value Imputation
class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for imputing missing values in numeric columns using a regression model.

    Attributes:
    imputation_models (dict): Dictionary storing models for imputing missing values per column.
    """
    def __init__(self):
        self.imputation_models = {}
    
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include='number').columns
        missing_numeric_cols = [col for col in numeric_cols if X[col].isnull().any()]
        #X_with_y = pd.concat([X, y], axis=1)
        for col in missing_numeric_cols:
            # Prepare data for imputation
            # non_missing_data = X_with_y[X_with_y[col].notnull()]
            non_missing_data = X[X[col].notnull()]
            # Alternative: remove all missing columns for the features!
            train_features = non_missing_data.drop(columns=[col])
            train_target = non_missing_data[col]
            cat_cols_with_two_values = [col for col in train_features.columns if X[col].nunique() == 2]
            cat_cols_with_more_values = [col for col in train_features.columns if X[col].nunique() > 2]
            
            # Train the imputation model
            imputation_model = SASDecisionTreeRegressor()
            nominal_features = train_features.select_dtypes(exclude='number').columns.tolist()
            if len(nominal_features)>0:
                imputation_model.fit(train_features, train_target, nominals=nominal_features)
            else:
                imputation_model.fit(train_features, train_target)
            
            # Store the model for this column
            self.imputation_models[col] = imputation_model
        return self
    
    def transform(self, X):
        """
        Imputes missing values in the dataset using the previously trained models.

        Parameters:
        X (pandas.DataFrame): The dataset with missing values to be imputed.

        Returns:
        pandas.DataFrame: The dataset with imputed values.
        """
        X = X.copy()
        #y = y.copy()
        #X_with_y = pd.concat([X, y], axis=1)
        for col, model in self.imputation_models.items():
            # Separate rows with missing values
            # missing_data = X_with_y[X_with_y[col].isnull()]
            missing_data = X[X[col].isnull()]
            if not missing_data.empty:
                missing_features = missing_data.drop(columns=[col])
                imputed_values = model.predict(missing_features)
                
                # Fill missing values
                #X_with_y.loc[X_with_y[col].isnull(), col] = imputed_values
                X.loc[X[col].isnull(), col] = imputed_values
        # return X_with_y.drop(columns=[y.name], axis=1)
        return X

class DataScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for standardizing numeric columns in a dataset.

    Attributes:
    numeric_cols (list): List of numeric columns to be scaled.
    scaler (StandardScaler): The scaler used for standardization.
    """
    def __init__(self, numeric_cols):
        self.numeric_cols = numeric_cols
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        """
        Fits the scaler on the numeric columns.

        Parameters:
        X (pandas.DataFrame): The dataset to be fitted.
        y (pandas.Series, optional): The target variable (not used in this transformer).

        Returns:
        self: The fitted scaler.
        """
        self.scaler.fit(X[self.numeric_cols])
        return self
    
    def transform(self, X):
        """
        Transforms the dataset by scaling the numeric columns.

        Parameters:
        X (pandas.DataFrame): The dataset to be transformed.

        Returns:
        pandas.DataFrame: The dataset with scaled numeric columns.
        """
        X = X.copy()
        X = X.apply(lambda x: x.astype('float64') if x.dtype == 'int64' else x)
        X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        return X

class CategoricalLevelFilter(BaseEstimator, TransformerMixin):
    """
    Custom transformer to filter rows based on the categories observed during training for categorical columns.

    Attributes:
    levels (dict): Dictionary of categorical columns and their observed levels.
    """
    def __init__(self):
        self.levels = {}

    def fit(self, X, y=None):
        """
        Identifies unique categories for each categorical column.

        Parameters:
        X (pandas.DataFrame): The dataset to be fitted.
        y (pandas.Series, optional): The target variable (not used in this transformer).

        Returns:
        self: The fitted transformer.
        """
        cat_cols = X.select_dtypes(exclude='number').columns
        self.levels = {col: set(X[col].unique()) 
                       for col in cat_cols}
        return self

    def transform(self, X):
        """
        Filters rows containing categories that were not observed during training.

        Parameters:
        X (pandas.DataFrame): The dataset to be transformed.

        Returns:
        pandas.DataFrame: The filtered dataset.
        """
        X_filtered = X.copy()
        # Create a mask for all categorical columns
        mask = pd.Series(True, index=X.index)
        for col, levels in self.levels.items():
            # Filter rows that contain categories unseen during training
            mask &= X[col].isin(levels)
        # Apply the mask to filter rows
        X_filtered = X[mask]
        return X_filtered

# Define the pipeline
def create_pipeline(model, numeric_cols=None):
    """
    Creates a machine learning pipeline with steps for missing value imputation, scaling, and categorical filtering.

    Parameters:
    model (sklearn.base.BaseEstimator): The machine learning model to be used in the pipeline.
    numeric_cols (list, optional): List of numeric columns to be scaled.

    Returns:
    sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    steps = []

    # Custom missing value imputation transformer
    steps.append(('imputer', MissingValueImputer()))

    # Standardization (only for numeric columns)
    if numeric_cols is not None:
        steps.append(('scaler', DataScaler(numeric_cols)))

    # Add the custom categorical level filter
    steps.append(('cat_filter', CategoricalLevelFilter()))

    # Model
    steps.append(('model', model))

    return Pipeline(steps)

def compute_auc(model, X_train, y_train, X_val, y_val):
    """
    Computes the AUC (Area Under the Curve) score for both training and validation datasets.

    Parameters:
    model (sklearn.base.BaseEstimator): The trained machine learning model.
    X_train (pandas.DataFrame): The training dataset features.
    y_train (pandas.Series): The training dataset target.
    X_val (pandas.DataFrame): The validation dataset features.
    y_val (pandas.Series): The validation dataset target.

    Returns:
    tuple: A tuple containing:
        - AUC score for the training data.
        - AUC score for the validation data.
    """
    if type(model.predict_proba(X_train))==pd.core.frame.DataFrame:
        y_proba_train = model.predict_proba(X_train).iloc[:, 1]
        y_proba_val = model.predict_proba(X_val).iloc[:, 1]
    else:
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_val = model.predict_proba(X_val)[:, 1]

    # print('X_train:', X_train.shape)
    # print('y_train:', y_train.shape)
    # print('y_proba_train:', y_proba_train.shape)
    # print('X_val:', X_val.shape)
    # print('y_val:', y_val.shape)
    # print('y_proba_val:', y_proba_val.shape)
    
    return roc_auc_score(y_train, y_proba_train), roc_auc_score(y_val, y_proba_val)

def train_and_evaluate_model(
    X, y, model, model_type, library, 
    numeric_cols=None, nominals=None, k=5, 
    resampling=True, sampling_strategy=0.25
):
    """
    Trains and evaluates a model using k-fold cross-validation.

    Parameters:
    X (pandas.DataFrame): The features dataset.
    y (pandas.Series): The target variable dataset.
    model (sklearn.base.BaseEstimator): The machine learning model to be trained.
    model_type (str): The type of model.
    library (str): The library being used (e.g., "SAS" or "sklearn").
    numeric_cols (list, optional): List of numeric columns to be scaled.
    nominals (list, optional): List of nominal columns.
    k (int, optional): Number of cross-validation folds (default is 5).
    resampling (bool, optional): Whether to use resampling (default is True).
    sampling_strategy (float, optional): Resampling strategy for imbalanced datasets (default is 0.25).

    Returns:
    dict: A dictionary containing the model, evaluation metrics, and cross-validation results.
    """
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=12345)

    auc_train_scores, auc_val_scores = [], []

    # Initialize preprocessors
    rus = RandomUnderSampler(random_state=12345, sampling_strategy=sampling_strategy) if resampling else None
        
    # scaler = StandardScaler() if numeric_cols is not None else None

    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Apply random undersampling
        if rus:
            X_train, y_train = rus.fit_resample(X_train, y_train)
        
        pipeline = create_pipeline(model, numeric_cols=numeric_cols)
        if nominals is not None:
             pipeline.fit(X_train, y_train, model__nominals=nominals)
        else:
             pipeline.fit(X_train, y_train)

        # # Scale numeric columns
        # if scaler:
        #     X_train = X_train.apply(lambda x: x.astype('float64') if x.dtype == 'int64' else x)
        #     X_val = X_val.apply(lambda x: x.astype('float64') if x.dtype == 'int64' else x)
        #     X_train.loc[:,numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        #     X_val.loc[:,numeric_cols] = scaler.transform(X_val[numeric_cols])

        cat_levels = pipeline.named_steps['cat_filter'].levels
        valid_rows_mask = X_val[list(cat_levels.keys())].apply(lambda col: col.isin(cat_levels[col.name])).all(axis=1)
        y_val = y_val[valid_rows_mask]
    
        # Compute AUC scores
        train_auc, val_auc = compute_auc(pipeline, X_train, y_train, X_val, y_val)
        auc_train_scores.append(train_auc)
        auc_val_scores.append(val_auc)

    # Aggregate results
    return pd.DataFrame([{
        'Model Type': model_type,
        'Library': library,
        'AUC_train': round(np.mean(auc_train_scores), 3),
        'AUC_val': round(np.mean(auc_val_scores), 3),
        'AUC_train_std': round(np.std(auc_train_scores), 3),
        'AUC_val_std': round(np.std(auc_val_scores), 3)
    }])


def plot_auc_and_runtime(metrics_df, data_version="non-transformed"):
    """
    Plots a side-by-side comparison of AUC scores and runtimes for SAS and scikit-learn models.

    Parameters:
    - metrics_df (pd.DataFrame): DataFrame containing model metrics, including AUC scores and runtimes.
    """
    colors = {"SAS": "blue", "scikit-learn": "orange"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AUC Comparison Plot
    ax = sns.barplot(
        data=metrics_df, 
        x="Model Type", 
        y="AUC_val", 
        hue="Library", 
        palette=colors,
        ax=axes[0]
    )
    axes[0].set_title(f"AUC Comparison of SAS vs scikit-learn Models ({data_version})")
    axes[0].set_ylabel("AUC Score")
    axes[0].set_xlabel("Model Type")
    axes[0].legend(title="Library")
    axes[0].set_ylim(0, metrics_df["AUC_val"].max() * 1.2)

    # Add AUC values to bars
    for p in ax.patches:
        axes[0].annotate(f"{p.get_height():.2f}", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=10, color='black')

    # Runtime Comparison Plot
    ax = sns.barplot(
        data=metrics_df, 
        x="Model Type", 
        y="Runtime", 
        hue="Library", 
        palette=colors,
        ax=axes[1]
    )
    axes[1].set_title(f"Runtime Comparison of SAS vs scikit-learn Models ({data_version})")
    axes[1].set_ylabel("Runtime (seconds)")
    axes[1].set_xlabel("Model Type")
    axes[1].legend(title="Library")

    # Add runtime values to bars
    for p in ax.patches:
        axes[1].annotate(f"{p.get_height():.2f}s", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()


def create_sklearn_rf(trial):
    """
    Creates a Random Forest classifier using scikit-learn with hyperparameters optimized via Optuna.

    Parameters:
    trial (optuna.trial.Trial): The Optuna trial used for hyperparameter optimization.

    Returns:
    SKRandomForestClassifier: A scikit-learn RandomForestClassifier instance with hyperparameters set based on the trial.
    """
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 20, 100)
    rf_max_depth = trial.suggest_int("rf_max_depth", 5, 25)
    rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 5, 30)
    rf_max_samples = trial.suggest_float("rf_max_samples", 0.4, 0.8)
    rf_n_bins = trial.suggest_int("rf_n_bins", 20, 50)
    rf_class_weight = trial.suggest_categorical("rf_class_weight", [None, "balanced"])
    return SKRandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf,
        max_samples = rf_max_samples,
        class_weight=rf_class_weight,
        random_state=12345
    )

def create_sas_rf(trial):
    """
    Creates a Random Forest classifier using sasviya with hyperparameters optimized via Optuna.

    Parameters:
    trial (optuna.trial.Trial): The Optuna trial used for hyperparameter optimization.

    Returns:
    SASForestClassifier: A SAS Viya RandomForestClassifier instance with hyperparameters set based on the trial.
    """
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 20, 100)
    rf_max_depth = trial.suggest_int("rf_max_depth", 5, 25)
    rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 5, 30)
    rf_bootstrap = trial.suggest_float("rf_bootstrap", 0.4, 0.8)
    rf_n_bins = trial.suggest_int("rf_n_bins", 20, 50)
    return SASForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf,
        bootstrap=rf_bootstrap,
        random_state=12345
    )

def create_sklearn_dtree(trial):
    """
    Creates a Decision Tree classifier using scikit-learn with hyperparameters optimized via Optuna.

    Parameters:
    trial (optuna.trial.Trial): The Optuna trial used for hyperparameter optimization.

    Returns:
    SKDecisionTreeClassifier: A scikit-learn DecisionTreeClassifier instance with hyperparameters set based on the trial.
    """
    dtree_max_depth = trial.suggest_int("dtree_max_depth", 5, 15)
    dtree_min_samples_leaf = trial.suggest_int("dtree_min_samples_leaf", 5, 25)
    return SKDecisionTreeClassifier(
        max_depth=dtree_max_depth,
        min_samples_leaf=dtree_min_samples_leaf,
        random_state=12345
    )

def create_sas_dtree(trial):
    """
    Creates a Decision Tree classifier using sasviya with hyperparameters optimized via Optuna.

    Parameters:
    trial (optuna.trial.Trial): The Optuna trial used for hyperparameter optimization.

    Returns:
    SASDecisionTreeClassifier: A SAS Viya DecisionTreeClassifier instance with hyperparameters set based on the trial.
    """
    dtree_max_depth = trial.suggest_int("dtree_max_depth", 5, 15)
    dtree_min_samples_leaf = trial.suggest_int("dtree_min_samples_leaf", 5, 25)
    return SASDecisionTreeClassifier(
        max_depth=dtree_max_depth,
        min_samples_leaf=dtree_min_samples_leaf
    )

def create_sklearn_gb(trial):
    """
    Creates a Gradient Boosting classifier using scikit-learn with hyperparameters optimized via Optuna.

    Parameters:
    trial (optuna.trial.Trial): The Optuna trial used for hyperparameter optimization.

    Returns:
    SKGradientBoostingClassifier: A scikit-learn GradientBoostingClassifier instance with hyperparameters set based on the trial.
    """
    gb_n_estimators = trial.suggest_int("gb_n_estimators", 20, 100)
    gb_max_depth = trial.suggest_int("gb_max_depth", 3, 10)
    gb_min_samples_leaf = trial.suggest_int("gb_min_samples_leaf", 5, 30)
    gb_n_bins = trial.suggest_int("gb_n_bins", 20, 50)
    gb_subsample = trial.suggest_float("gb_subsample", 0.4, 0.8)
    gb_learning_rate = trial.suggest_categorical("gb_learning_rate", [0.01, 0.1, 1.0])
    return SKGradientBoostingClassifier(
        n_estimators=gb_n_estimators,
        max_depth=gb_max_depth,
        min_samples_leaf=gb_min_samples_leaf,
        subsample=gb_subsample,
        learning_rate=gb_learning_rate,
        random_state=12345
    )

def create_sas_gb(trial):
    """
    Creates a Gradient Boosting classifier using sasviya with hyperparameters optimized via Optuna.

    Parameters:
    trial (optuna.trial.Trial): The Optuna trial used for hyperparameter optimization.

    Returns:
    SASGradientBoostingClassifier: A SAS Viya GradientBoostingClassifier instance with hyperparameters set based on the trial.
    """
    gb_n_estimators = trial.suggest_int("gb_n_estimators", 20, 100)
    gb_max_depth = trial.suggest_int("gb_max_depth", 3, 10)
    gb_min_samples_leaf = trial.suggest_int("gb_min_samples_leaf", 5, 30)
    gb_n_bins = trial.suggest_int("gb_n_bins", 20, 50)
    gb_subsample = trial.suggest_float("gb_subsample", 0.4, 0.8)
    gb_learning_rate = trial.suggest_categorical("gb_learning_rate", [0.01, 0.1, 1.0])
    return SASGradientBoostingClassifier(
        n_estimators=gb_n_estimators,
        max_depth=gb_max_depth,
        min_samples_leaf=gb_min_samples_leaf,
        subsample=gb_subsample,
        learning_rate=gb_learning_rate,
        random_state=12345
    )

# Define objective function
class Objective:
    """
    Objective function for hyperparameter optimization with Optuna.

    The objective function optimizes the hyperparameters of different classifiers (Random Forest, Decision Tree, Gradient Boosting)
    from either scikit-learn or SAS Viya. The function also handles pre-processing, cross-validation, and resampling.

    Parameters:
    X (pandas.DataFrame): The features dataset used for model training and validation.
    y (pandas.Series): The target variable dataset used for model training and validation.
    preprocessor (sklearn.pipeline.Pipeline): The preprocessor for the data, typically used for scaling or transformation.
    """
    def __init__(self, X, y, preprocessor):
        self.X = X
        self.y = y
        self.preprocessor = preprocessor

    def __call__(self, trial):
        # classifiers = ["lr", "dtree", "rf", "gb", "svc"]
        # classifiers = ["lr", "dtree", "rf", "gb"]
        classifiers = ["rf", "dtree", "gb"]
        # libraries = ["sasviya", "sklearn"]
        libraries = ["sklearn", "sasviya"]
        resamplings = [True, False]
        classifier_name = trial.suggest_categorical("classifier", classifiers)
        library_name = trial.suggest_categorical("library", libraries)
        resampling = trial.suggest_categorical("resampling", resamplings)

        if resampling:
            sampling_strategy = trial.suggest_float("sampling_strategy", 0.15, 0.5)
            rus = RandomUnderSampler(random_state=12345, sampling_strategy=sampling_strategy)
        
        if classifier_name == "rf" and library_name == "sklearn":
            classifier_obj = create_sklearn_rf(trial)
        elif classifier_name == "rf" and library_name == "sasviya":
            classifier_obj = create_sas_rf(trial)
        elif classifier_name == "dtree" and library_name == "sklearn":
            classifier_obj = create_sklearn_dtree(trial)
        elif classifier_name == "dtree" and library_name == "sasviya":
            classifier_obj = create_sas_dtree(trial)
        elif classifier_name == "gb" and library_name == "sklearn":
            classifier_obj = create_sklearn_gb(trial)
        elif classifier_name == "gb" and library_name == "sasviya":
            classifier_obj = create_sas_gb(trial)
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=12345)
        auc_scores = []

        trial_pipeline_steps = []
        # Custom missing value imputation transformer
        trial_pipeline_steps.append(('imputer', MissingValueImputer()))
        # Add the custom categorical level filter
        trial_pipeline_steps.append(('cat_filter', CategoricalLevelFilter()))
        if library_name == "sklearn":
            trial_pipeline_steps.append(('preprocessor', self.preprocessor))
        # Model
        trial_pipeline_steps.append(('model', classifier_obj))
        trial_pipeline = Pipeline(trial_pipeline_steps)

        for train_index, val_index in cv.split(self.X, self.y):
            X_train, X_val = self.X.iloc[train_index, :], self.X.iloc[val_index, :]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

            # Apply random undersampling
            if resampling:
                X_train, y_train = rus.fit_resample(X_train, y_train)

            # Standardization (only for numeric columns)
            # if numeric_cols is not None:
            #     steps.append(('scaler', DataScaler(numeric_cols)))
            
            if library_name == "sasviya":
                cat_cols = X_train.select_dtypes(exclude='number').columns
                trial_pipeline.fit(X_train, y_train, model__nominals=cat_cols)
            elif library_name == "sklearn":
                 trial_pipeline.fit(X_train, y_train)

            cat_levels = trial_pipeline.named_steps['cat_filter'].levels
            valid_rows_mask = X_val[list(cat_levels.keys())].apply(lambda col: col.isin(cat_levels[col.name])).all(axis=1)
            y_val = y_val[valid_rows_mask]
        
            # Compute AUC scores
            _, auc_val = compute_auc(trial_pipeline, X_train, y_train, X_val, y_val)
            auc_scores.append(auc_val)

        return np.mean(auc_scores)


def create_best_pipeline(best_params, preprocessor):
    """
    Creates a machine learning pipeline using the best parameters found through hyperparameter tuning.

    The function selects the appropriate classifier from SAS Viya or scikit-learn based on the given parameters.
    It then constructs a pipeline with preprocessing steps.

    Parameters:
    best_params : dict
        Dictionary containing the best hyperparameters, including:
        - "classifier": The chosen classifier ("rf" for Random Forest, "dtree" for Decision Tree, "gb" for Gradient Boosting).
        - "library": The ML library ("sasviya" or "sklearn").
        - "resampling": Boolean indicating whether to apply resampling.
        - "sampling_strategy": If resampling is enabled, specifies the strategy for undersampling.
    preprocessor : ColumnTransformer
        Preprocessing pipeline for categorical and numerical features (used for scikit-learn models).

    Returns:
    Pipeline
        A pipeline containing preprocessing steps and the selected classifier.

    Raises:
    ValueError
        If the specified classifier is not supported.
    """
    classifiers = {
        "sasviya": {"rf": create_sas_rf,
                    "dtree": create_sas_dtree,
                    "gb": create_sas_gb},
        "sklearn": {"rf": create_sklearn_rf,
                    "dtree": create_sklearn_dtree,
                    "gb": create_sklearn_gb},
    }

    # Extract the best parameters
    classifier_name = best_params["classifier"]
    library_name = best_params["library"]
    resampling = best_params["resampling"]

    # Create the classifier using the best parameters
    library_classifiers = classifiers[library_name]
    if classifier_name in library_classifiers:
        classifier_obj = library_classifiers[classifier_name](optuna.trial.FixedTrial(best_params))
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    # Create the pipeline steps
    pipeline_steps = []
    pipeline_steps.append(('imputer', MissingValueImputer()))
    pipeline_steps.append(('cat_filter', CategoricalLevelFilter()))

    if library_name == "sklearn":
        pipeline_steps.append(('preprocessor', preprocessor))

    pipeline_steps.append(('model', classifier_obj))
    
    pipeline = Pipeline(pipeline_steps)

    return pipeline

# def create_best_pipeline(best_params, preprocessor):
#     """
#     Creates a machine learning pipeline using the best parameters found through hyperparameter tuning.

    
#     """
#     classifiers = {
#         "sasviya": {"rf": create_sas_rf,
#                     "dtree": create_sas_dtree,
#                     "gb": create_sas_gb},
#         "sklearn": {"rf": create_sklearn_rf,
#                     "dtree": create_sklearn_dtree,
#                     "gb": create_sklearn_gb},
#     }

#     # Extract the best parameters
#     classifier_name = best_params["classifier"]
#     library_name = best_params["library"]

#     # Create the classifier using the best parameters
#     library_classifiers = classifiers.get(library_name, {})
#     if classifier_name in library_classifiers:
#         classifier_obj = library_classifiers[classifier_name](optuna.trial.FixedTrial(best_params))
#     else:
#         raise ValueError(f"Unsupported classifier: {classifier_name}")

#     # Create the pipeline steps
#     pipeline_steps = [('imputer', MissingValueImputer()), ('cat_filter', CategoricalLevelFilter())]

#     if library_name == "sklearn":
#         pipeline_steps.append(('preprocessor', preprocessor))

#     pipeline_steps.append(('model', classifier_obj))
    
#     return Pipeline(pipeline_steps)


def train_pipeline(pipeline, X, y, best_params):
    """
    Trains the given pipeline on the provided dataset, applying resampling if needed.

    Parameters:
    pipeline : Pipeline
        The machine learning pipeline to train.
    X : pd.DataFrame
        Feature matrix used for training.
    y : pd.Series
        Target variable.
    best_params : dict
        Dictionary containing additional hyperparameters, including:
        - "resampling": Boolean indicating whether to apply resampling.
        - "sampling_strategy": If resampling is enabled, specifies the strategy for undersampling.

    Returns:
    Pipeline
        The trained pipeline.
    """
    resampling = best_params["resampling"]

    # Apply resampling if specified
    if resampling:
        sampling_strategy = best_params["sampling_strategy"]
        rus = RandomUnderSampler(random_state=12345, sampling_strategy=sampling_strategy)
        X, y = rus.fit_resample(X, y)

    # Train the pipeline
    library_name = best_params["library"]
    if library_name == "sasviya":
        cat_cols = X.select_dtypes(exclude='number').columns
        pipeline.fit(X, y, model__nominals=cat_cols)
    elif library_name == "sklearn":
        pipeline.fit(X, y)

    return pipeline

def optimize_discount_strategy(y, preds, margins, event_value='Yes', discount_rate=0.1, discount_efficiency=0.8):
    """
    Evaluates a retention strategy by optimizing the probability threshold for predicting customer churn. 
    The function calculates the expected profit for different probability cutoffs and identifies the optimal 
    threshold that maximizes financial impact.

    Parameters:
    y : array-like
        True labels indicating whether a customer churned or not.
    preds : array-like
        Predicted churn probabilities for each customer.
    margins : array-like
        Monthly profit contribution of each customer.
    event_value : str, default='Yes'
        Value in `y` representing churned customers.
    discount_rate : float, default=0.1
        Discount percentage applied to prevent churn (as a fraction, e.g., 0.1 for 10%).
    discount_efficiency : float, default=0.8
        Effectiveness of the discount in preventing churn (as a fraction, e.g., 0.8 for 80%).

    Returns:
    None
        Displays the optimal probability threshold, estimated financial impact, and plots the profit function.

    Notes:
    ------
    - The function assumes that applying a discount to high-risk customers can prevent a portion of them from 
      churning based on `discount_efficiency`.
    - The optimization is based on maximizing profit by balancing prevented losses with the cost of applied discounts.
    """
    def find_best_p(preds, discount_rate, discount_efficiency, granularity=200):
        """Finds the best probability cutoff to maximize profit."""
        probas = np.linspace(0, 1, granularity)
        output_profits = np.zeros(len(probas))
        actual_churners = (y == event_value)

        # Gains: Total potential profit assuming no customers churn
        gains = np.sum(margins) * 12  # Annual profit assumption

        # Loss: Profit lost due to actual churners
        loss = np.sum(margins[actual_churners]) * 12

        for i, p in enumerate(probas):
            predicted_churners = preds>=p

            # Prevented Loss: Profit saved from targeted churners who actually would have churned
            prevented_loss = np.sum(margins[predicted_churners & actual_churners]) * 12 * discount_efficiency
            
            # Discount Loss (Non-Churners): Cost of giving discounts to those who wouldn't have churned
            discount_loss_non_churners = np.sum(margins[predicted_churners & ~actual_churners]) * 12 * discount_rate
            
            # Discount Loss (Churners): Cost of giving discounts to those who would've churned without it
            discount_loss_churners = prevented_loss * discount_rate
            
            # Total Discount Loss
            discount_loss = discount_loss_non_churners + discount_loss_churners
            
            # Compute Profit
            output_profits[i] = gains - loss + prevented_loss - discount_loss

        best_profit = max(output_profits)
        best_p_index = np.argmax(output_profits)
        best_p = probas[best_p_index]
        
        # Baseline profit without intervention
        worst_profit = gains - loss

        return probas, output_profits, best_p, best_profit, worst_profit
    
    # Find optimal cutoff and profit
    probas, output_profits, best_p, best_profit, worst_profit = find_best_p(preds, discount_rate, discount_efficiency, 200)
    
    # Calculate additional metrics
    num_customers_best_p = np.sum(preds >= best_p)
    total_customers = len(y)
    
    print(f"Optimal threshold: {round(best_p, 4)}")
    print(f"At this threshold, apply a {discount_rate * 100}% discount to identified high-risk customers.")
    print(f"Estimated financial impact: {round(best_profit - worst_profit, 2)} € ({round(100 * (best_profit - worst_profit) / worst_profit, 2)}% increase in profit compared to no intervention).")
    print(f"{num_customers_best_p} out of {total_customers} customers ({round(100 * num_customers_best_p / total_customers, 2)}%) would receive the discount.")
    
    plt.plot(probas, output_profits)
    plt.xlabel("Probability Cutoff")
    plt.ylabel("Annual Profit (€)")
    plt.axvline(x=best_p, color='red', linestyle='--', label=f'Optimal p* = {round(best_p, 4)}')
    plt.legend()
    plt.show()