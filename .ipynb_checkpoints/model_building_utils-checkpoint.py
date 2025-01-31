import numpy as np
import pandas as pd

# SAS models
from sasviya.ml.linear_model import LogisticRegression as SASLogisticRegression
from sasviya.ml.tree import DecisionTreeClassifier as SASDecisionTreeClassifier
from sasviya.ml.tree import DecisionTreeRegressor as SASDecisionTreeRegressor
from sasviya.ml.tree import ForestClassifier as SASForestClassifier
from sasviya.ml.tree import GradientBoostingClassifier as SASGradientBoostingClassifier

# scikit-learn classes and models
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier
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