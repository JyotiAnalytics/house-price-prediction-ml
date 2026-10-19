# ============================================================
# HOUSE PRICE PREDICTION
# COMPLETE DATA SCIENCE PROJECT
# WINDOWS SAFE VERSION
# ============================================================

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

warnings.filterwarnings("ignore")

np.random.seed(42)
sns.set_theme(style="whitegrid")


# ============================================================
# 1. PROJECT SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_PATH = os.path.join(
    BASE_DIR,
    "House Price Prediction Dataset.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CHART_DIR = os.path.join(
    OUTPUT_DIR,
    "charts"
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("=" * 70)
print("HOUSE PRICE PREDICTION")
print("COMPLETE DATA SCIENCE PROJECT")
print("=" * 70)


# ============================================================
# 2. IMPORT ACTUAL DATASET
# ============================================================

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {FILE_PATH}\n"
        "Place the CSV file beside house_price_ml.py"
    )

df = pd.read_csv(FILE_PATH)

if df.empty:
    raise ValueError("Dataset is empty.")

print("\nDATASET LOADED SUCCESSFULLY")

print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nOriginal columns:")
print(df.columns.tolist())


# ============================================================
# 3. STANDARDIZE COLUMN NAMES
# ============================================================

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

if df.columns.duplicated().any():
    raise ValueError(
        "Duplicate column names after standardization."
    )

print("\nStandardized columns:")
print(df.columns.tolist())


# ============================================================
# 4. TARGET COLUMN DETECTION
# ============================================================

target_candidates = [
    "price",
    "house_price",
    "sale_price",
    "selling_price",
    "property_price",
    "price_in_usd",
    "price_in_inr",
    "price_inr",
    "price_in_lakhs",
    "price_lakhs",
    "median_house_value",
    "target"
]

target = next(
    (
        column
        for column in target_candidates
        if column in df.columns
    ),
    None
)

if target is None:
    raise ValueError(
        "Price target column not found.\n"
        f"Available columns: {df.columns.tolist()}\n"
        "Add your actual target name to target_candidates."
    )

print("\nTarget column:", target)


# ============================================================
# 5. DATA QUALITY CHECK
# ============================================================

print("\nDATASET INFORMATION")

df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

df = df.drop_duplicates().copy()

print("\nShape after removing duplicates:")
print(df.shape)


# ============================================================
# 6. DATA CLEANING
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

categorical_columns = df.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

for column in categorical_columns:

    df[column] = df[column].apply(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )

# Convert target into numeric format.

if not pd.api.types.is_numeric_dtype(df[target]):

    df[target] = (
        df[target]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(
            r"[₹$£€]",
            "",
            regex=True
        )
        .str.strip()
    )

df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)

df[target] = df[target].replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(subset=[target]).copy()

df = df[df[target] > 0].copy()

if len(df) < 10:
    raise ValueError(
        "At least 10 valid rows are recommended "
        "for this training and evaluation workflow."
    )

print("\nCleaned dataset shape:")
print(df.shape)

df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cleaned_dataset.csv"
    ),
    index=False
)


# ============================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================

print("\nDESCRIPTIVE STATISTICS")

description = df.describe(include="all").T

print(description)

description.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "descriptive_statistics.csv"
    )
)

print("\nMean:", df[target].mean())
print("Median:", df[target].median())
print("Standard deviation:", df[target].std())
print("Minimum:", df[target].min())
print("Maximum:", df[target].max())
print("Skewness:", df[target].skew())
print("Kurtosis:", df[target].kurt())


# ============================================================
# 8. PROBABILITY ANALYSIS
# ============================================================

price_threshold = df[target].quantile(0.75)

probability = (
    df[target] > price_threshold
).mean()

print("\nPROBABILITY ANALYSIS")

print("75th percentile:", price_threshold)

print(
    "Probability above 75th percentile:",
    round(probability, 4)
)


# ============================================================
# 9. PRICE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x=target,
    kde=True
)

plt.title("House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "price_distribution.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 10. PRICE BOXPLOT
# ============================================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x=target
)

plt.title("House Price Boxplot")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "price_boxplot.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 11. MISSING VALUE ANALYSIS
# ============================================================

missing_report = pd.DataFrame({

    "Missing_Count": df.isnull().sum(),

    "Missing_Percentage": (
        df.isnull().mean() * 100
    ).round(2)

})

print("\nMISSING VALUE REPORT")
print(missing_report)

missing_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "missing_values_report.csv"
    )
)


# ============================================================
# 12. CORRELATION ANALYSIS
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_columns) >= 2:

    correlation_matrix = df[numeric_columns].corr()

    correlation_matrix.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "correlation_matrix.csv"
        )
    )

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "correlation_heatmap.png"
        ),
        dpi=150
    )

    plt.close()

    price_correlations = (
        correlation_matrix[target]
        .drop(target)
        .sort_values(
            key=lambda values: values.abs(),
            ascending=False
        )
    )

    print("\nPRICE CORRELATIONS")
    print(price_correlations)

    price_correlations.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "price_correlations.csv"
        )
    )


# ============================================================
# 13. OUTLIER DETECTION USING IQR
# ============================================================

outlier_rows = []

for column in numeric_columns:

    values = df[column].dropna()

    if len(values) < 4:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = (
        (values < lower_bound)
        | (values > upper_bound)
    ).sum()

    outlier_rows.append({

        "Column": column,

        "Lower_Bound": lower_bound,

        "Upper_Bound": upper_bound,

        "Outlier_Count": int(outlier_count)

    })

outlier_report = pd.DataFrame(outlier_rows)

print("\nOUTLIER REPORT")
print(outlier_report)

outlier_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "outlier_report.csv"
    ),
    index=False
)

# Outliers are reported, not automatically deleted.


# ============================================================
# 14. HYPOTHESIS TESTING
# ============================================================

location_candidates = [
    "location",
    "city",
    "locality",
    "neighborhood",
    "area_name"
]

location_column = next(
    (
        column
        for column in location_candidates
        if column in df.columns
    ),
    None
)

if location_column is not None:

    groups = df[location_column].dropna().unique()

    if len(groups) >= 2:

        group_a = df.loc[
            df[location_column] == groups[0],
            target
        ].dropna()

        group_b = df.loc[
            df[location_column] == groups[1],
            target
        ].dropna()

        if len(group_a) >= 2 and len(group_b) >= 2:

            t_statistic, p_value = stats.ttest_ind(
                group_a,
                group_b,
                equal_var=False
            )

            print("\nWELCH T-TEST")

            print("Group A:", groups[0])
            print("Group B:", groups[1])

            print("Mean A:", group_a.mean())
            print("Mean B:", group_b.mean())

            print("T-statistic:", t_statistic)
            print("P-value:", p_value)

            if pd.notna(p_value):

                if p_value < 0.05:
                    print(
                        "Statistically significant difference."
                    )
                else:
                    print(
                        "Insufficient evidence of a difference."
                    )

            else:
                print(
                    "P-value unavailable for these groups."
                )

        else:
            print(
                "\nHypothesis test skipped: "
                "Not enough observations per group."
            )

    else:
        print(
            "\nHypothesis test skipped: "
            "Fewer than two location groups."
        )

else:
    print(
        "\nHypothesis test skipped: "
        "Location column unavailable."
    )


# ============================================================
# 15. FEATURE SELECTION
# ============================================================

id_candidates = [
    "id",
    "house_id",
    "property_id",
    "listing_id",
    "serial_number"
]

id_columns = [
    column
    for column in id_candidates
    if column in df.columns
]

X = df.drop(
    columns=[target] + id_columns
).copy()

y = df[target].copy()

# Remove completely empty columns.

empty_columns = X.columns[
    X.isna().all()
].tolist()

if empty_columns:

    print("\nRemoving empty columns:")
    print(empty_columns)

    X = X.drop(columns=empty_columns)

# Convert mostly numeric text columns to numbers.

for column in X.columns:

    if X[column].dtype == "object":

        converted = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        non_missing_count = X[column].notna().sum()

        if non_missing_count > 0:

            numeric_ratio = (
                converted.notna().sum()
                / non_missing_count
            )

            if numeric_ratio >= 0.90:
                X[column] = converted

# Remove raw date columns.

date_columns = [
    column
    for column in X.columns
    if "date" in column.lower()
]

if date_columns:

    print("\nRemoving raw date columns:")
    print(date_columns)

    X = X.drop(columns=date_columns)

# Convert remaining categorical columns to object strings.

for column in X.columns:

    if not pd.api.types.is_numeric_dtype(X[column]):

        X[column] = X[column].astype("object")

        X[column] = X[column].where(
            X[column].notna(),
            np.nan
        )

        X[column] = X[column].apply(
            lambda value: str(value)
            if pd.notna(value)
            else np.nan
        )

if X.shape[1] == 0:
    raise ValueError("No usable features found.")

print("\nFEATURE COLUMNS")
print(X.columns.tolist())

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 16. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAIN TEST SPLIT")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 17. PREPROCESSING PIPELINE
# ============================================================

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = [
    column
    for column in X_train.columns
    if column not in numeric_features
]

transformers = []

if numeric_features:

    numeric_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        ),

        (
            "scaler",
            StandardScaler()
        )

    ])

    transformers.append((
        "numeric",
        numeric_pipeline,
        numeric_features
    ))

if categorical_features:

    categorical_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
                keep_empty_features=True
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )

    ])

    transformers.append((
        "categorical",
        categorical_pipeline,
        categorical_features
    ))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

print("\nPREPROCESSING PIPELINE CREATED")


# ============================================================
# 18. DEFINE FIVE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    )

}

print("\nFIVE REGRESSION MODELS INITIALIZED")


# ============================================================
# 19. MODEL TRAINING AND EVALUATION
# ============================================================

results = []
trained_models = {}

for model_name, model in models.items():

    print("\n" + "-" * 60)
    print("TRAINING:", model_name)
    print("-" * 60)

    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )

    ])

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({

        "Model": model_name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2

    })

    trained_models[model_name] = pipeline

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 4))


# ============================================================
# 20. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE",
    ascending=True
)

print("\nMODEL COMPARISON")

print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)

best_baseline_name = results_df.iloc[0]["Model"]

print("\nBest baseline model:")
print(best_baseline_name)


# ============================================================
# 21. CROSS VALIDATION - WINDOWS SAFE
# ============================================================

print("\n" + "=" * 70)
print("CROSS VALIDATION STARTED")
print("=" * 70)

cv_splits = min(5, len(X_train))

if cv_splits < 2:
    raise ValueError(
        "Not enough training records for cross-validation."
    )

cv = KFold(
    n_splits=cv_splits,
    shuffle=True,
    random_state=42
)

cv_rows = []

for model_name, model in models.items():

    print("\nCross-validating:", model_name)

    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )

    ])

    scores = cross_val_score(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        error_score="raise"
    )

    rmse_scores = -scores

    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()

    cv_rows.append({

        "Model": model_name,

        "Mean_CV_RMSE": mean_rmse,

        "Std_CV_RMSE": std_rmse

    })

    print("Fold RMSE:", rmse_scores)

    print(
        "Mean CV RMSE:",
        round(mean_rmse, 2)
    )

    print(
        "Standard Deviation:",
        round(std_rmse, 2)
    )

cv_results = pd.DataFrame(cv_rows)

cv_results = cv_results.sort_values(
    "Mean_CV_RMSE",
    ascending=True
)

print("\nCROSS VALIDATION RESULTS")

print(cv_results.to_string(index=False))

cv_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cross_validation_results.csv"
    ),
    index=False
)

print("\nCROSS VALIDATION COMPLETED")


# ============================================================
# 22. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING STARTED")
print("=" * 70)

tuning_pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        RandomForestRegressor(
            random_state=42,
            n_jobs=1
        )
    )

])

param_distributions = {

    "model__n_estimators": [
        100, 150, 200, 300
    ],

    "model__max_depth": [
        None, 10, 15, 20, 25
    ],

    "model__min_samples_split": [
        2, 5, 10
    ],

    "model__min_samples_leaf": [
        1, 2, 4
    ],

    "model__max_features": [
        0.7, 1.0
    ]

}

search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=param_distributions,
    n_iter=12,
    scoring="neg_root_mean_squared_error",
    cv=min(3, len(X_train)),
    random_state=42,
    n_jobs=1,
    verbose=1,
    error_score="raise"
)

search.fit(
    X_train,
    y_train
)

tuned_model = search.best_estimator_

print("\nBEST PARAMETERS")

print(search.best_params_)

print(
    "\nBest CV RMSE:",
    round(-search.best_score_, 2)
)


# ============================================================
# 23. EVALUATE TUNED MODEL
# ============================================================

tuned_predictions = tuned_model.predict(X_test)

tuned_mae = mean_absolute_error(
    y_test,
    tuned_predictions
)

tuned_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tuned_predictions
    )
)

tuned_r2 = r2_score(
    y_test,
    tuned_predictions
)

print("\nTUNED RANDOM FOREST RESULTS")

print("MAE:", round(tuned_mae, 2))
print("RMSE:", round(tuned_rmse, 2))
print("R2:", round(tuned_r2, 4))

tuned_metrics = pd.DataFrame([{

    "Model": "Tuned Random Forest",

    "MAE": tuned_mae,

    "RMSE": tuned_rmse,

    "R2": tuned_r2,

    "Best_CV_RMSE": -search.best_score_

}])

tuned_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "tuned_model_metrics.csv"
    ),
    index=False
)


# ============================================================
# 24. SAVE FINAL MODEL
# ============================================================

# Select the final model using training cross-validation.
# Do not choose based on test-set performance.

best_cv_row = cv_results.iloc[0]

best_cv_name = best_cv_row["Model"]
best_cv_rmse = best_cv_row["Mean_CV_RMSE"]

tuned_cv_rmse = -search.best_score_

# Note: The tuned model uses 3-fold CV, whereas baseline
# comparison uses up to 5 folds. This is a practical baseline
# comparison, not a perfectly matched model-selection study.

if tuned_cv_rmse < best_cv_rmse:

    final_model = tuned_model
    final_model_name = "Tuned Random Forest"

else:

    final_model_name = best_cv_name

    final_model = trained_models[
        final_model_name
    ]

print("\nFINAL SELECTED MODEL:")
print(final_model_name)

model_path = os.path.join(
    ARTIFACT_DIR,
    "best_house_price_model.joblib"
)

joblib.dump(
    final_model,
    model_path
)

print("\nMODEL SAVED:")
print(model_path)


# ============================================================
# 25. FINAL MODEL EVALUATION
# ============================================================

final_predictions = final_model.predict(X_test)

final_mae = mean_absolute_error(
    y_test,
    final_predictions
)

final_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_predictions
    )
)

final_r2 = r2_score(
    y_test,
    final_predictions
)

print("\nFINAL MODEL METRICS")

print("Model:", final_model_name)
print("MAE:", round(final_mae, 2))
print("RMSE:", round(final_rmse, 2))
print("R2:", round(final_r2, 4))

final_metrics_df = pd.DataFrame([{

    "Model": final_model_name,

    "MAE": final_mae,

    "RMSE": final_rmse,

    "R2": final_r2

}])

final_metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_model_metrics.csv"
    ),
    index=False
)


# ============================================================
# 26. SAVE EVALUATION REPORT
# ============================================================

report = f"""
HOUSE PRICE PREDICTION REPORT
========================================

Dataset Size: {len(df)}

Target Column: {target}

Features:
{X.columns.tolist()}

Training Records: {len(X_train)}

Testing Records: {len(X_test)}

Best Baseline Model:
{best_baseline_name}

Best Cross-Validation Model:
{best_cv_name}

Tuned Random Forest Parameters:
{search.best_params_}

Tuned Random Forest CV RMSE:
{tuned_cv_rmse:,.2f}

Final Selected Model:
{final_model_name}

Final Test MAE:
{final_mae:,.2f}

Final Test RMSE:
{final_rmse:,.2f}

Final Test R2:
{final_r2:.4f}

IMPORTANT:
The project uses the actual CSV dataset.

Predictions are estimates and do not guarantee
real-world market accuracy.
"""

report_path = os.path.join(
    OUTPUT_DIR,
    "best_model_metrics.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)

print(report)


# ============================================================
# 27. SAVE PREDICTIONS
# ============================================================

prediction_results = X_test.copy()

prediction_results["Actual_Price"] = (
    y_test.to_numpy()
)

prediction_results["Predicted_Price"] = (
    final_predictions
)

prediction_results["Absolute_Error"] = np.abs(
    prediction_results["Actual_Price"]
    - prediction_results["Predicted_Price"]
)

prediction_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "predictions.csv"
    ),
    index=False
)

print("\nPredictions CSV saved.")


# ============================================================
# 28. ACTUAL VS PREDICTED CHART
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    y_test,
    final_predictions,
    alpha=0.6
)

minimum = min(
    y_test.min(),
    final_predictions.min()
)

maximum = max(
    y_test.max(),
    final_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "actual_vs_predicted.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 29. FEATURE IMPORTANCE
# ============================================================

fitted_preprocessor = (
    final_model.named_steps["preprocessor"]
)

fitted_model = (
    final_model.named_steps["model"]
)

if hasattr(
    fitted_model,
    "feature_importances_"
):

    feature_names = (
        fitted_preprocessor.get_feature_names_out()
    )

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": (
            fitted_model.feature_importances_
        )

    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    print("\nTOP 10 IMPORTANT FEATURES")

    print(importance_df.head(10))

    top_features = (
        importance_df.head(15)
        .sort_values("Importance")
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel("Feature Importance")

    plt.title("House Price Feature Importance")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "feature_importance.png"
        ),
        dpi=150
    )

    plt.close()

elif hasattr(
    fitted_model,
    "coef_"
):

    feature_names = (
        fitted_preprocessor.get_feature_names_out()
    )

    coefficients = np.asarray(
        fitted_model.coef_
    ).ravel()

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Coefficient": coefficients,

        "Absolute_Coefficient": np.abs(
            coefficients
        )

    })

    importance_df = importance_df.sort_values(
        "Absolute_Coefficient",
        ascending=False
    )

    importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        ),
        index=False
    )

    print("\nTOP LINEAR MODEL COEFFICIENTS")

    print(importance_df.head(10))

else:

    print(
        "\nFeature importance unavailable "
        "for the selected model."
    )


# ============================================================
# 30. LOAD MODEL AND TEST PREDICTION
# ============================================================

loaded_model = joblib.load(model_path)

sample_house = X_test.iloc[[0]].copy()

sample_prediction = loaded_model.predict(
    sample_house
)[0]

print("\nSAMPLE PREDICTION")

print("\nInput features:")
print(sample_house)

print(
    "\nPredicted price:",
    round(float(sample_prediction), 2)
)

print(
    "Actual price:",
    round(float(y_test.iloc[0]), 2)
)


# ============================================================
# 31. SHAP EXPLAINABILITY
# ============================================================

try:

    import shap

    if not hasattr(
        fitted_model,
        "feature_importances_"
    ):
        print(
            "\nSHAP TreeExplainer skipped: "
            "Final model is not tree-based."
        )

    else:

        X_explain = X_test.sample(
            n=min(50, len(X_test)),
            random_state=42
        )

        X_transformed = (
            fitted_preprocessor.transform(
                X_explain
            )
        )

        if hasattr(
            X_transformed,
            "toarray"
        ):

            X_transformed = (
                X_transformed.toarray()
            )

        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=(
                fitted_preprocessor
                .get_feature_names_out()
            )
        )

        explainer = shap.TreeExplainer(
            fitted_model
        )

        shap_values = explainer(
            X_transformed_df
        )

        shap.summary_plot(
            shap_values,
            X_transformed_df,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                CHART_DIR,
                "shap_summary.png"
            ),
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "\nSHAP summary saved successfully."
        )

except Exception as error:

    print(
        "\nSHAP explanation could not be generated."
    )

    print("Reason:", error)


# ============================================================
# 32. SAVE STREAMLIT FEATURE INFORMATION
# ============================================================

feature_info = pd.DataFrame({

    "Feature": X.columns,

    "Data_Type": [
        str(X[column].dtype)
        for column in X.columns
    ]

})

feature_info.to_csv(
    os.path.join(
        ARTIFACT_DIR,
        "feature_information.csv"
    ),
    index=False
)

# Save sample values to help build matching Streamlit inputs.

X_train.head(20).to_csv(
    os.path.join(
        ARTIFACT_DIR,
        "sample_features.csv"
    ),
    index=False
)

print("\nStreamlit feature information saved.")


# ============================================================
# 33. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("HOUSE PRICE DATA SCIENCE PROJECT COMPLETED")

print("=" * 70)

print("\nCOMPLETED TASKS")

print("1. Actual CSV dataset imported")
print("2. Column standardization")
print("3. Data quality checks")
print("4. Data cleaning")
print("5. Descriptive statistics")
print("6. Probability analysis")
print("7. Price distribution")
print("8. Price boxplot")
print("9. Missing value analysis")
print("10. Correlation analysis")
print("11. Outlier detection")
print("12. Hypothesis testing where applicable")
print("13. Feature selection")
print("14. Train-test split")
print("15. Preprocessing pipeline")
print("16. Five regression models")
print("17. Model training")
print("18. Model comparison")
print("19. Cross-validation")
print("20. Hyperparameter tuning")
print("21. Final model evaluation")
print("22. Model saving")
print("23. Evaluation report")
print("24. Predictions CSV")
print("25. Actual vs predicted visualization")
print("26. Feature importance where supported")
print("27. Sample prediction")
print("28. SHAP explainability where supported")
print("29. Streamlit feature information")

print("\nFINAL MODEL:")
print(final_model_name)

print("\nMODEL PATH:")
print(model_path)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nPROJECT FINISHED SUCCESSFULLY!")