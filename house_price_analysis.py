
# ============================================================
# HOUSE PRICE PREDICTION DATASET
# COMPLETE DATA ANALYST PROJECT
# Python | NumPy | Pandas | Matplotlib | Seaborn | Statistics
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

FILE_PATH = "House Price Prediction Dataset.csv"

OUTPUT_DIR = "outputs"

CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

print("=" * 70)
print("HOUSE PRICE PREDICTION - DATA ANALYSIS PROJECT")
print("=" * 70)


# ============================================================
# 3. IMPORT DATASET
# ============================================================

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {FILE_PATH}\n"
        "Keep the CSV file in the same folder as this Python script."
    )

df = pd.read_csv(FILE_PATH)

print("\n1. DATASET IMPORTED SUCCESSFULLY")
print("-" * 50)

print("Dataset Shape:", df.shape)
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\n2. DATASET INFORMATION")
print("-" * 50)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all").T)


# ============================================================
# 5. STANDARDIZE COLUMN NAMES
# ============================================================

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

print("\n3. STANDARDIZED COLUMN NAMES")
print("-" * 50)

print(df.columns.tolist())


# ============================================================
# 6. IDENTIFY IMPORTANT COLUMNS
# ============================================================

def find_column(dataframe, candidates):
    """
    Find the first matching column from a list of candidates.
    """

    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


price_col = find_column(
    df,
    [
        "price",
        "house_price",
        "sale_price",
        "selling_price",
        "property_price",
        "price_in_usd",
        "price_inr"
    ]
)

area_col = find_column(
    df,
    [
        "area",
        "area_sqft",
        "square_feet",
        "square_feet_area",
        "sqft",
        "living_area",
        "size",
        "house_area"
    ]
)

bedroom_col = find_column(
    df,
    [
        "bedrooms",
        "bedroom",
        "beds",
        "bhk",
        "number_of_bedrooms"
    ]
)

bathroom_col = find_column(
    df,
    [
        "bathrooms",
        "bathroom",
        "baths",
        "number_of_bathrooms"
    ]
)

location_col = find_column(
    df,
    [
        "location",
        "city",
        "locality",
        "neighborhood",
        "area_name"
    ]
)

condition_col = find_column(
    df,
    [
        "condition",
        "house_condition",
        "property_condition"
    ]
)

print("\n4. IMPORTANT COLUMN IDENTIFICATION")
print("-" * 50)

print("Price Column:", price_col)
print("Area Column:", area_col)
print("Bedroom Column:", bedroom_col)
print("Bathroom Column:", bathroom_col)
print("Location Column:", location_col)
print("Condition Column:", condition_col)

if price_col is None:
    raise ValueError(
        "Price column was not identified automatically.\n"
        f"Available columns: {df.columns.tolist()}\n"
        "Add your actual price column name to the candidates list."
    )


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

print("\n5. MISSING VALUE ANALYSIS")
print("-" * 50)

missing_values = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
).round(2)

missing_report = pd.DataFrame({
    "Missing_Count": missing_values,
    "Missing_Percentage": missing_percentage
})

missing_report = missing_report.sort_values(
    by="Missing_Count",
    ascending=False
)

print(missing_report)

missing_report.to_csv(
    os.path.join(OUTPUT_DIR, "missing_values_report.csv")
)


# Missing value heatmap

plt.figure(figsize=(12, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False
)

plt.title("Missing Values Heatmap")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "missing_values_heatmap.png"),
    dpi=300
)

plt.close()


# ============================================================
# 8. DUPLICATE ANALYSIS
# ============================================================

print("\n6. DUPLICATE ANALYSIS")
print("-" * 50)

duplicate_count = df.duplicated().sum()

print("Duplicate Rows:", duplicate_count)

df = df.drop_duplicates().copy()

print("Shape After Removing Duplicates:", df.shape)


# ============================================================
# 9. DATA CLEANING
# ============================================================

print("\n7. DATA CLEANING")
print("-" * 50)

# Replace infinite numeric values with NaN

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove unnecessary spaces from text columns

categorical_columns = df.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

for column in categorical_columns:
    df[column] = df[column].apply(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )

# Remove rows where target price is missing

missing_price_count = df[price_col].isna().sum()

print("Rows with Missing Price:", missing_price_count)

df = df.dropna(subset=[price_col]).copy()

# Convert price to numeric where possible

df[price_col] = pd.to_numeric(
    df[price_col],
    errors="coerce"
)

df = df.dropna(subset=[price_col]).copy()

# Remove impossible non-positive prices

invalid_price_count = (df[price_col] <= 0).sum()

print("Rows with Non-positive Price:", invalid_price_count)

df = df[df[price_col] > 0].copy()

print("Cleaned Dataset Shape:", df.shape)


# Save cleaned dataset

df.to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_house_price_data.csv"),
    index=False
)

print("Cleaned dataset saved successfully.")


# ============================================================
# 10. NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

print("\n8. COLUMN CATEGORIZATION")
print("-" * 50)

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print("Numerical Columns:")
print(numeric_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================================
# 11. DESCRIPTIVE STATISTICS
# ============================================================

print("\n9. DESCRIPTIVE STATISTICS")
print("-" * 50)

print("\nMean:")
print(df[numeric_columns].mean())

print("\nMedian:")
print(df[numeric_columns].median())

print("\nMode:")
print(df[numeric_columns].mode().head())

print("\nStandard Deviation:")
print(df[numeric_columns].std())

print("\nVariance:")
print(df[numeric_columns].var())

print("\nMinimum Values:")
print(df[numeric_columns].min())

print("\nMaximum Values:")
print(df[numeric_columns].max())

print("\n25th Percentile:")
print(df[numeric_columns].quantile(0.25))

print("\n50th Percentile:")
print(df[numeric_columns].quantile(0.50))

print("\n75th Percentile:")
print(df[numeric_columns].quantile(0.75))


# Save descriptive statistics

df[numeric_columns].describe().T.to_csv(
    os.path.join(OUTPUT_DIR, "descriptive_statistics.csv")
)


# ============================================================
# 12. PRICE DISTRIBUTION
# ============================================================

print("\n10. HOUSE PRICE DISTRIBUTION")
print("-" * 50)

print("Average Price:", df[price_col].mean())

print("Median Price:", df[price_col].median())

print("Minimum Price:", df[price_col].min())

print("Maximum Price:", df[price_col].max())

print("Price Standard Deviation:", df[price_col].std())

print("Price Variance:", df[price_col].var())

print("Price Skewness:", df[price_col].skew())

print("Price Kurtosis:", df[price_col].kurt())


# Histogram

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x=price_col,
    kde=True
)

plt.title("House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "price_distribution.png"),
    dpi=300
)

plt.close()


# Boxplot

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x=price_col
)

plt.title("House Price Boxplot")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "price_boxplot.png"),
    dpi=300
)

plt.close()


# ============================================================
# 13. AREA ANALYSIS
# ============================================================

if area_col is not None:

    print("\n11. AREA ANALYSIS")
    print("-" * 50)

    df[area_col] = pd.to_numeric(
        df[area_col],
        errors="coerce"
    )

    print("Average Area:", df[area_col].mean())

    print("Median Area:", df[area_col].median())

    print("Minimum Area:", df[area_col].min())

    print("Maximum Area:", df[area_col].max())

    # Area versus Price

    plot_data = df[[area_col, price_col]].dropna()

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=plot_data,
        x=area_col,
        y=price_col
    )

    plt.title("House Area vs House Price")

    plt.tight_layout()

    plt.savefig(
        os.path.join(CHART_DIR, "area_vs_price.png"),
        dpi=300
    )

    plt.close()

    correlation = df[area_col].corr(df[price_col])

    print("Area-Price Correlation:", correlation)


# ============================================================
# 14. BEDROOM ANALYSIS
# ============================================================

if bedroom_col is not None:

    print("\n12. BEDROOM ANALYSIS")
    print("-" * 50)

    print("Bedroom Counts:")

    print(df[bedroom_col].value_counts().sort_index())

    bedroom_price = (
        df.groupby(bedroom_col)[price_col]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_index()
    )

    print("\nPrice by Bedroom Count:")

    print(bedroom_price)

    bedroom_price.to_csv(
        os.path.join(OUTPUT_DIR, "price_by_bedrooms.csv")
    )

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x=bedroom_col,
        y=price_col
    )

    plt.title("House Price by Number of Bedrooms")

    plt.tight_layout()

    plt.savefig(
        os.path.join(CHART_DIR, "price_by_bedrooms.png"),
        dpi=300
    )

    plt.close()


# ============================================================
# 15. BATHROOM ANALYSIS
# ============================================================

if bathroom_col is not None:

    print("\n13. BATHROOM ANALYSIS")
    print("-" * 50)

    bathroom_price = (
        df.groupby(bathroom_col)[price_col]
        .agg(["count", "mean", "median"])
        .sort_index()
    )

    print(bathroom_price)

    bathroom_price.to_csv(
        os.path.join(OUTPUT_DIR, "price_by_bathrooms.csv")
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x=bathroom_col,
        y=price_col,
        errorbar=None
    )

    plt.title("Average House Price by Bathrooms")

    plt.tight_layout()

    plt.savefig(
        os.path.join(CHART_DIR, "price_by_bathrooms.png"),
        dpi=300
    )

    plt.close()

# ============================================================
# 17. CONDITION ANALYSIS
# ============================================================

if condition_col is not None:

    print("\n15. HOUSE CONDITION ANALYSIS")
    print("-" * 50)

    condition_analysis = (
        df.groupby(condition_col)[price_col]
        .agg(["count", "mean", "median"])
        .sort_values("mean", ascending=False)
    )

    print(condition_analysis)

    condition_analysis.to_csv(
        os.path.join(OUTPUT_DIR, "condition_price_analysis.csv")
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        x=condition_col,
        y=price_col,
        errorbar=None
    )

    plt.title("Average House Price by Condition")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        os.path.join(CHART_DIR, "condition_vs_price.png"),
        dpi=300
    )

    plt.close()


# ============================================================
# 18. CORRELATION ANALYSIS
# ============================================================

print("\n16. CORRELATION ANALYSIS")
print("-" * 50)

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

correlation_matrix = df[numeric_columns].corr()

print(correlation_matrix)

correlation_matrix.to_csv(
    os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
)


# Correlation heatmap

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
    os.path.join(CHART_DIR, "correlation_heatmap.png"),
    dpi=300
)

plt.close()


# Features most correlated with price

price_correlations = (
    correlation_matrix[price_col]
    .drop(price_col)
    .sort_values(ascending=False)
)

print("\nFeatures Most Positively Correlated with Price:")

print(price_correlations.head(10))

print("\nFeatures Most Negatively Correlated with Price:")

print(price_correlations.tail(10))

price_correlations.to_csv(
    os.path.join(OUTPUT_DIR, "price_correlations.csv")
)


# ============================================================
# 19. OUTLIER DETECTION USING IQR
# ============================================================

print("\n17. OUTLIER DETECTION")
print("-" * 50)

outlier_results = []

for column in numeric_columns:

    values = df[column].dropna()

    if len(values) < 4:
        continue

    q1 = values.quantile(0.25)

    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr

    upper_bound = q3 + 1.5 * iqr

    outliers = values[
        (values < lower_bound) |
        (values > upper_bound)
    ]

    outlier_results.append({
        "Column": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": len(outliers),
        "Outlier_Percentage": round(
            len(outliers) / len(values) * 100,
            2
        )
    })

outlier_report = pd.DataFrame(outlier_results)

print(outlier_report)

outlier_report.to_csv(
    os.path.join(OUTPUT_DIR, "outlier_report.csv"),
    index=False
)


# ============================================================
# 20. PRICE DISTRIBUTION BY NUMERICAL FEATURES
# ============================================================

print("\n18. NUMERICAL FEATURE VISUALIZATION")
print("-" * 50)

for column in numeric_columns:

    if column == price_col:
        continue

    plot_data = df[[column, price_col]].dropna()

    if len(plot_data) < 2:
        continue

    plt.figure(figsize=(9, 5))

    sns.scatterplot(
        data=plot_data,
        x=column,
        y=price_col
    )

    plt.title(f"{column} vs {price_col}")

    plt.tight_layout()

    safe_name = column.replace("/", "_")

    plt.savefig(
        os.path.join(
            CHART_DIR,
            f"{safe_name}_vs_price.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 21. STATISTICAL TEST: T-TEST
# ============================================================

print("\n19. STATISTICAL HYPOTHESIS TESTING")
print("-" * 50)

if condition_col is not None:

    condition_groups = df[condition_col].dropna().unique()

    if len(condition_groups) >= 2:

        group_a = df.loc[
            df[condition_col] == condition_groups[0],
            price_col
        ].dropna()

        group_b = df.loc[
            df[condition_col] == condition_groups[1],
            price_col
        ].dropna()

        if len(group_a) >= 2 and len(group_b) >= 2:

            t_statistic, p_value = stats.ttest_ind(
                group_a,
                group_b,
                equal_var=False
            )

            print("Group A:", condition_groups[0])

            print("Group B:", condition_groups[1])

            print("Mean Price - Group A:", group_a.mean())

            print("Mean Price - Group B:", group_b.mean())

            print("T-Statistic:", t_statistic)

            print("P-Value:", p_value)

            alpha = 0.05

            if p_value < alpha:

                print(
                    "Result: Statistically significant difference "
                    "between the two group means."
                )

            else:

                print(
                    "Result: Insufficient evidence of a "
                    "difference between the two group means."
                )

            print(
                "Note: Statistical significance does not prove "
                "that condition causes price differences."
            )

        else:

            print(
                "Not enough observations in both condition groups "
                "to perform the t-test."
            )

    else:

        print(
            "At least two condition groups are required "
            "for this comparison."
        )

else:

    print(
        "Condition column not found. "
        "T-test was skipped."
    )


# ============================================================
# 22. PRICE RANKING
# ============================================================

print("\n20. HOUSE PRICE RANKING")
print("-" * 50)

top_expensive_houses = df.nlargest(
    10,
    price_col
)

print("\nTop 10 Most Expensive Houses:")

print(top_expensive_houses)

top_expensive_houses.to_csv(
    os.path.join(OUTPUT_DIR, "top_10_expensive_houses.csv"),
    index=False
)


cheapest_houses = df.nsmallest(
    10,
    price_col
)

print("\nTop 10 Least Expensive Houses:")

print(cheapest_houses)

cheapest_houses.to_csv(
    os.path.join(OUTPUT_DIR, "top_10_cheapest_houses.csv"),
    index=False
)


# ============================================================
# 23. PRICE SEGMENTATION
# ============================================================

print("\n21. HOUSE PRICE SEGMENTATION")
print("-" * 50)

price_25 = df[price_col].quantile(0.25)

price_75 = df[price_col].quantile(0.75)


def categorize_price(price):

    if price <= price_25:
        return "Low Price"

    elif price <= price_75:
        return "Medium Price"

    return "High Price"


df["price_segment"] = df[price_col].apply(
    categorize_price
)

price_segment_counts = df["price_segment"].value_counts()

print(price_segment_counts)

price_segment_counts.to_csv(
    os.path.join(OUTPUT_DIR, "price_segment_counts.csv")
)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="price_segment",
    order=["Low Price", "Medium Price", "High Price"]
)

plt.title("House Price Segmentation")

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "price_segmentation.png"),
    dpi=300
)

plt.close()


# ============================================================
# 24. BUSINESS INSIGHTS
# ============================================================

print("\n22. BUSINESS INSIGHTS")
print("-" * 50)

print("Total Properties Analyzed:", len(df))

print("Average House Price:", round(df[price_col].mean(), 2))

print("Median House Price:", round(df[price_col].median(), 2))

print("Minimum House Price:", round(df[price_col].min(), 2))

print("Maximum House Price:", round(df[price_col].max(), 2))


if area_col is not None:

    area_price_corr = df[area_col].corr(df[price_col])

    print("\nArea-Price Correlation:", round(area_price_corr, 3))

    if area_price_corr > 0.5:

        print(
            "Insight: Area has a strong positive linear "
            "relationship with house price."
        )

    elif area_price_corr > 0:

        print(
            "Insight: Area has a positive linear "
            "relationship with house price."
        )

    else:

        print(
            "Insight: Area does not show a positive "
            "linear relationship with house price."
        )


if location_col is not None:

    print("\nMost Expensive Location:")

    print(condition_analysis.head(1))


if bedroom_col is not None:

    print("\nAverage Price by Bedrooms:")

    print(
        df.groupby(bedroom_col)[price_col]
        .mean()
        .sort_values(ascending=False)
    )


if not price_correlations.empty:

    strongest_feature = price_correlations.abs().idxmax()

    strongest_value = price_correlations[strongest_feature]

    print("\nFeature with Strongest Absolute Price Correlation:")

    print("Feature:", strongest_feature)

    print("Correlation:", round(strongest_value, 3))


# ============================================================
# 25. SAVE FINAL ANALYZED DATASET
# ============================================================

print("\n23. SAVING FINAL DATASET")
print("-" * 50)

final_output_path = os.path.join(
    OUTPUT_DIR,
    "final_house_price_analysis.csv"
)

df.to_csv(
    final_output_path,
    index=False
)

print("Final Dataset Saved:", final_output_path)


# ============================================================
# 26. CREATE TEXT REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "house_price_analysis_report.txt"
)

with open(report_path, "w", encoding="utf-8") as report:

    report.write("HOUSE PRICE DATA ANALYSIS REPORT\n")

    report.write("=" * 50 + "\n\n")

    report.write(f"Total Rows: {len(df)}\n")

    report.write(f"Total Columns: {len(df.columns)}\n")

    report.write(f"Price Column: {price_col}\n")

    report.write(
        f"Average Price: {df[price_col].mean():.2f}\n"
    )

    report.write(
        f"Median Price: {df[price_col].median():.2f}\n"
    )

    report.write(
        f"Minimum Price: {df[price_col].min():.2f}\n"
    )

    report.write(
        f"Maximum Price: {df[price_col].max():.2f}\n"
    )

    report.write(
        f"Missing Values Remaining: {df.isnull().sum().sum()}\n"
    )

    report.write(
        f"Duplicate Rows Remaining: {df.duplicated().sum()}\n"
    )

    report.write("\nPRICE CORRELATIONS\n")

    report.write(str(price_correlations))

    report.write("\n\nOUTLIER ANALYSIS\n")

    report.write(str(outlier_report))

    report.write("\n\nBUSINESS INSIGHTS\n")

    report.write(
        "Review the generated tables and charts before "
        "making business decisions.\n"
    )

print("Text Report Saved:", report_path)


# ============================================================
# 27. PROJECT COMPLETED
# ============================================================

print("\n" + "=" * 70)

print("HOUSE PRICE DATA ANALYSIS COMPLETED SUCCESSFULLY!")

print("=" * 70)

print("\nGenerated Files:")

print("1. Cleaned Dataset")

print("2. Descriptive Statistics")

print("3. Missing Values Report")

print("4. Correlation Matrix")

print("5. Outlier Report")

print("6. Location and Price Analysis")

print("7. House Price Charts")

print("8. Business Insights Report")

print("\nCheck the 'outputs' folder for all generated files.")

print("\nThank You!")

