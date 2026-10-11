
import streamlit as st
import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CSV Health Checker",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 CSV Health Checker")

st.write(
    "Check your dataset before analysis or machine learning. "
    "This tool helps you discover potential data quality problems "
    "and understand what you should investigate next."
)

st.info(
    "Workflow: Overview → Column Information → Data Quality → "
    "Statistics → Relationships → Critical Issues → Recommendations"
)


# --------------------------------------------------
# 2. FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    help="Choose a CSV file from your computer."
)

if uploaded_file is None:
    st.markdown("### How to use this tool")
    st.write("1. Upload a CSV file.")
    st.write("2. Review the dataset overview.")
    st.write("3. Inspect missing values, duplicates, and other potential problems.")
    st.write("4. Explore statistics and relationships.")
    st.write("5. Read the recommendations before deciding what to do next.")

    st.caption(
        "Important: Recommendations are suggestions for investigation, "
        "not automatic instructions to modify your data."
    )

    st.stop()


# --------------------------------------------------
# 3. LOAD AND VALIDATE THE DATASET
# --------------------------------------------------

try:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)

except pd.errors.EmptyDataError:
    st.error(
        "This file is empty or does not contain readable CSV data. "
        "Please upload another file."
    )
    st.stop()

except UnicodeDecodeError:
    st.error(
        "The file encoding could not be read. "
        "Try saving the dataset as a UTF-8 CSV file and uploading it again."
    )
    st.stop()

except pd.errors.ParserError:
    st.error(
        "The CSV structure could not be parsed. "
        "Check whether the file has consistent columns and separators."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the CSV file: {error}")
    st.stop()


if len(df.columns) == 0:
    st.error("This CSV has no columns to analyze.")
    st.stop()

if len(df) == 0:
    st.warning(
        "The file contains column names but no data rows. "
        "Upload a dataset containing at least one row to continue."
    )
    st.stop()

if df.columns.duplicated().any():
    st.warning(
        "Some column names are duplicated. Pandas may rename duplicate "
        "column names while reading the file. Check the original CSV."
    )

if (df.columns.astype(str).str.strip()=="").any():
    st.warning(
        "Some columns have blank or whitespace-only names. "
        "Consider giving these columns meaningful names."
    )


# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def format_percentage(value):
    """Format a number as a percentage."""
    return f"{value:.2f}%"


def get_id_like_columns(data):
    """
    Identify possible identifier columns using both column names
    and high uniqueness. This is a heuristic, not a definitive test.
    """
    if len(data) == 0:
        return []

    id_keywords = {
        "id",
        "identifier",
        "customer_number",
        "account_number",
        "reference",
        "record_number",
        "user_id",
        "customer_id",
        "transaction_id",
    }

    id_like = []

    for col in data.columns:
        name = str(col).strip().lower()
        normalized_name = name.replace(" ", "_").replace("-", "_")

        uniqueness_ratio = data[col].nunique(dropna=True) / len(data)

        name_suggests_id = (
            normalized_name == "id"
            or normalized_name.endswith("_id")
            or normalized_name in id_keywords
            or normalized_name.endswith("_identifier")
            or normalized_name.endswith("_reference")
            or normalized_name.endswith("_number")
        )

        if name_suggests_id and uniqueness_ratio >= 0.80:
            id_like.append(col)

    return id_like


# --------------------------------------------------
# 5. DATASET OVERVIEW
# --------------------------------------------------

st.header("1. Dataset Overview")

st.write(
    "Start here. These measurements describe the size of your dataset "
    "and give you a quick look at the information it contains."
)

rows, columns = df.shape

missing_cells = int(df.isna().sum().sum())
total_cells = int(df.shape[0] * df.shape[1])

missing_percentage = (
    missing_cells / total_cells * 100
    if total_cells > 0
    else 0
)

duplicate_rows = int(df.duplicated().sum())

duplicate_percentage = (
    duplicate_rows / len(df) * 100
    if len(df) > 0
    else 0
)

memory_usage_kb = df.memory_usage(deep=True).sum() / 1024

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", f"{rows:,}")
col2.metric("Columns", f"{columns:,}")
col3.metric("Missing Cells", f"{missing_cells:,}")
col4.metric("Duplicate Rows", f"{duplicate_rows:,}")

st.caption(
    f"Approximate in-memory size: {memory_usage_kb:,.2f} KB. "
    "This is not necessarily the original CSV file size."
)

with st.expander("What do these numbers mean?"):
    st.write(
        "- **Rows:** Number of observations or records.\n"
        "- **Columns:** Number of variables or fields.\n"
        "- **Missing cells:** Number of empty or missing values across the dataset.\n"
        "- **Duplicate rows:** Number of rows that repeat an earlier row across all columns."
    )

st.subheader("Dataset Preview")

st.write(
    "Inspect the first rows to understand how the data is organized "
    "and whether the values look reasonable."
)

st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------
# 6. COLUMN INFORMATION
# --------------------------------------------------

st.header("2. Column Information")

st.write(
    "Each column represents a variable. Check its data type, missing "
    "values, and number of distinct values before deciding how to use it."
)

column_info = pd.DataFrame({
    "Column": df.columns.astype(str),
    "Data Type": df.dtypes.astype(str).values,
    "Non-Missing Values": df.notna().sum().values,
    "Missing Values": df.isna().sum().values,
    "Missing (%)": (
        df.isna().mean().values * 100
    ).round(2),
    "Unique Values": df.nunique(dropna=True).values,
})

st.dataframe(column_info, use_container_width=True, hide_index=True)

with st.expander("How should I interpret column information?"):
    st.write(
        "- **Data type:** How pandas currently interprets the column.\n"
        "- **Non-missing values:** Number of cells that contain a value.\n"
        "- **Missing values:** Number of empty or missing cells.\n"
        "- **Missing (%):** Percentage of rows with a missing value in that column.\n"
        "- **Unique values:** Number of distinct non-missing values."
    )

st.caption(
    "A column stored as text may contain numbers or dates. "
    "Check the actual values before changing its data type."
)


# --------------------------------------------------
# 7. DATA QUALITY
# --------------------------------------------------

st.header("3. Data Quality")

st.write(
    "This section checks for missing values, repeated rows, "
    "and columns that contain little or no variation."
)

constant_columns = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

quality_col1, quality_col2, quality_col3 = st.columns(3)

quality_col1.metric(
    "Missing Data",
    format_percentage(missing_percentage)
)

quality_col2.metric(
    "Duplicate Rows",
    format_percentage(duplicate_percentage)
)

quality_col3.metric(
    "Constant Columns",
    len(constant_columns)
)

# Missing values by column
missing_by_column = df.isna().sum()
missing_by_column = missing_by_column[missing_by_column > 0]

if not missing_by_column.empty:
    st.subheader("Missing Values by Column")

    missing_chart = missing_by_column.sort_values(ascending=False)

    st.bar_chart(missing_chart)

    st.write("Columns with missing values:")

    st.dataframe(
        pd.DataFrame({
            "Missing Values": missing_by_column,
            "Missing (%)": (
                missing_by_column / len(df) * 100
            ).round(2),
        }).sort_values("Missing Values", ascending=False),
        use_container_width=True
    )

    st.warning(
        "Investigate why values are missing. Missing values may be "
        "caused by collection problems, unavailable information, or "
        "a value that genuinely does not apply."
    )
else:
    st.success("No missing values were detected.")

# Duplicate rows
if duplicate_rows > 0:
    st.warning(
        f"Found {duplicate_rows:,} duplicate rows "
        f"({duplicate_percentage:.2f}% of all rows)."
    )

    st.write(
        "Investigate whether these are accidental duplicates or legitimate "
        "repeated observations before deciding whether to remove them."
    )

    with st.expander("Preview duplicate rows"):
        duplicate_preview = df[df.duplicated(keep=False)].head(20)
        st.dataframe(duplicate_preview, use_container_width=True)
else:
    st.success("No exact duplicate rows were detected.")

# Constant columns
if constant_columns:
    st.warning(
        f"Potentially uninformative columns detected: {constant_columns}"
    )

    st.write(
        "A constant column has only one distinct value, including missing "
        "values in this check. It usually provides little information for "
        "modeling, but confirm its role before removing it."
    )
else:
    st.success("No constant columns were detected.")


# --------------------------------------------------
# 8. STATISTICS
# --------------------------------------------------

st.header("4. Statistical Summary")

st.write(
    "Statistics help you understand value ranges, typical values, "
    "and possible unusual patterns. Interpret them using knowledge "
    "of what each variable represents."
)

numeric_df = df.select_dtypes(include="number")

if not numeric_df.empty:
    st.subheader("Numerical Columns")

    st.write(
        "These summaries describe numerical columns. They can help you "
        "identify unusual ranges or large differences between values."
    )

    numeric_summary = numeric_df.describe().T
    st.dataframe(numeric_summary, use_container_width=True)

    st.subheader("Potentially Skewed Numerical Columns")

    skew_values = numeric_df.skew(numeric_only=True)
    skewed_columns = skew_values[skew_values.abs() > 1]

    if not skewed_columns.empty:
        skew_table = pd.DataFrame({
            "Skewness": skewed_columns.round(3),
            "Interpretation": [
                "Potentially right-skewed"
                if value > 1
                else "Potentially left-skewed"
                for value in skewed_columns
            ],
        })

        st.dataframe(skew_table, use_container_width=True)

        st.info(
            "Strong skewness can be normal for some variables. "
            "Investigate the distribution and the purpose of your analysis "
            "before considering a transformation."
        )
    else:
        st.success(
            "No numerical columns exceeded the current absolute skewness "
            "threshold of 1."
        )

else:
    st.info(
        "No numerical columns were detected. Check whether columns containing "
        "numbers have been imported as text."
    )


# Categorical summaries
categorical_df = df.select_dtypes(
    include=["object", "category", "bool"]
)

if not categorical_df.empty:
    st.subheader("Categorical Columns")

    st.write(
        "Categorical summaries show the most common values in text, "
        "category, and Boolean columns."
    )

    for col in categorical_df.columns:
        with st.expander(f"Column: {col}"):
            counts = df[col].value_counts(dropna=False).head(10)

            category_summary = counts.rename_axis(
                "Value"
            ).reset_index(name="Count")

            st.dataframe(
                category_summary,
                use_container_width=True,
                hide_index=True
            )

            if df[col].nunique(dropna=True) > 100:
                st.caption(
                    "This column has many distinct values. "
                    "It may be free text, a code, a date, or an identifier."
                )


# --------------------------------------------------
# 9. RELATIONSHIPS
# --------------------------------------------------

st.header("5. Relationships Between Numerical Columns")

st.write(
    "Correlation measures how two numerical variables move together "
    "in a linear pattern. It can help you identify relationships worth "
    "investigating."
)

st.caption(
    "Correlation does not prove causation. A weak correlation also does "
    "not rule out a nonlinear relationship."
)

if numeric_df.shape[1] >= 2:
    correlation_matrix = numeric_df.corr()

    st.subheader("Correlation Matrix")

    st.dataframe(
        correlation_matrix.round(3),
        use_container_width=True
    )

    strong_correlations = []

    numeric_columns = correlation_matrix.columns

    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            col_a = numeric_columns[i]
            col_b = numeric_columns[j]

            corr_value = correlation_matrix.loc[col_a, col_b]

            if pd.notna(corr_value) and abs(corr_value) >= 0.7:
                strong_correlations.append({
                    "Column A": col_a,
                    "Column B": col_b,
                    "Correlation": round(corr_value, 3),
                })

    if strong_correlations:
        st.subheader("Potentially Strong Correlations")

        st.dataframe(
            pd.DataFrame(strong_correlations),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Strong correlations may indicate related variables, overlapping "
            "information, or a shared underlying cause. Check the context "
            "before deciding what they mean."
        )
    else:
        st.info(
            "No pairs of numerical columns had an absolute correlation "
            "of 0.7 or higher."
        )

else:
    st.info(
        "At least two numerical columns are needed to calculate correlations."
    )


# --------------------------------------------------
# 10. CRITICAL ISSUES
# --------------------------------------------------

st.header("6. Critical Issues to Investigate")

st.write(
    "This section brings together potential problems that may affect "
    "your analysis. These are warnings to investigate, not proof that "
    "the dataset is unusable."
)

critical_issues = []

# High missingness
missing_ratio = df.isna().mean()

high_missing_columns = missing_ratio[
    missing_ratio > 0.50
].index.tolist()

if high_missing_columns:
    critical_issues.append(
        f"More than 50% of values are missing in: {high_missing_columns}"
    )

# Duplicate rows
if duplicate_rows > 0:
    critical_issues.append(
        f"{duplicate_rows:,} exact duplicate rows were found."
    )

# Constant columns
if constant_columns:
    critical_issues.append(
        f"Columns with no variation were found: {constant_columns}"
    )

# Possible identifier columns
id_like_columns = get_id_like_columns(df)

if id_like_columns:
    critical_issues.append(
        f"Potential identifier columns were detected: {id_like_columns}"
    )

# Columns with no names
blank_column_names = [
    col for col in df.columns
    if str(col).strip() == ""
]

if blank_column_names:
    critical_issues.append(
        "One or more columns have blank names."
    )

if critical_issues:
    for issue in critical_issues:
        st.warning(issue)

    st.caption(
        "A warning does not automatically mean that a column or row "
        "should be deleted. Check its meaning and role first."
    )
else:
    st.success(
        "No issues covered by these critical checks were detected. "
        "Other data problems may still exist."
    )

if id_like_columns:
    st.subheader("Potential Identifier Columns")

    st.write(
        "These columns have names that suggest identifiers and high "
        "uniqueness. They may identify records rather than represent "
        "meaningful measurements."
    )

    st.write(id_like_columns)

    st.info(
        "If you are building a machine learning model, investigate whether "
        "these columns could cause leakage or encourage memorization. "
        "Do not remove them automatically: identifiers may be needed for "
        "record tracking, grouping, or making predictions."
    )


# --------------------------------------------------
# 11. RECOMMENDATIONS
# --------------------------------------------------

st.header("7. Recommendations")

st.write(
    "Use these suggestions as a checklist for your next investigation. "
    "The right action depends on the dataset, the problem you are solving, "
    "and how the data was collected."
)

recommendations = []

if missing_cells > 0:
    recommendations.append(
        "Investigate missing values. Check how much is missing, which "
        "columns are affected, and why the values are absent. Choose a "
        "handling strategy based on the variable and analysis goal."
    )

if high_missing_columns:
    recommendations.append(
        f"Review columns with more than 50% missing values: "
        f"{high_missing_columns}. Determine whether they are important "
        "before considering exclusion or imputation."
    )

if duplicate_rows > 0:
    recommendations.append(
        "Investigate duplicate rows. Confirm whether they represent "
        "accidental repetition or legitimate repeated observations before "
        "deciding whether to remove them."
    )

if constant_columns:
    recommendations.append(
        f"Review constant columns: {constant_columns}. They usually provide "
        "little information for modeling, but confirm their role before "
        "removing them."
    )

if id_like_columns:
    recommendations.append(
        f"Investigate possible identifier columns: {id_like_columns}. "
        "Decide whether each is an identifier, a meaningful feature, "
        "or a potential source of leakage for your specific task."
    )

if not numeric_df.empty:
    if not skewed_columns.empty:
        recommendations.append(
            "Inspect the distributions of potentially skewed numerical "
            "columns. Transformations may help some analyses or models, "
            "but are not always necessary."
        )

if len(df) < 30:
    recommendations.append(
        "Your dataset has fewer than 30 rows. Check whether it is "
        "representative enough for your intended analysis or modeling."
    )

if recommendations:
    for number, recommendation in enumerate(recommendations, start=1):
        st.markdown(f"**{number}.** {recommendation}")
else:
    st.success(
        "No recommendations were triggered by the current checks. "
        "Review the dataset with your subject knowledge before proceeding."
    )

st.divider()

st.caption(
    "CSV Health Checker provides preliminary diagnostics. "
    "It does not automatically clean your data or guarantee that a dataset "
    "is suitable for analysis or machine learning."
)