import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import shapiro
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Anemia EDA Dashboard",
    page_icon="🩸",
    layout="wide"
)

st.title("🩸 Anemia Data Analysis Dashboard")
st.write(
    "An interactive Exploratory Data Analysis (EDA) dashboard "
    "for studying blood parameters and anemia status."
)

sns.set_theme(style="whitegrid")

REQUIRED_COLUMNS = [
    "Gender", "Hemoglobin", "MCH", "MCHC", "MCV", "Result"
]
FEATURES = ["Hemoglobin", "MCH", "MCHC", "MCV"]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload anemia.csv (optional)",
    type=["csv"]
)

default_path = Path(__file__).parent / "anemia.csv"

try:
    if uploaded_file is not None:
        df_original = load_csv(uploaded_file.getvalue())
        st.sidebar.success("Uploaded CSV loaded.")
    elif default_path.exists():
        df_original = pd.read_csv(default_path)
        st.sidebar.success("Local anemia.csv loaded.")
    else:
        st.error(
            "Dataset not found. Place anemia.csv in the same folder "
            "as app.py, or upload it using the sidebar."
        )
        st.info(
            "Download anemia.csv from the Kaggle dataset listed in the "
            "project README, then put it beside app.py."
        )
        st.stop()
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

missing_columns = [c for c in REQUIRED_COLUMNS if c not in df_original.columns]
if missing_columns:
    st.error(
        "The CSV does not contain the expected columns: "
        + ", ".join(missing_columns)
    )
    st.write("Columns found:", df_original.columns.tolist())
    st.stop()

# Convert analysis columns to numeric where appropriate.
df_original = df_original.copy()
for column in FEATURES + ["Result"]:
    df_original[column] = pd.to_numeric(
        df_original[column], errors="coerce"
    )

# ---------------------------------------------------------
# SIDEBAR FILTERS / CLEANING OPTIONS
# ---------------------------------------------------------
st.sidebar.header("Analysis settings")

remove_duplicates = st.sidebar.checkbox(
    "Exclude duplicate rows from analysis",
    value=False
)

df = df_original.drop_duplicates().copy() if remove_duplicates else df_original.copy()

gender_values = sorted(df["Gender"].dropna().unique().tolist(), key=str)
selected_genders = st.sidebar.multiselect(
    "Select Gender values",
    options=gender_values,
    default=gender_values
)

result_values = sorted(df["Result"].dropna().unique().tolist())
selected_results = st.sidebar.multiselect(
    "Select Result values",
    options=result_values,
    default=result_values
)

df = df[
    df["Gender"].isin(selected_genders)
    & df["Result"].isin(selected_results)
].copy()

st.sidebar.caption(
    "Result coding in this project: 0 = No Anemia, 1 = Anemia. "
    "Gender values are shown using the dataset's original coding."
)


# ---------------------------------------------------------
# TOP-LEVEL SUMMARY
# ---------------------------------------------------------
st.subheader("Dataset Overview")

total_rows = len(df)
total_columns = len(df.columns)
missing_total = int(df.isnull().sum().sum())
duplicate_total = int(df_original.duplicated().sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records in current view", f"{total_rows:,}")
c2.metric("Columns", total_columns)
c3.metric("Missing cells", f"{missing_total:,}")
c4.metric("Duplicate rows in original data", f"{duplicate_total:,}")

tab_overview, tab_distribution, tab_comparison, tab_correlation, tab_data, tab_prediction = st.tabs(
    [
        "Overview & Statistics",
        "Distributions",
        "Anemia Comparisons",
        "Correlation",
        "Data Explorer",
        "ML Prediction"
    ]
)


# ---------------------------------------------------------
# TAB 1: OVERVIEW AND STATISTICS
# ---------------------------------------------------------
with tab_overview:
    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Dataset structure")
    left, right = st.columns(2)

    with left:
        st.write("**Column data types**")
        dtype_table = pd.DataFrame({
            "Column": df.columns,
            "Data type": df.dtypes.astype(str).values
        })
        st.dataframe(dtype_table, use_container_width=True)

    with right:
        st.write("**Missing values by column**")
        missing_table = df.isnull().sum().rename("Missing values").to_frame()
        st.dataframe(missing_table, use_container_width=True)

    st.subheader("Statistical summary")
    st.dataframe(df.describe().T, use_container_width=True)

    st.subheader("Feature statistics")
    available_features = [f for f in FEATURES if f in df.columns]
    stats = df[available_features].agg(
        ["mean", "median", "min", "max", "std"]
    ).T
    st.dataframe(stats, use_container_width=True)

    st.subheader("Gender distribution")
    gender_counts = df["Gender"].value_counts(dropna=False)
    st.dataframe(
        gender_counts.rename_axis("Gender value").to_frame("Count"),
        use_container_width=True
    )

    st.subheader("Exploratory Hemoglobin grouping")
    st.caption(
        "These are data-based quantile groups for exploration, not medical "
        "or diagnostic Hemoglobin ranges."
    )
    group_count = st.slider(
        "Number of Hemoglobin groups",
        min_value=2,
        max_value=5,
        value=3
    )
    hb_values = df["Hemoglobin"].dropna()
    if hb_values.nunique() >= 2:
        hb_groups = pd.qcut(
            hb_values,
            q=group_count,
            duplicates="drop"
        )
        hb_group_counts = hb_groups.value_counts().sort_index()
        hb_group_table = hb_group_counts.rename_axis(
            "Hemoglobin quantile group"
        ).reset_index(name="Number of records")
        st.dataframe(hb_group_table, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=hb_group_table,
            x="Hemoglobin quantile group",
            y="Number of records",
            ax=ax
        )
        ax.set_title("Hemoglobin Quantile Groups")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    else:
        st.info("Not enough distinct Hemoglobin values for grouping.")


# ---------------------------------------------------------
# TAB 2: DISTRIBUTIONS
# ---------------------------------------------------------
with tab_distribution:
    st.subheader("Numerical feature distributions")

    feature = st.selectbox(
        "Choose a blood parameter",
        FEATURES,
        key="distribution_feature"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=df,
        x=feature,
        kde=True,
        ax=ax,
        color="steelblue"
    )
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(
        data=df,
        x=feature,
        fill=True,
        ax=ax,
        color="teal"
    )
    ax.set_title(f"Probability Density Curve - {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("Outlier analysis using the IQR method")
    st.caption(
        "An observation is flagged when it is below Q1 - 1.5 × IQR "
        "or above Q3 + 1.5 × IQR. This is a statistical flag, not proof "
        "that a measurement is erroneous."
    )
    outlier_rows = []
    for col in FEATURES:
        values = df[col].dropna()
        if not values.empty:
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((values < lower) | (values > upper)).sum())
            outlier_rows.append({
                "Feature": col,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower bound": lower,
                "Upper bound": upper,
                "Potential outliers": count
            })
    st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)

    st.subheader("Normality and skewness")
    normality_rows = []

    for col in FEATURES:
        values = df[col].dropna()
        if len(values) >= 3:
            # Shapiro-Wilk is designed for sample sizes up to 5000.
            sample = values.sample(
                n=min(len(values), 5000),
                random_state=42
            ) if len(values) > 5000 else values

            statistic, p_value = shapiro(sample)
            skew_value = values.skew()

            normality_rows.append({
                "Feature": col,
                "Shapiro-Wilk statistic": statistic,
                "p-value": p_value,
                "Skewness": skew_value,
                "Interpretation (p < 0.05)": (
                    "Evidence against normality"
                    if p_value < 0.05
                    else "Insufficient evidence against normality"
                )
            })

    if normality_rows:
        st.dataframe(
            pd.DataFrame(normality_rows),
            use_container_width=True
        )
        st.caption(
            "The Shapiro-Wilk test is run on up to 5,000 non-missing "
            "observations per feature. A p-value below 0.05 indicates "
            "evidence against a normal distribution."
        )
    else:
        st.warning("Not enough valid numerical observations for the test.")


# ---------------------------------------------------------
# TAB 3: ANEMIA COMPARISONS
# ---------------------------------------------------------
with tab_comparison:
    st.subheader("Anemia status distribution")

    result_labels = df["Result"].map({
        0: "No Anemia (0)",
        1: "Anemia (1)"
    }).fillna(df["Result"].astype(str))

    result_plot = result_labels.value_counts().rename_axis(
        "Anemia status"
    ).reset_index(name="Number of records")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=result_plot,
        x="Anemia status",
        y="Number of records",
        ax=ax
    )
    ax.set_title("Anemia Status Distribution")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("Blood parameters vs anemia status")
    selected_feature = st.selectbox(
        "Choose a parameter to compare",
        FEATURES,
        key="comparison_feature"
    )

    box_df = df.copy()
    box_df["Anemia status"] = box_df["Result"].map({
        0: "No Anemia (0)",
        1: "Anemia (1)"
    }).fillna(box_df["Result"].astype(str))

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=box_df,
        x="Anemia status",
        y=selected_feature,
        ax=ax
    )
    ax.set_title(f"{selected_feature} vs Anemia Status")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("Relationship between two numerical features")
    scatter_x = st.selectbox(
        "X-axis feature",
        FEATURES,
        index=0,
        key="scatter_x"
    )
    scatter_y = st.selectbox(
        "Y-axis feature",
        FEATURES,
        index=1,
        key="scatter_y"
    )
    if scatter_x == scatter_y:
        st.info("Choose two different features to view their relationship.")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            data=df,
            x=scatter_x,
            y=scatter_y,
            hue="Result",
            ax=ax
        )
        ax.set_title(f"{scatter_x} vs {scatter_y}")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    st.subheader("Gender vs anemia status")
    gender_plot = df.copy()
    gender_plot["Anemia status"] = gender_plot["Result"].map({
        0: "No Anemia (0)",
        1: "Anemia (1)"
    }).fillna(gender_plot["Result"].astype(str))

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(
        data=gender_plot,
        x="Gender",
        hue="Anemia status",
        ax=ax
    )
    ax.set_title("Gender vs Anemia Status")
    ax.set_xlabel("Gender value (original dataset coding)")
    ax.set_ylabel("Number of records")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("Gender and anemia status counts")
    cross_tab = pd.crosstab(
        df["Gender"],
        df["Result"],
        margins=True
    )
    st.dataframe(cross_tab, use_container_width=True)


# ---------------------------------------------------------
# TAB 4: CORRELATION
# ---------------------------------------------------------
with tab_correlation:
    st.subheader("Correlation heatmap")

    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax,
        vmin=-1,
        vmax=1
    )
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.caption(
        "Correlation ranges from -1 to +1. Positive values indicate "
        "a positive linear association, negative values indicate a "
        "negative linear association, and values near 0 indicate a "
        "weak linear association. Correlation does not prove causation."
    )


# ---------------------------------------------------------
# TAB 5: DATA EXPLORER
# ---------------------------------------------------------
with tab_data:
    st.subheader("Filtered dataset")
    st.write(f"Showing {len(df):,} records after the selected filters.")
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_data,
        file_name="anemia_filtered_data.csv",
        mime="text/csv"
    )
# ---------------------------------------------------------
# TAB 6: ML PREDICTION
# ---------------------------------------------------------
with tab_prediction:
    st.subheader("Anemia Prediction using Machine Learning")

    st.info(
        "Enter the blood parameter values below to predict anemia "
        "status using Logistic Regression and Random Forest."
    )

    st.caption(
        "This is an academic machine-learning demonstration, "
        "not a medical diagnosis."
    )

    @st.cache_resource
    def train_prediction_models(data):
        model_data = data[REQUIRED_COLUMNS].copy()

        for column in REQUIRED_COLUMNS:
            model_data[column] = pd.to_numeric(
                model_data[column], errors="coerce"
            )

        model_data = model_data.dropna().drop_duplicates()
        model_data = model_data[model_data["Result"].isin([0, 1])]

        input_features = ["Gender"] + FEATURES

        X = model_data[input_features]
        y = model_data["Result"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                ))
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42
            )
        }

        performance = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            performance[name] = {
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "F1-score": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "Confusion Matrix": confusion_matrix(
                    y_test, predictions, labels=[0, 1]
                )
            }

        return models, performance, model_data

    try:
        models, performance, model_data = train_prediction_models(
            df_original
        )

        st.subheader("Model Performance")

        performance_table = pd.DataFrame({
            name: {
                metric: values[metric]
                for metric in ["Accuracy", "Precision", "Recall", "F1-score"]
            }
            for name, values in performance.items()
        }).T

        st.dataframe(
            performance_table.style.format("{:.3f}"),
            use_container_width=True
        )

        st.caption(
            "Metrics are calculated using a 20% test set. "
            "Duplicate rows are removed before splitting the data."
        )

        st.subheader("Enter Your Blood Parameter Values")

        st.write(
            "Enter values for all five features to get a prediction."
        )

        input_features = ["Gender"] + FEATURES

        gender_options = sorted(
            model_data["Gender"].unique().tolist()
        )

        input_values = {}

        with st.form("anemia_prediction_form"):

            gender = st.selectbox(
                "Gender (original dataset code)",
                gender_options
            )
            input_values["Gender"] = gender

            col1, col2 = st.columns(2)

            for i, feature in enumerate(FEATURES):
                minimum = float(model_data[feature].min())
                maximum = float(model_data[feature].max())
                median = float(model_data[feature].median())

                with (col1 if i % 2 == 0 else col2):
                    input_values[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=median,
                        step=0.1
                    )

            submitted = st.form_submit_button(
                "Predict Anemia Status",
                use_container_width=True
            )

        if submitted:
            user_data = pd.DataFrame(
                [input_values],
                columns=input_features
            )

            st.subheader("Prediction Results")

            result_columns = st.columns(2)

            for i, (name, model) in enumerate(models.items()):
                prediction = int(model.predict(user_data)[0])

                if prediction == 1:
                    label = "Anemia"
                else:
                    label = "No Anemia"

                with result_columns[i]:
                    st.markdown(f"**{name}**")
                    if prediction == 1:
                        st.error(f"Prediction: {label}")
                    else:
                        st.success(f"Prediction: {label}")

                    probabilities = model.predict_proba(user_data)[0]
                    classes = model.classes_
                    predicted_probability = probabilities[
                        list(classes).index(prediction)
                    ]

                    st.write(
                        f"Model confidence score: "
                        f"{predicted_probability * 100:.2f}%"
                    )

            st.caption(
                "The confidence score is the model's estimated probability "
                "for its predicted class. It is not a medical certainty."
            )

        st.divider()
        st.subheader("Confusion Matrices")

        matrix_columns = st.columns(2)

        for i, (name, values) in enumerate(performance.items()):
            with matrix_columns[i]:
                st.write(f"**{name}**")

                matrix_df = pd.DataFrame(
                    values["Confusion Matrix"],
                    index=["Actual No Anemia", "Actual Anemia"],
                    columns=["Predicted No Anemia", "Predicted Anemia"]
                )

                st.dataframe(matrix_df, use_container_width=True)

    except Exception as error:
        st.error(f"Could not train the prediction models: {error}")
st.divider()
st.caption(
    "Academic EDA dashboard. This dashboard presents statistical "
    "patterns in the supplied dataset; it is not a medical diagnostic tool."
)
