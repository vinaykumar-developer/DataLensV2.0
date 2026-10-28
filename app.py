"""
DataLens (Simplified) - Flask backend
--------------------------------------
Uploads a CSV, profiles each column, runs 4 statistical insight checks,
and renders the results on a single HTML page.

No database. No ML. Everything is computed fresh, in memory, per request.
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import io
import base64

import matplotlib
matplotlib.use("Agg")  # headless backend - required since Flask has no display/GUI
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

app = Flask(__name__)

# Severity ordering used to sort findings so the most serious ones show first.
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


# ---------------------------------------------------------------------------
# STEP 1: COLUMN PROFILING
# ---------------------------------------------------------------------------
def missing_bar_color(missing_pct):
    """Simple traffic-light color for the missing % bar (used in the profile table)."""
    if missing_pct > 30:
        return "#c62828"   # red   - matches 'critical' severity
    elif missing_pct > 10:
        return "#ef6c00"   # orange - matches 'warning' severity
    else:
        return "#2e7d32"   # green - healthy


def profile_columns(df):
    """
    Build a simple profile for every column:
    - dtype (as pandas sees it)
    - missing % (rounded) + a bar color for the mini visualization
    - unique value count
    """
    profiles = []
    total_rows = len(df)

    for col in df.columns:
        series = df[col]
        missing_count = series.isna().sum()
        missing_pct = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        profiles.append({
            "column": col,
            "dtype": str(series.dtype),
            "missing_pct": missing_pct,
            "missing_count": int(missing_count),
            "unique_count": int(series.nunique(dropna=True)),
            "bar_color": missing_bar_color(missing_pct),
        })

    return profiles


def build_data_preview(df, max_rows=50):
    """
    Build a preview of the raw uploaded data so the user can see what
    they actually uploaded, side by side with the profile/insights.
    Limited to `max_rows` so a huge CSV doesn't blow up the page.
    """
    total_rows = len(df)
    preview_df = df.head(max_rows)

    # Replace NaN with a visible placeholder so missing cells are obvious
    # rather than rendering as a blank or the literal text "nan".
    preview_df = preview_df.where(pd.notnull(preview_df), "—")

    return {
        "columns": list(df.columns),
        "rows": preview_df.to_dict(orient="records"),
        "shown_rows": len(preview_df),
        "total_rows": total_rows,
        "truncated": total_rows > max_rows,
    }


# ---------------------------------------------------------------------------
# STEP 1b: CHARTS (matplotlib + seaborn)
# Every chart is drawn once per request into an in-memory PNG buffer (never
# saved to disk) and embedded straight into the HTML as a base64 data URI.
# ---------------------------------------------------------------------------
def fig_to_base64(fig):
    """Render a matplotlib figure to a base64 PNG string and free its memory."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)  # important: frees memory, since Flask keeps running across requests
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_correlation_heatmap(df, high_corr_threshold=0.85):
    """
    Rule 3 visualization: a seaborn heatmap of the full correlation matrix
    for all numeric columns. Returns None if there are fewer than 2.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    size = max(4, len(corr.columns) * 0.9)

    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Matrix (numeric columns)")
    fig.tight_layout()
    return fig_to_base64(fig)


def build_histogram_image(series, col_name, is_flagged, skew_value):
    """
    Rule 2 (Skewness) visualization: a seaborn histogram + KDE curve for one
    numeric column, so the distribution shape is visible at a glance.
    """
    data = series.dropna()
    if len(data) == 0 or data.std() == 0:
        return None

    color = "#ef6c00" if is_flagged else "#2b6cb0"  # orange if skewed, blue otherwise

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    sns.histplot(data, bins=10, kde=True, color=color, ax=ax)
    ax.set_title(f"{col_name}  (skew = {skew_value})")
    ax.set_xlabel("")
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig_to_base64(fig)


def build_boxplot_image(series, col_name, outlier_count):
    """
    Rule 4 (Outliers - IQR) visualization: a seaborn box plot for one numeric
    column. Seaborn's default box plot already uses the 1.5x IQR rule to
    decide which points are drawn as individual outlier markers.
    """
    data = series.dropna()
    if len(data) < 4:
        return None

    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    sns.boxplot(
        x=data, ax=ax, color="#cfe3fa",
        flierprops=dict(markerfacecolor="#c62828", marker="o", markersize=6, alpha=0.85),
    )
    ax.set_title(f"{col_name}  ({outlier_count} outlier{'s' if outlier_count != 1 else ''})")
    ax.set_xlabel("")
    fig.tight_layout()
    return fig_to_base64(fig)


def build_numeric_visuals(df, skew_threshold=1.0):
    """
    For every numeric column, build the histogram (skewness) and box plot
    (outliers) images, plus the numbers used in their titles. Powers the
    'Skewness' and 'Outliers' chart sections in the template.
    """
    visuals = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Skewness for this column (same logic as check_skewness)
        skew_value = None
        is_skewed = False
        if len(series) >= 3 and series.std() != 0:
            raw_skew = series.skew()
            if not pd.isna(raw_skew):
                skew_value = round(float(raw_skew), 2)
                is_skewed = abs(raw_skew) > skew_threshold

        # Outlier count for this column (same logic as check_outliers_iqr)
        outlier_count = 0
        if len(series) >= 4:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr != 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_count = int(((series < lower) | (series > upper)).sum())

        visuals.append({
            "column": col,
            "skew_value": skew_value,
            "is_skewed": is_skewed,
            "histogram_image": build_histogram_image(series, col, is_skewed, skew_value),
            "outlier_count": outlier_count,
            "boxplot_image": build_boxplot_image(series, col, outlier_count),
        })

    return visuals


# ---------------------------------------------------------------------------
# STEP 2: THE 4 INSIGHT RULES
# Each rule returns a list of finding dicts:
#   { "column": str, "message": str, "severity": "critical"|"warning"|"info", "rule": str }
# ---------------------------------------------------------------------------

def check_missing_values(df, threshold=30.0):
    """Rule 1: flag columns with more than `threshold`% missing data."""
    findings = []
    total_rows = len(df)
    if total_rows == 0:
        return findings

    for col in df.columns:
        missing_pct = (df[col].isna().sum() / total_rows) * 100
        if missing_pct > threshold:
            findings.append({
                "column": col,
                "rule": "Missing Values",
                "severity": "critical",
                "message": (
                    f"Column '{col}' has {missing_pct:.1f}% missing values "
                    f"(threshold is {threshold:.0f}%). Consider imputing or dropping this column."
                ),
            })
    return findings


def check_skewness(df, threshold=1.0):
    """Rule 2: flag numeric columns with |skew| > threshold."""
    findings = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        # Skewness needs at least 3 data points and some variance to be meaningful.
        if len(series) < 3 or series.std() == 0:
            continue

        skew_val = series.skew()
        if pd.isna(skew_val):
            continue

        if abs(skew_val) > threshold:
            direction = "right (positive)" if skew_val > 0 else "left (negative)"
            transform = "log" if skew_val > 0 else "square (or reflect and log)"
            findings.append({
                "column": col,
                "rule": "Skewness",
                "severity": "warning",
                "message": (
                    f"Column '{col}' is skewed {direction} with a skewness of {skew_val:.2f} "
                    f"(threshold is |{threshold:.1f}|). A log or sqrt transform may help normalize it."
                    if skew_val > 0 else
                    f"Column '{col}' is skewed {direction} with a skewness of {skew_val:.2f} "
                    f"(threshold is |{threshold:.1f}|). A transform may help normalize it."
                ),
            })
    return findings


def check_correlation(df, threshold=0.85):
    """Rule 3: flag numeric column pairs with |correlation| above threshold."""
    findings = []
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return findings

    corr_matrix = numeric_df.corr()
    cols = corr_matrix.columns

    # Only look at each pair once (upper triangle), skip self-correlation.
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a, col_b = cols[i], cols[j]
            corr_val = corr_matrix.loc[col_a, col_b]

            if pd.isna(corr_val):
                continue

            if abs(corr_val) > threshold:
                findings.append({
                    "column": f"{col_a} & {col_b}",
                    "rule": "Correlation",
                    "severity": "warning",
                    "message": (
                        f"Columns '{col_a}' and '{col_b}' are highly correlated "
                        f"(r = {corr_val:.2f}, threshold is {threshold:.2f}). "
                        f"One of them may be redundant for modeling."
                    ),
                })
    return findings


def check_outliers_iqr(df):
    """Rule 4: flag numeric columns with values outside 1.5x IQR from Q1/Q3."""
    findings = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = len(outliers)

        if outlier_count > 0:
            outlier_pct = (outlier_count / len(series)) * 100
            findings.append({
                "column": col,
                "rule": "Outliers (IQR)",
                "severity": "info",
                "message": (
                    f"Column '{col}' has {outlier_count} outlier value(s) "
                    f"({outlier_pct:.1f}% of non-missing rows) outside the range "
                    f"[{lower_bound:.2f}, {upper_bound:.2f}] (1.5x IQR from Q1/Q3)."
                ),
            })
    return findings


def run_all_insights(df):
    """
    Run all 4 rules and return one combined, sorted list of findings.
    Sorted so 'critical' findings appear before 'warning' before 'info'.
    """
    findings = []
    findings += check_missing_values(df)
    findings += check_skewness(df)
    findings += check_correlation(df)
    findings += check_outliers_iqr(df)

    findings.sort(key=lambda f: SEVERITY_ORDER.get(f["severity"], 99))
    return findings


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=None, error=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("csv_file")

    # --- Basic error handling ---
    if file is None or file.filename == "":
        return render_template("index.html", results=None, error="No file selected. Please choose a CSV file.")

    if not file.filename.lower().endswith(".csv"):
        return render_template("index.html", results=None, error="Only .csv files are supported.")

    raw_bytes = file.read()
    if len(raw_bytes) == 0:
        return render_template("index.html", results=None, error="The uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        return render_template("index.html", results=None, error=f"Could not parse CSV file: {e}")

    if df.empty or len(df.columns) == 0:
        return render_template("index.html", results=None, error="The CSV file has no data to analyze.")

    # --- Run profiling + insights ---
    column_profiles = profile_columns(df)
    findings = run_all_insights(df)
    correlation_heatmap_image = build_correlation_heatmap(df)
    numeric_visuals = build_numeric_visuals(df)
    data_preview = build_data_preview(df)

    results = {
        "filename": file.filename,
        "row_count": len(df),
        "column_count": len(df.columns),
        "profiles": column_profiles,
        "findings": findings,
        "findings_count": len(findings),
        "correlation_heatmap_image": correlation_heatmap_image,
        "numeric_visuals": numeric_visuals,
        "data_preview": data_preview,
    }

    return render_template("index.html", results=results, error=None)


if __name__ == "__main__":
    app.run(debug=True)
