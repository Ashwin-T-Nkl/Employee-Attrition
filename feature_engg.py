import pandas as pd

INPUT_FILE = "clean_employee.csv"
OUTPUT_FILE = "engg_employee.csv"

def main():
    print("Step 2: Feature Engineering...")

    try:
        df = pd.read_csv(INPUT_FILE)
        print("File loaded ✅")
    except FileNotFoundError:
        print("Cleaned Data not found.")
        return

    # 1) Age Group (simple buckets)
    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[17, 30, 45, 60],
            labels=["Young", "Mid", "Senior"]
        )

    # 2) Tenure Group from YearsAtCompany
    if "YearsAtCompany" in df.columns:
        df["TenureGroup"] = pd.cut(
            df["YearsAtCompany"],
            bins=[-1, 2, 5, 40],   # 0–2, 3–5, 6+
            labels=["New", "Mid", "Long"]
        )

    # 3) Performance metric (simple score)
    if "PerformanceRating" in df.columns and "PercentSalaryHike" in df.columns:
        df["PerformanceScore"] = df["PerformanceRating"] * df["PercentSalaryHike"]

    # 4) Engagement score (average of satisfaction-type columns)
    engagement_cols = []
    for c in [
        "JobSatisfaction",
        "JobInvolvement",
        "WorkLifeBalance",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction"
    ]:
        if c in df.columns:
            engagement_cols.append(c)

    if len(engagement_cols) > 0:
        df["EngagementScore"] = df[engagement_cols].mean(axis=1)

    # 5) Convert all categorical columns to numbers
    df = pd.get_dummies(df, drop_first=True)

    # 6) Save processed file
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Feature engineering done ✅ Saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
