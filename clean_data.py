import pandas as pd

INPUT_FILE = "employee_data.csv"
OUTPUT_FILE = "clean_employee.csv"

def main():
    print("Cleaning data...")

    # Read file safely
    try:
        df = pd.read_csv(INPUT_FILE)
        print("File loaded. Rows:", len(df))
    except FileNotFoundError:
        print("File not found:", INPUT_FILE)
        return   # ✅ now correctly inside main()

    # Drop unnecessary columns
    columns_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
    for c in columns_to_drop:
        if c in df.columns:
            df = df.drop(columns=c)

    # Convert Attrition column to numeric (0 = Stay, 1 = Leave)
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].replace({
            "Yes": 1, "No": 0,
            "Y": 1, "N": 0,
            "Left": 1, "Stayed": 0, "Stay": 0
        }).astype(int)
    else:
        print("WARNING: 'Attrition' column not found in dataset.")

    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))  # numeric columns
    df = df.fillna(df.mode().iloc[0])             # categorical columns

    # Save cleaned file
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaning done ✅ Saved file as: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
