import pandas as pd
import os

# ---------------- Step 1: Load dataset ----------------
df = pd.read_csv("final_dataset.csv")   # <-- change filename if needed

# ---------------- Step 2: Clean data ----------------
# Drop duplicates
df = df.drop_duplicates()

# Forward fill missing values
df = df.ffill()

# Create 'date' column from year + month
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')

# Sort by english_name and date
df = df.sort_values(['english_name', 'date'])

# ---------------- Step 3: Replace values ----------------
df["english_name"] = df["english_name"].replace({
    "(": "extra",
    "+ / -": "extra1"   # <-- added your second condition too
})

# ---------------- Step 4: Save cleaned dataset ----------------
output_path = "data/final_cleaned_dataset.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # make sure folder exists
df.to_csv(output_path, index=False)

print(f"✅ Cleaned dataset saved at: {output_path}")
