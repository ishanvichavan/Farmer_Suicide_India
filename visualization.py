import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------
# Step 1: Load & Clean Data
# --------------------------
df = pd.read_csv("RS_Session_259_AU_204_1.csv")

# Drop unnecessary column
df.drop(columns=["Sl. No."], inplace=True)

# Rename columns for consistency
df.columns = ["State/UT", "2017", "2018", "2019", "2020", "2021"]

# Fill missing values with 0
df.fillna(0, inplace=True)

# --------------------------
# Step 2: Create Long Format
# --------------------------
df_long = df.melt(
    id_vars="State/UT",
    value_vars=["2017", "2018", "2019", "2020", "2021"],
    var_name="Year",
    value_name="Farmers"
)
df_long["Year"] = df_long["Year"].astype(int)

# --------------------------
# Step 3: Create Output Folder
# --------------------------
output_dir = "eda_outputs"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Step 4: Visualization 1 - National Trend
# --------------------------
sns.set_theme(style="whitegrid")
national_trend = df_long.groupby("Year")["Farmers"].sum().reset_index()

plt.figure(figsize=(8, 5))
sns.lineplot(data=national_trend, x="Year", y="Farmers", marker="o")
plt.title("National Trend of Farmers (2017–2021)")
plt.ylabel("Total Farmers")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig(f"{output_dir}/national_trend.png")
plt.show()

# --------------------------
# Visualization 2 - Top/Bottom States in 2021
# --------------------------
state_2021 = df_long[df_long["Year"] == 2021].sort_values(by="Farmers", ascending=False)

# Top 5
plt.figure(figsize=(10, 6))
sns.barplot(data=state_2021.head(5), x="Farmers", y="State/UT", palette="viridis")
plt.title("Top 5 States with Most Farmers (2021)")
plt.tight_layout()
plt.savefig(f"{output_dir}/top5_states_2021.png")
plt.show()

# Bottom 5
plt.figure(figsize=(10, 6))
sns.barplot(data=state_2021.tail(5), x="Farmers", y="State/UT", palette="magma")
plt.title("Bottom 5 States with Fewest Farmers (2021)")
plt.tight_layout()
plt.savefig(f"{output_dir}/bottom5_states_2021.png")
plt.show()

# --------------------------
# Visualization 3 - Change from 2017 to 2021
# --------------------------
df_change = df.copy()
df_change["Change_2017_2021"] = df_change["2021"] - df_change["2017"]
df_change_sorted = df_change.sort_values(by="Change_2017_2021", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=df_change_sorted, x="Change_2017_2021", y="State/UT", palette="coolwarm")
plt.title("Change in Farmers Count (2017 → 2021)")
plt.xlabel("Change in Farmers")
plt.tight_layout()
plt.savefig(f"{output_dir}/change_2017_2021.png")
plt.show()

# --------------------------
# Step 5: Feature Engineering - YoY Growth & CAGR
# --------------------------
df_growth = df.copy()

# YoY growth columns
for y1, y2 in zip(["2017", "2018", "2019", "2020"], ["2018", "2019", "2020", "2021"]):
    df_growth[f"YoY_{y1}_{y2}"] = df_growth[y2] - df_growth[y1]

# CAGR calculation
def calculate_cagr(start, end, periods):
    return ((end / start) ** (1 / periods) - 1) * 100 if start > 0 and end > 0 else 0

df_growth["CAGR_2017_2021"] = df_growth.apply(
    lambda row: calculate_cagr(row["2017"], row["2021"], 4), axis=1
)

# Save growth stats to CSV
df_growth.to_csv(f"{output_dir}/growth_summary.csv", index=False)

# View top 5 states by CAGR
top_cagr = df_growth[["State/UT", "2017", "2021", "CAGR_2017_2021"]].sort_values(by="CAGR_2017_2021", ascending=False)
print("Top 5 States by CAGR (2017–2021):")
print(top_cagr.head())