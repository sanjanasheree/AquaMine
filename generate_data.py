"""Export all built-in datasets to ./data as CSV (run: python generate_data.py)."""
import os
from core.datasets import water_quality_data, loan_data, customer_data, house_data

os.makedirs("data", exist_ok=True)
for name, df in {"groundwater_quality.csv": water_quality_data(),
                 "loan_applicants.csv": loan_data(),
                 "mall_customers.csv": customer_data(),
                 "house_prices.csv": house_data()}.items():
    df.to_csv(os.path.join("data", name), index=False)
    print(f"data/{name}: {df.shape[0]} rows x {df.shape[1]} cols")
