"""
Synthetic but realistic datasets used across the application.

All generators are deterministic (fixed random seed) so results are
reproducible in lab records. Each one is also exported as CSV into /data
by `generate_data.py`, so students can open them in Excel / Jupyter.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# BIS IS 10500:2012 drinking-water limits (mg/L unless stated)
#   acceptable = desirable limit, permissible = limit in absence of alternate source
# ---------------------------------------------------------------------------
BIS_LIMITS = {
    "pH":               {"acceptable": (6.5, 8.5), "permissible": (6.5, 8.5), "unit": "-"},
    "TDS":              {"acceptable": 500,  "permissible": 2000, "unit": "mg/L"},
    "Total_Hardness":   {"acceptable": 200,  "permissible": 600,  "unit": "mg/L as CaCO3"},
    "Turbidity":        {"acceptable": 1,    "permissible": 5,    "unit": "NTU"},
    "Fluoride":         {"acceptable": 1.0,  "permissible": 1.5,  "unit": "mg/L"},
    "Arsenic":          {"acceptable": 0.01, "permissible": 0.01, "unit": "mg/L"},
    "Nitrate":          {"acceptable": 45,   "permissible": 45,   "unit": "mg/L"},
    "Iron":             {"acceptable": 1.0,  "permissible": 1.0,  "unit": "mg/L"},
    "Chloride":         {"acceptable": 250,  "permissible": 1000, "unit": "mg/L"},
    "Sulphate":         {"acceptable": 200,  "permissible": 400,  "unit": "mg/L"},
    "Total_Coliform":   {"acceptable": 0,    "permissible": 0,    "unit": "MPN/100 mL"},
}

# District profiles: (state, fluoride_bias, arsenic_bias, nitrate_bias, salinity_bias, coliform_bias)
# Biases reflect well-documented contamination belts in India (illustrative, not measured values).
DISTRICTS = {
    "Nalgonda":       ("Telangana",      2.2, 0.0, 0.8, 1.2, 0.8),
    "Anantapur":      ("Andhra Pradesh", 1.8, 0.0, 1.0, 1.3, 0.8),
    "Dharmapuri":     ("Tamil Nadu",     1.9, 0.0, 0.9, 1.0, 0.9),
    "Tiruvannamalai": ("Tamil Nadu",     1.2, 0.0, 1.1, 1.0, 1.0),
    "Coimbatore":     ("Tamil Nadu",     0.9, 0.0, 1.0, 1.0, 0.9),
    "Murshidabad":    ("West Bengal",    0.2, 2.5, 0.8, 0.8, 1.2),
    "Ballia":         ("Uttar Pradesh",  0.3, 2.0, 1.1, 0.9, 1.2),
    "Bhojpur":        ("Bihar",          0.3, 1.8, 1.0, 0.9, 1.3),
    "Bathinda":       ("Punjab",         0.9, 0.6, 2.0, 1.4, 0.8),
    "Jaipur":         ("Rajasthan",      1.6, 0.1, 1.6, 1.6, 0.7),
    "Barmer":         ("Rajasthan",      1.7, 0.0, 1.2, 2.2, 0.7),
    "Thiruvananthapuram": ("Kerala",     0.2, 0.0, 0.7, 0.6, 1.6),
}

SOURCE_TYPES = ["Borewell", "Open well", "Hand pump", "Piped supply"]
LAND_USE = ["Agricultural", "Urban", "Industrial", "Rural residential"]
SEASONS = ["Pre-monsoon", "Monsoon", "Post-monsoon"]

SENSOR_FEATURES = ["pH", "EC", "TDS", "Turbidity", "Temperature", "Dissolved_Oxygen",
                   "Total_Hardness", "Well_Depth_m", "Dist_to_Sanitation_m", "Rainfall_mm",
                   "Source_Type", "Land_Use", "Season"]
LAB_FEATURES = ["Fluoride", "Arsenic", "Nitrate", "Iron", "Chloride", "Sulphate", "Total_Coliform"]


def water_quality_data(n=2000, seed=42):
    """Groundwater samples: cheap field-sensor readings + expensive lab results + safety label."""
    rng = np.random.default_rng(seed)
    names = list(DISTRICTS)
    district = rng.choice(names, n)
    prof = np.array([DISTRICTS[d][1:] for d in district], dtype=float)
    f_b, as_b, no3_b, sal_b, col_b = prof.T

    source = rng.choice(SOURCE_TYPES, n, p=[0.42, 0.20, 0.23, 0.15])
    land = rng.choice(LAND_USE, n, p=[0.45, 0.22, 0.10, 0.23])
    season = rng.choice(SEASONS, n)

    is_bore = source == "Borewell"
    is_open = source == "Open well"
    is_pipe = source == "Piped supply"
    depth = np.where(is_bore, rng.gamma(4, 25, n) + 30,
             np.where(is_open, rng.uniform(5, 18, n),
             np.where(is_pipe, rng.uniform(0, 5, n), rng.uniform(15, 60, n))))
    rain = np.where(season == "Monsoon", rng.normal(220, 60, n),
           np.where(season == "Post-monsoon", rng.normal(110, 40, n), rng.normal(25, 15, n))).clip(0)
    dist_san = rng.gamma(2.5, 18, n).clip(2, 300)

    ph = rng.normal(7.4, 0.45, n) + 0.25 * (f_b - 1) + np.where(land == "Industrial", -0.4, 0)
    hardness = (rng.normal(260, 90, n) * sal_b - 40 * (f_b - 1)).clip(30)
    ec = (rng.lognormal(6.7, 0.45, n) * sal_b * np.where(is_pipe, 0.5, 1.0)
          + np.where(land == "Industrial", 400, 0)).clip(80)
    tds = ec * rng.normal(0.65, 0.03, n)
    turb = (rng.gamma(1.6, 1.3, n) * np.where(is_open, 2.2, 1.0)
            * np.where(season == "Monsoon", 1.8, 1.0)).clip(0.1)
    temp = rng.normal(27, 2.5, n) + np.where(season == "Pre-monsoon", 2.5, 0)
    do = (rng.normal(6.5, 1.0, n) - 0.012 * depth.clip(0, 150) - 0.15 * (turb > 5)).clip(0.5)

    # --- Lab parameters (what we want to predict without testing) -----------
    fluoride = (rng.lognormal(-0.6, 0.3, n) * f_b * (1 + 0.004 * depth)
                * (1 + 0.6 * (ph - 7.4)) * (1 - 0.0006 * (hardness - 260))).clip(0.05)
    arsenic = (rng.lognormal(-5.6, 0.4, n) * (as_b + 0.15)
               * np.where((depth > 10) & (depth < 80), 1.6, 0.6)).clip(0.0005)
    nitrate = (rng.lognormal(2.45, 0.3, n) * no3_b
               * np.where(land == "Agricultural", 1.7, 1.0)
               * np.where(depth < 25, 1.5, 0.8) * (1 + 60 / (dist_san + 20) * 0.3)).clip(0.5)
    iron = (rng.lognormal(-1.2, 0.8, n) * (1 + 0.6 * as_b) * np.where(do < 4, 1.8, 1.0)).clip(0.02)
    chloride = (tds * rng.normal(0.28, 0.06, n)).clip(5)
    sulphate = (tds * rng.normal(0.17, 0.05, n) + np.where(land == "Industrial", 80, 0)).clip(3)
    col_lambda = (col_b * 0.6 * np.where(is_open, 4, np.where(is_pipe, 0.5, 1.2))
                  * np.where(season == "Monsoon", 2.5, 1.0) * (40 / (dist_san + 10)) * (1 + 0.15 * turb))
    col_risk = np.log(col_lambda + 1e-6) + rng.normal(0, 0.45, n)
    coliform = np.where(col_risk > 1.25, rng.poisson(np.exp(col_risk.clip(None, 4))) + 1, 0)

    df = pd.DataFrame({
        "Sample_ID": [f"WQ{i:05d}" for i in range(1, n + 1)],
        "District": district,
        "State": [DISTRICTS[d][0] for d in district],
        "Source_Type": source, "Land_Use": land, "Season": season,
        "pH": ph.round(2), "EC": ec.round(0), "TDS": tds.round(0),
        "Turbidity": turb.round(2), "Temperature": temp.round(1),
        "Dissolved_Oxygen": do.round(2), "Total_Hardness": hardness.round(0),
        "Well_Depth_m": depth.round(1), "Dist_to_Sanitation_m": dist_san.round(0),
        "Rainfall_mm": rain.round(0),
        "Fluoride": fluoride.round(2), "Arsenic": arsenic.round(4), "Nitrate": nitrate.round(1),
        "Iron": iron.round(2), "Chloride": chloride.round(0), "Sulphate": sulphate.round(0),
        "Total_Coliform": coliform,
    })
    return label_water(df)


def exceedances(df):
    """Boolean frame: which BIS permissible limits each sample breaches."""
    e = pd.DataFrame(index=df.index)
    e["pH"] = (df["pH"] < 6.5) | (df["pH"] > 8.5)
    for p in ["TDS", "Total_Hardness", "Turbidity", "Fluoride", "Nitrate", "Iron", "Chloride", "Sulphate"]:
        e[p] = df[p] > BIS_LIMITS[p]["permissible"]
    e["Arsenic"] = df["Arsenic"] > 0.01
    e["Total_Coliform"] = df["Total_Coliform"] > 0
    return e


def label_water(df):
    e = exceedances(df)
    health = e[["Fluoride", "Arsenic", "Nitrate", "Total_Coliform"]].any(axis=1)
    n_exc = e.sum(axis=1)
    df = df.copy()
    df["Exceedance_Count"] = n_exc
    df["Potability"] = np.where(health | (n_exc >= 2), "Unsafe", "Safe")
    df["Risk_Level"] = np.select([health & (n_exc >= 3), health | (n_exc >= 2)],
                                 ["High", "Moderate"], "Low")
    df["WQI"] = water_quality_index(df).round(1)
    return df


def water_quality_index(df):
    """Weighted Arithmetic Water Quality Index (Brown et al., 1972)."""
    params = {"pH": 8.5, "TDS": 500, "Total_Hardness": 200, "Turbidity": 5, "Fluoride": 1.0,
              "Nitrate": 45, "Iron": 1.0, "Chloride": 250, "Sulphate": 200}
    ideal = {"pH": 7.0}
    k = 1 / sum(1 / s for s in params.values())
    num, den = 0, 0
    for p, s in params.items():
        w = k / s
        vi = ideal.get(p, 0)
        q = 100 * (df[p] - vi) / (s - vi)
        num = num + w * q.clip(lower=0)
        den += w
    return num / den


def wqi_class(w):
    if w <= 25: return "Excellent"
    if w <= 50: return "Good"
    if w <= 75: return "Poor"
    if w <= 100: return "Very poor"
    return "Unfit for drinking"


# ---------------------------------------------------------------------------
def loan_data(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 66, n)
    income = rng.lognormal(13.1, 0.55, n).round(-3)            # annual income, INR
    emp_years = np.clip(rng.gamma(2.2, 3.0, n), 0, age - 20).round(1)
    credit = np.clip(rng.normal(690, 75, n), 300, 900).round(0)
    loan_amt = (income * rng.uniform(0.3, 3.0, n)).round(-3)
    term = rng.choice([12, 24, 36, 60, 84], n, p=[.1, .2, .3, .3, .1])
    rate = np.clip(22 - (credit - 300) / 600 * 14 + rng.normal(0, 1.2, n), 7.5, 26).round(2)
    dti = np.clip(rng.beta(2.2, 5, n) * 0.9, 0.02, 0.85).round(3)
    lines = rng.poisson(4, n) + 1
    prev = rng.binomial(1, 0.08, n)
    purpose = rng.choice(["Home", "Education", "Vehicle", "Business", "Personal", "Medical"], n)
    home = rng.choice(["Own", "Rent", "Mortgage"], n, p=[.3, .45, .25])
    emi = loan_amt * (rate / 1200) / (1 - (1 + rate / 1200) ** -term)
    emi_ratio = (emi * 12 / income)
    z = (-3.1 - 0.012 * (credit - 690) + 3.2 * dti + 1.6 * emi_ratio + 1.7 * prev
         - 0.06 * emp_years - 0.012 * (age - 40) + 0.09 * (rate - 14)
         + np.where(purpose == "Business", 0.45, 0) + np.where(home == "Rent", 0.25, 0)
         + rng.normal(0, 0.6, n))
    default = rng.binomial(1, 1 / (1 + np.exp(-z)))
    return pd.DataFrame({
        "Age": age, "Annual_Income": income, "Employment_Years": emp_years,
        "Credit_Score": credit, "Loan_Amount": loan_amt, "Loan_Term_Months": term,
        "Interest_Rate": rate, "Debt_to_Income": dti, "Num_Credit_Lines": lines,
        "Previous_Default": prev, "Loan_Purpose": purpose, "Home_Ownership": home,
        "Default": np.where(default == 1, "Yes", "No"),
    })


def customer_data(n=400, seed=21):
    """Mall-customer style data with 5 latent segments."""
    rng = np.random.default_rng(seed)
    centers = [(45, 25, 20), (25, 25, 80), (42, 55, 50), (33, 88, 82), (44, 88, 17)]
    sizes = rng.multinomial(n, [.15, .15, .35, .18, .17])
    rows = []
    for (a, inc, sp), s in zip(centers, sizes):
        rows.append(np.column_stack([rng.normal(a, 8, s), rng.normal(inc, 8, s), rng.normal(sp, 8, s)]))
    X = np.vstack(rows)
    X[:, 0] = X[:, 0].clip(18, 70)
    X[:, 1] = X[:, 1].clip(12, 140)
    X[:, 2] = X[:, 2].clip(1, 99)
    idx = rng.permutation(len(X))
    X = X[idx]
    return pd.DataFrame({
        "CustomerID": np.arange(1, len(X) + 1),
        "Gender": rng.choice(["Male", "Female"], len(X)),
        "Age": X[:, 0].round(0).astype(int),
        "Annual_Income_kINR": (X[:, 1] * 10).round(0).astype(int) / 10 * 10,
        "Spending_Score": X[:, 2].round(0).astype(int),
    })


def house_data(n=500, seed=11):
    """For Multiple Linear Regression: house price (lakh INR) from property features."""
    rng = np.random.default_rng(seed)
    area = rng.normal(1400, 450, n).clip(450, 3500).round(0)
    beds = np.clip(np.round(area / 550 + rng.normal(0, 0.6, n)), 1, 5).astype(int)
    baths = np.clip(beds - rng.integers(0, 2, n), 1, 4)
    age = rng.integers(0, 35, n)
    dist = rng.gamma(2.2, 3.5, n).round(1)
    parking = rng.integers(0, 3, n)
    price = (8 + 0.042 * area + 3.5 * beds + 2.8 * baths - 0.45 * age - 1.6 * dist
             + 2.5 * parking + rng.normal(0, 6, n)).clip(10).round(2)
    return pd.DataFrame({"Area_sqft": area, "Bedrooms": beds, "Bathrooms": baths,
                         "Age_years": age, "Distance_to_City_km": dist,
                         "Parking_Slots": parking, "Price_Lakh": price})
