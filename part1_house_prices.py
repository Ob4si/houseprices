"""
Part 1: Predict House Prices Based on Square Footage and Location
--------------------------------------------------------------------
Model: Linear Regression (scikit-learn)

DATA SOURCE NOTE:
The dataset below is SYNTHETICALLY GENERATED (n=150) using numpy, built to
mimic realistic real-estate patterns (base price per sq ft, location premiums,
bedroom count effects, and random noise) rather than copied from a single
public file. If your instructor's announcement points to a specific dataset
(e.g., a Kaggle "House Prices" CSV or Ames Housing dataset), replace the
`generate_house_data()` function below with:
    df = pd.read_csv("your_downloaded_file.csv")
and adjust the column names accordingly. Everything downstream (encoding,
training, evaluation) will still work.

Improvements over generic starter code:
  - 150 records instead of a handful of rows
  - Added a `bedrooms` and `age_years` feature (more realistic than sq ft + location alone)
  - Train/test split with evaluation metrics (R^2, RMSE) instead of just one prediction
  - Reproducible via a fixed random seed
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_SEED = 42


def generate_house_data(n_records: int = 150) -> pd.DataFrame:
    """Generate a realistic synthetic house-price dataset.

    Price model (roughly grounded in real market patterns):
        price = base_price_per_location
              + (price_per_sqft) * square_footage
              + bedroom_premium * bedrooms
              - depreciation * age_years
              + noise
    """
    rng = np.random.default_rng(RANDOM_SEED)

    locations = ["Downtown", "Suburb", "Rural", "Uptown"]
    # Different base price/sqft multiplier per location (location premium)
    location_price_per_sqft = {
        "Downtown": 350,
        "Uptown": 275,
        "Suburb": 200,
        "Rural": 120,
    }
    location_base = {
        "Downtown": 150_000,
        "Uptown": 100_000,
        "Suburb": 60_000,
        "Rural": 30_000,
    }

    location_choices = rng.choice(locations, size=n_records, p=[0.25, 0.25, 0.3, 0.2])
    square_footage = rng.normal(1800, 550, size=n_records).clip(600, 5000).round(0)
    bedrooms = rng.integers(1, 6, size=n_records)
    age_years = rng.integers(0, 60, size=n_records)

    price = np.zeros(n_records)
    for i in range(n_records):
        loc = location_choices[i]
        price[i] = (
            location_base[loc]
            + location_price_per_sqft[loc] * square_footage[i]
            + 8_000 * bedrooms[i]
            - 500 * age_years[i]
            + rng.normal(0, 15_000)  # market noise
        )

    price = price.clip(min=40_000).round(2)

    return pd.DataFrame(
        {
            "square_footage": square_footage,
            "bedrooms": bedrooms,
            "age_years": age_years,
            "location": location_choices,
            "price": price,
        }
    )


def main():
    df = generate_house_data(150)
    print(f"Dataset shape: {df.shape}")
    print(df.head(), "\n")

    X = df[["square_footage", "bedrooms", "age_years", "location"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("location_ohe", OneHotEncoder(drop="first"), ["location"]),
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: ${rmse:,.2f}")
    print(f"Test R^2:  {r2:.3f}\n")

    # Predict price for a 2000 sq ft house in Downtown, 3 bed, 10 years old
    new_house = pd.DataFrame(
        {
            "square_footage": [2000],
            "bedrooms": [3],
            "age_years": [10],
            "location": ["Downtown"],
        }
    )
    predicted_price = model.predict(new_house)[0]
    print(f"Predicted price for 2000 sq ft, 3bd, 10yr old house in Downtown: "
          f"${predicted_price:,.2f}")

    # Inspect coefficients
    ohe = model.named_steps["preprocess"].named_transformers_["location_ohe"]
    location_feature_names = ohe.get_feature_names_out(["location"])
    all_feature_names = list(location_feature_names) + ["square_footage", "bedrooms", "age_years"]
    coefs = model.named_steps["regressor"].coef_
    print("\nModel coefficients:")
    for name, coef in zip(all_feature_names, coefs):
        print(f"  {name}: {coef:,.2f}")


if __name__ == "__main__":
    main()
