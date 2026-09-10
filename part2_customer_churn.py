"""
Part 2: Predict Customer Churn
--------------------------------------------------------------------
Model: Logistic Regression (scikit-learn)

DATA SOURCE NOTE:
The dataset below is SYNTHETICALLY GENERATED (n=200) using numpy, with churn
probability driven by realistic factors (low usage, high customer-service
call volume, and short tenure all increase churn risk -- this mirrors
patterns seen in real telecom/subscription churn datasets such as the
IBM Telco Customer Churn dataset). If your instructor's announcement points
to a specific real dataset, swap `generate_churn_data()` for:
    df = pd.read_csv("your_downloaded_file.csv")

Improvements over generic starter code:
  - 200 records instead of a handful
  - Added `tenure_months` and `contract_type` features (major real churn drivers)
  - Train/test split with accuracy, precision, recall, and confusion matrix
  - Reproducible via a fixed random seed
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42


def generate_churn_data(n_records: int = 200) -> pd.DataFrame:
    """Generate a realistic synthetic customer-churn dataset."""
    rng = np.random.default_rng(RANDOM_SEED)

    regions = ["North", "South", "East", "West"]
    contract_types = ["Month-to-month", "One year", "Two year"]

    age = rng.integers(18, 75, size=n_records)
    monthly_usage = rng.normal(50, 20, size=n_records).clip(1, 150)  # hours/units per month
    purchase_amount = rng.normal(80, 30, size=n_records).clip(5, 300)  # $ per month
    customer_service_calls = rng.poisson(2, size=n_records)
    tenure_months = rng.integers(1, 72, size=n_records)
    region = rng.choice(regions, size=n_records)
    contract_type = rng.choice(contract_types, size=n_records, p=[0.55, 0.25, 0.20])

    # Build churn probability from realistic risk factors, then sample a label
    contract_risk = np.select(
        [contract_type == "Month-to-month", contract_type == "One year", contract_type == "Two year"],
        [0.9, 0.0, -0.5],
    )
    logit = (
        -0.7
        + contract_risk
        + 0.25 * customer_service_calls
        - 0.04 * tenure_months
        - 0.02 * monthly_usage
        + 0.3 * (age < 25).astype(float)
        + rng.normal(0, 0.4, size=n_records)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, size=n_records) < churn_prob).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "monthly_usage": monthly_usage.round(1),
            "purchase_amount": purchase_amount.round(2),
            "customer_service_calls": customer_service_calls,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "region": region,
            "churn": churn,
        }
    )


def main():
    df = generate_churn_data(200)
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.1%}\n")
    print(df.head(), "\n")

    numeric_features = ["age", "monthly_usage", "purchase_amount", "customer_service_calls", "tenure_months"]
    categorical_features = ["region", "contract_type"]

    X = df[numeric_features + categorical_features]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Predict churn probability for a new customer
    new_customer = pd.DataFrame(
        {
            "age": [30],
            "monthly_usage": [15],
            "purchase_amount": [45],
            "customer_service_calls": [5],
            "tenure_months": [3],
            "contract_type": ["Month-to-month"],
            "region": ["South"],
        }
    )
    churn_probability = model.predict_proba(new_customer)[0][1]
    predicted_class = model.predict(new_customer)[0]
    print(f"\nNew customer churn probability: {churn_probability:.2f} "
          f"({'AT RISK' if predicted_class == 1 else 'not at risk'})")

    # Inspect coefficients
    ohe = model.named_steps["preprocess"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)
    coefs = model.named_steps["classifier"].coef_[0]
    print("\nModel coefficients (positive = increases churn risk):")
    for name, coef in zip(all_feature_names, coefs):
        print(f"  {name}: {coef:.3f}")


if __name__ == "__main__":
    main()
