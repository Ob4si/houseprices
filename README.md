# Machine Learning Assignment: Regression, Classification & Clustering

Three notebooks that each apply a different machine learning technique with scikit-learn. Every notebook includes the code, the saved output, and a written explanation of the approach and results. Open any `.ipynb` file on GitHub to view it with outputs.

| Notebook | Problem | Model | Key result |
|---|---|---|---|
| [part1_house_prices.ipynb](part1_house_prices.ipynb) | Predict house prices from square footage, location, bedrooms, and age | Linear Regression | Test R² = 0.966, RMSE ≈ $43.6k |
| [part2_customer_churn.ipynb](part2_customer_churn.ipynb) | Predict whether a customer will churn | Logistic Regression | Balanced class weights raised churn recall from 0.14 to 0.71 |
| [part3_customer_segmentation.ipynb](part3_customer_segmentation.ipynb) | Group customers into segments for targeted marketing | K-Means Clustering | K = 3 via elbow method: one high-value segment and two lower-spend segments split by age |

## Data
All three datasets are synthetically generated with NumPy (fixed random seed for reproducibility) to reflect realistic patterns. Each notebook explains how its data was built.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook
```
Then open any notebook and choose **Run All**.

## Files
- `elbow_plot.png`: elbow and silhouette charts from Part 3
- `customer_segments.csv`: customer data with assigned cluster labels from Part 3
