# Real Estate Investment Advisor -- Full Project Documentation

## Overview

This project builds a complete **ML-powered Real Estate Price Prediction
App** using: - **Python** - **Pandas** - **Scikit-learn** -
**Streamlit** - **Gradient Boosting Regressor** - **One-hot encoding &
label encoding** - **Data cleaning & feature engineering**

The goal is to allow users to: - Explore **data insights** - Predict
**property price** - Use a clean & user-friendly web UI

------------------------------------------------------------------------

## 1. Data Loading

### Problem:

The dataset was initially loaded using a long absolute path.

### Fix:

Use a simple import:

``` python
df = pd.read_csv("india_housing_prices.csv")
```

------------------------------------------------------------------------

## 2. Initial Sanity Checks

Performed: - Shape check\
- Column datatype check\
- Missing value scan\
- Outlier scan

### Findings:

-   Some columns had incorrect data (e.g., **Price_per_SqFt NaNs**)
-   Floor numbers inconsistent\
-   Categorical columns clean but huge cardinality in Locality

------------------------------------------------------------------------

## 3. Fixing Numeric Data Issues

### Issue: `Total_Floors` \< `Floor_No`

Solution: Swap only where needed.

------------------------------------------------------------------------

### Issue: `Year_Built` had some values outside expected range (e.g., 2018--2023)

Solution: Clip to valid range:

``` python
df['Year_Built'] = df['Year_Built'].clip(1990, 2023)
```

------------------------------------------------------------------------

### Issue: `Price_per_SqFt` had NaNs

Solution: Fill NaN using **median** rather than mean because: - Median
is resistant to extreme values - Price_per_SqFt distribution was
right-skewed

``` python
df['Price_per_SqFt'] = df['Price_per_SqFt'].fillna(df['Price_per_SqFt'].median())
```

------------------------------------------------------------------------

## 4. Fixing Categorical Columns

### Issue: Locality had 500 categories → not suitable for one-hot encoding

Solution: ❌ Removed entirely from ML features\
Removed from UI as well

------------------------------------------------------------------------

### Property_Type Fix

Dataset had: - Villa\
- Independent House\
Model UI needed 3rd category: - Apartment (default → no one-hot column
active)

New rules: - Apartment → both one-hot columns = 0 - Independent House →
set its column = 1\
- Villa → set its column = 1

------------------------------------------------------------------------

## 5. Encoding Strategy

### Label Encoding:

-   Locality (later removed)

### One‑Hot Encoding:

-   State\
-   City\
-   Property_Type\
-   Amenities

------------------------------------------------------------------------

## 6. Amenities Encoding

Converted multi-valued text into:

    Amenity_Pool
    Amenity_Playground
    Amenity_Garden
    Amenity_Gym
    Amenity_Clubhouse

------------------------------------------------------------------------

## 7. Handling Nearby Features

### Issue:

Dataset values = 1--10\
UI expected Yes/No

### Fix:

Use Selectbox (1--10)

``` python
st.selectbox("Nearby Schools (1–10)", list(range(1,11)))
```

------------------------------------------------------------------------

## 8. Splitting Train/Test

``` python
X_train, X_test, y_train, y_test = train_test_split(..., test_size=0.2)
```

------------------------------------------------------------------------

## 9. NaN Detection Prior to ML

Found:

    Price_per_SqFt    2534

Fixed using median imputation → **all missing removed**.

------------------------------------------------------------------------

## 10. ML Model Training

Used:

``` python
GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6
)
```

------------------------------------------------------------------------

## 11. Performance Issues

### Issue:

Model training took too long\
Cause: Large dataset (250k rows × 80+ columns)

### Fix:

-   Reduced unnecessary columns\
-   Improved datatypes\
-   Removed Locality to speed up training

------------------------------------------------------------------------

## 12. Streamlit App Issues & Solutions

### Issue 1

`KeyError: Property_Type`

Cause:\
Wrong reference after encoding.

Fix:\
Ensure all user-input fields use original raw column names.

------------------------------------------------------------------------

### Issue 2

`ValueError: The truth value of an empty array is ambiguous`

Cause:\
Conditional check comparing arrays incorrectly.

Fix:\
Replaced with:

``` python
if k in user_input and user_input[k] not in (None, "", []):
```

------------------------------------------------------------------------

### Issue 3

Locality dropdown showing all values\
Fix:\
Removed Locality from UI entirely.

------------------------------------------------------------------------

### Issue 4

Prediction crashes because feature order mismatched\
Fix:\
Recreated a **feature template** from training data and ensured final
row matches order.

------------------------------------------------------------------------

## 13. Final App Features

### Pages:

1.  **Price Prediction**
2.  **Data Insights**
3.  **About Project**

### Data Insights:

-   Displays existing property metrics\
-   Uses model prediction\
-   Summary statistics

### Prediction Engine:

-   State\
-   City\
-   BHK\
-   Size in SqFt\
-   Property Type\
-   Furnishing\
-   Floor & Total Floors\
-   Amenities\
-   Nearby schools/hospitals\
-   Owner Type\
-   Availability

------------------------------------------------------------------------

## 14. Final Files

-   `app.py` (Streamlit app)
-   `model.pkl` (trained model)
-   `template.pkl` (feature template)
-   `encoder.pkl` (saved encoders)
-   `cleaned_df.csv` (clean dataset)
-   `README.md` (this file)

------------------------------------------------------------------------

## 15. Conclusion

This project now includes: - Clean ML-ready dataset\
- Fully functional prediction app\
- Accurate price prediction\
- Stable UI\
- Robust error handling\
- Detailed EDA

You can now deploy it directly with:

    streamlit run app.py
