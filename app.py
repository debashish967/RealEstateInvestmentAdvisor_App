# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# ---------- Config ----------
# Use relative paths so it works on Streamlit Cloud
CLEANED_GZ = "cleaned_housing_data.csv.gz"
MODEL_PATH = "model_gbr.pkl"
ENC_LOCALITY_PATH = "encoder_locality.pkl"   # optional
ONEHOT_COLS_PATH = "onehot_columns.pkl"      # optional (list)

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

# ---------- Utilities ----------
@st.cache_data(show_spinner=False)
def load_cleaned_df(path):
    return pd.read_csv(path, compression='gzip', low_memory=False)

@st.cache_data(show_spinner=False)
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_median(series):
    try:
        return float(series.median())
    except:
        return 0.0

def is_empty_val(val):
    """Safe emptiness check for user inputs"""
    if val is None:
        return True
    if isinstance(val, (list, tuple, set, np.ndarray)):
        return len(val) == 0
    if isinstance(val, float) and np.isnan(val):
        return True
    return (val == "")  # strings or exact empty

def build_feature_row_from_input(user_input, template_df, onehot_columns, encoders):
    """
    Build one-row DataFrame matching template_df.columns order.
    - user_input: dict with human-friendly values (strings / numbers / lists)
    - template_df: a 1-row DataFrame that contains the column order used when training
    - onehot_columns: list of one-hot column names present in template
    - encoders: dict of LabelEncoder objects for columns encoded during preprocessing
    """
    # start with medians (preferred) to avoid missing/model mismatch
    row = pd.Series(0.0, index=template_df.columns).astype(float)

    # fill medians first
    for c in template_df.columns:
        try:
            row[c] = float(safe_median(template_df[c]))
        except:
            # if cannot convert, leave as 0.0
            row[c] = 0.0

    # Numeric fields to prefer user inputs
    numeric_fields = ["ID","BHK","Size_in_SqFt","Price_per_SqFt","Year_Built",
                      "Floor_No","Total_Floors","Age_of_Property","Nearby_Schools","Nearby_Hospitals"]
    for k in numeric_fields:
        if k in row.index and k in user_input and not is_empty_val(user_input[k]):
            try:
                row[k] = float(user_input[k])
            except:
                # keep median
                row[k] = float(safe_median(template_df[k]))

    # Compute Price_per_SqFt from Price_in_Lakhs & Size if user gave price but not ppsqft
    if ("Price_in_Lakhs" in user_input) and (not is_empty_val(user_input.get("Price_in_Lakhs"))):
        try:
            price_l = float(user_input["Price_in_Lakhs"])
            size = float(user_input.get("Size_in_SqFt", safe_median(template_df["Size_in_SqFt"])))
            row["Price_per_SqFt"] = (price_l * 100000.0) / max(1.0, size)
        except:
            pass

    # Age from Year_Built
    if "Year_Built" in user_input and not is_empty_val(user_input.get("Year_Built")):
        try:
            row["Age_of_Property"] = max(0, 2023 - int(user_input["Year_Built"]))
        except:
            pass

    # Label-encoded columns: try to transform using encoders (if encoder exists)
    label_cols = [c for c in encoders.keys() if c in row.index]
    for col in label_cols:
        val = user_input.get(col, None)
        if val is not None and (not is_empty_val(val)):
            le = encoders.get(col)
            if isinstance(le, LabelEncoder):
                try:
                    row[col] = float(le.transform([str(val)])[0])
                except Exception:
                    # fallback to median
                    row[col] = float(safe_median(template_df[col]))
            else:
                # no encoder available; attempt common mappings
                sval = str(val)
                if col in ("Parking_Space", "Security"):
                    row[col] = 1.0 if sval.lower().startswith("y") else 0.0
                elif col == "Furnished_Status":
                    mapping = {"Unfurnished":0.0, "Semi-Furnished":1.0, "Semi-furnished":1.0, "Furnished":2.0}
                    row[col] = float(mapping.get(sval, safe_median(template_df[col])))
                elif col == "Public_Transport_Accessibility":
                    mapping = {"Low":0.0,"Medium":1.0,"High":2.0}
                    row[col] = float(mapping.get(sval, safe_median(template_df[col])))
                else:
                    # fallback
                    row[col] = float(safe_median(template_df[col]))
        else:
            # no user value -> keep median already set
            pass

    # One-hot columns (State_, City_, Property_Type_, Amenity_)
    for col in onehot_columns:
        if col not in row.index:
            continue
        # State
        if col.startswith("State_"):
            state_val = user_input.get("State", "")
            target = col.replace("State_","")
            row[col] = 1.0 if (state_val and target == state_val) else 0.0
        # City
        if col.startswith("City_"):
            city_val = user_input.get("City", "")
            target = col.replace("City_","")
            row[col] = 1.0 if (city_val and target == city_val) else 0.0
        # Property_Type: user picks "Apartment" / "Independent House" / "Villa"
        if col.startswith("Property_Type_"):
            pval = user_input.get("Property_Type", "")
            target = col.replace("Property_Type_","")
            # Apartment implies both encoded columns = 0
            row[col] = 1.0 if (pval and target == pval) else 0.0
        # Amenity columns
        if col.startswith("Amenity_"):
            amenities = user_input.get("Amenities", [])
            target = col.replace("Amenity_","")
            row[col] = 1.0 if (isinstance(amenities, (list, tuple)) and target in amenities) else 0.0

    # Locality: user opted to REMOVE locality selection from UI.
    # But model may expect a Locality feature. We'll set Locality to median value from template_df.
    if "Locality" in row.index:
        # template_df produced median already; keep that median (so no user input required)
        row["Locality"] = float(safe_median(template_df["Locality"]))

    # Reindex to ensure same order and no NaNs
    row = row.reindex(template_df.columns).fillna(0.0)
    return pd.DataFrame([row])

# ---------- Load resources ----------
with st.spinner("Loading model, encoders and cleaned dataset..."):
    # cleaned dataset
    try:
        cleaned_df = load_cleaned_df(CLEANED_GZ)
    except Exception as e:
        st.error(f"Could not load cleaned dataset at: {CLEANED_GZ}\n{e}")
        st.stop()

    # model
    try:
        model = load_pickle(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model at: {MODEL_PATH}\n{e}")
        st.stop()

    # optional pickles
    enc_locality = None
    try:
        enc_locality = load_pickle(ENC_LOCALITY_PATH)
    except Exception:
        enc_locality = None

    try:
        onehot_columns = load_pickle(ONEHOT_COLS_PATH)
    except Exception:
        # infer one-hot columns from cleaned_df
        onehot_columns = [c for c in cleaned_df.columns if c.startswith(("State_","City_","Property_Type_","Amenity_"))]

# ---------- Prepare feature template ----------
# Drop only the target (Price_in_Lakhs) from features - keep ID and Locality (model was trained with them)
feature_cols = [c for c in cleaned_df.columns if c != "Price_in_Lakhs"]
template_df = cleaned_df[feature_cols].head(1).copy()

# ---------- Prepare encoders for label-coded columns ----------
label_cols = ["Furnished_Status","Public_Transport_Accessibility","Parking_Space",
              "Security","Facing","Owner_Type","Availability_Status","Locality"]
encoders = {}
for col in label_cols:
    if col in cleaned_df.columns:
        # if column stored as numeric codes in cleaned_df, we still fit encoder on stringified values
        cleaned_df[col] = cleaned_df[col].astype(str)
        le = LabelEncoder()
        le.fit(cleaned_df[col].unique())
        encoders[col] = le

# ---------- UI ----------
st.title("🏠 Real Estate Investment Advisor — ML Edition")
tabs = st.tabs(["Price Prediction (New Buyer)", "Data Insights (Existing Property)"])

# ---------------- Price Prediction (New Buyer) ----------------
with tabs[0]:
    st.header("Price Prediction & 5-Year Projection (New Buyer)")

    # Ensure 'State' and 'City' exist or infer from one-hot
    if "State" not in cleaned_df.columns:
        state_cols = [c for c in cleaned_df.columns if c.startswith("State_")]
        if state_cols:
            def _state_from_row(r):
                for c in state_cols:
                    if r.get(c, 0) == 1:
                        return c.replace("State_","")
                return ""
            cleaned_df["State"] = cleaned_df.apply(_state_from_row, axis=1)
        else:
            cleaned_df["State"] = ""

    if "City" not in cleaned_df.columns:
        city_cols = [c for c in cleaned_df.columns if c.startswith("City_")]
        if city_cols:
            def _city_from_row(r):
                for c in city_cols:
                    if r.get(c, 0) == 1:
                        return c.replace("City_","")
                return ""
            cleaned_df["City"] = cleaned_df.apply(_city_from_row, axis=1)
        else:
            cleaned_df["City"] = ""

    # State -> City cascading dropdowns
    state_list = sorted([s for s in cleaned_df["State"].dropna().unique() if s != ""])
    selected_state = st.selectbox("Select State", options=["-- Select State --"] + state_list)

    city_list = []
    if selected_state and selected_state != "-- Select State --":
        city_list = sorted(cleaned_df.loc[cleaned_df["State"] == selected_state, "City"].dropna().unique())
    selected_city = st.selectbox("Select City", options=["-- Select City --"] + city_list)

    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        bhk_options = sorted(cleaned_df["BHK"].dropna().unique()) if "BHK" in cleaned_df.columns else [1,2,3]
        input_bhk = st.selectbox("BHK", options=bhk_options, index=0)
        input_size = st.number_input("Size (SqFt)", min_value=100, value=int(safe_median(cleaned_df["Size_in_SqFt"])), step=50)
        input_price = st.number_input("Optional: Proposed Price (Lakhs) — leave 0 to predict", min_value=0.0, value=0.0, step=0.5)
    with col2:
        # Property Type: show 3 options; map to one-hot cols
        pt_options = ["Apartment","Independent House","Villa"]
        input_prop_type = st.selectbox("Property Type", options=pt_options, index=0)
        # Furnished status UI
        furnished_opts = ["Unfurnished","Semi-furnished","Furnished"]
        input_furnished = st.selectbox("Furnished Status", options=furnished_opts)
        input_floor = st.number_input("Floor No", min_value=0, max_value=100, value=int(safe_median(cleaned_df["Floor_No"])))
    with col3:
        year_opts = sorted(cleaned_df["Year_Built"].dropna().unique(), reverse=True) if "Year_Built" in cleaned_df.columns else list(range(1990,2024))
        input_year_built = st.selectbox("Year Built", options=year_opts, index=0)
        # Parking & Security as human Yes/No
        parking_opts = ["Yes","No"]
        input_parking = st.selectbox("Parking Space", options=parking_opts)
        security_opts = ["Yes","No"]
        input_security = st.selectbox("Security", options=security_opts)

    # Nearby counts (1 to 10)
    schools_min, schools_max = 1, 10
    hospitals_min, hospitals_max = 1, 10
    input_schools = st.selectbox("Nearby Schools (1-10)", options=list(range(schools_min, schools_max+1)), index=int(safe_median(cleaned_df["Nearby_Schools"])-1))
    input_hospitals = st.selectbox("Nearby Hospitals (1-10)", options=list(range(hospitals_min, hospitals_max+1)), index=int(safe_median(cleaned_df["Nearby_Hospitals"])-1))

    # Amenities multi-select
    amenities_list = sorted([a.replace("Amenity_","") for a in onehot_columns if a.startswith("Amenity_")])
    selected_amenities = st.multiselect("Amenities (pick zero or more)", options=amenities_list)

    # Predict button
    if st.button("Predict Price"):
        # Build user input mapping
        user_input = {
            "State": selected_state if selected_state and selected_state != "-- Select State --" else "",
            "City": selected_city if selected_city and selected_city != "-- Select City --" else "",
            "BHK": input_bhk,
            "Size_in_SqFt": input_size,
            "Property_Type": input_prop_type,
            "Furnished_Status": input_furnished,
            "Floor_No": input_floor,
            "Year_Built": input_year_built,
            "Parking_Space": input_parking,
            "Security": input_security,
            "Nearby_Schools": input_schools,
            "Nearby_Hospitals": input_hospitals,
            "Amenities": selected_amenities,
            "Price_in_Lakhs": input_price if input_price > 0 else None
        }

        # Build feature row (Locality omitted from UI but median-filled inside builder)
        feature_row = build_feature_row_from_input(user_input, template_df, onehot_columns, encoders)

        # Ensure order matches template
        try:
            feature_row = feature_row[template_df.columns]
        except Exception:
            feature_row = feature_row.reindex(template_df.columns).fillna(0.0)

        # Predict
        try:
            pred_price = float(model.predict(feature_row)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.subheader("Prediction Result (for new buyer)")
        st.metric("Predicted Price (Lakhs)", f"{pred_price:,.2f}")
        future_price = pred_price * (1.08 ** 5)
        roi_pct = ((future_price - pred_price) / pred_price) * 100 if pred_price > 0 else 0.0
        st.write(f"Projected Price in 5 years (r=8%): **{future_price:,.2f} Lakhs**")
        st.write(f"Estimated ROI over 5 years: **{roi_pct:.2f}%**")

        # Feature importances if supported
        try:
            fi = model.feature_importances_
            fi_series = pd.Series(fi, index=template_df.columns).sort_values(ascending=False).head(15)
            fig = px.bar(fi_series[::-1], orientation='h', title="Top 15 Feature Importances")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

# ---------------- Data Insights (Existing Property) ----------------
with tabs[1]:
    st.header("Data Insights & Existing Property Lookup")

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total properties", f"{len(cleaned_df):,}")
    with col2:
        st.metric("Average Price (Lakhs)", f"{cleaned_df['Price_in_Lakhs'].mean():.2f}")
    with col3:
        st.metric("Median Price per SqFt", f"{cleaned_df['Price_per_SqFt'].median():.2f}")

    # Find property by State/City/filters
    sel_state2 = st.selectbox("State (existing)", options=["-- Select State --"] + sorted([s for s in cleaned_df["State"].dropna().unique() if s != ""]))
    sel_city2 = []
    if sel_state2 and sel_state2 != "-- Select State --":
        sel_city2 = sorted(cleaned_df.loc[cleaned_df["State"] == sel_state2, "City"].dropna().unique())
    sel_city2 = st.selectbox("City (existing)", options=["-- Select City --"] + sel_city2)

    # Filters
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        f_bhk = st.selectbox("BHK (filter)", options=["Any"] + sorted(cleaned_df["BHK"].dropna().unique().tolist()))
    with fcol2:
        f_min_price = st.number_input("Min Price (Lakhs)", min_value=0.0, value=0.0)
        f_max_price = st.number_input("Max Price (Lakhs)", min_value=0.0, value=float(cleaned_df["Price_in_Lakhs"].max()))
    with fcol3:
        f_size = st.number_input("Min Size (SqFt)", min_value=0, value=0)

    # Apply filters
    filtered = cleaned_df.copy()
    if sel_state2 and sel_state2 != "-- Select State --":
        filtered = filtered[filtered["State"] == sel_state2]
    if sel_city2 and sel_city2 != "-- Select City --":
        filtered = filtered[filtered["City"] == sel_city2]
    if f_bhk != "Any":
        filtered = filtered[filtered["BHK"] == f_bhk]
    filtered = filtered[(filtered["Price_in_Lakhs"] >= f_min_price) & (filtered["Price_in_Lakhs"] <= f_max_price)]
    if f_size > 0:
        filtered = filtered[filtered["Size_in_SqFt"] >= f_size]

    st.markdown(f"Filtered properties: {len(filtered):,}")
    st.dataframe(filtered.head(100), use_container_width=True)

    # Evaluate selected row
    if not filtered.empty:
        sel_index = st.selectbox("Pick a property by index (to predict its current & 5y price)", options=filtered.index.tolist())
        if st.button("Evaluate Selected Property"):
            selected_row = filtered.loc[[sel_index]]
            # Prepare user_input map from the selected row
            user_input_from_row = {}
            for c in ["BHK","Size_in_SqFt","Price_in_Lakhs","Price_per_SqFt","Year_Built",
                      "Floor_No","Total_Floors","Nearby_Schools","Nearby_Hospitals",
                      "State","City","Property_Type","Furnished_Status","Parking_Space","Security","Amenities"]:
                if c in selected_row.columns:
                    val = selected_row.iloc[0][c]
                    if pd.isna(val):
                        continue
                    if c == "Amenities" and isinstance(val, str):
                        user_input_from_row[c] = [a.strip() for a in val.split(",")]
                    else:
                        user_input_from_row[c] = val

            # Build feature row (Locality median-filled internally)
            feature_row2 = build_feature_row_from_input(user_input_from_row, template_df, onehot_columns, encoders)
            try:
                feature_row2 = feature_row2[template_df.columns]
            except:
                feature_row2 = feature_row2.reindex(template_df.columns).fillna(0.0)

            try:
                pred_price_cur = float(model.predict(feature_row2)[0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            future_price = pred_price_cur * (1.08 ** 5)
            roi_pct = ((future_price - pred_price_cur) / pred_price_cur) * 100 if pred_price_cur > 0 else 0.0

            st.subheader("Evaluation: Selected Property")
            st.write(f"Model Predicted Current Price: **{pred_price_cur:,.2f} Lakhs**")
            st.write(f"Projected Price in 5 years (r=8%): **{future_price:,.2f} Lakhs**")
            st.write(f"Estimated ROI over 5 years: **{roi_pct:.2f}%**")

# ---------- Footer ----------
st.markdown("---")
st.caption("Streamlit ML app — loads cleaned dataset and a pre-trained model. For production, consider model versioning with MLflow and serving.")
st.caption("Project is done by Debashish Borah")
