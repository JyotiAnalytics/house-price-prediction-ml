
import os
import streamlit as st
import pandas as pd
import joblib

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction System")

st.markdown("""
Estimate a property's price using a trained Machine Learning model.
Enter the property details below to generate a prediction.
""")

# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "best_house_price_model.joblib"
)

# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


if not os.path.exists(MODEL_PATH):
    st.error(
        "❌ Trained model not found!\n\n"
        "Please train the model first and save it to:\n"
        f"`{MODEL_PATH}`"
    )

    st.stop()

try:
    model = load_model(MODEL_PATH)

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

st.success("✅ Trained model loaded successfully!")

# ============================================================
# 4. INPUT FORM
# ============================================================

st.subheader("🏡 Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input(
        "Area (sq ft)",
        min_value=100,
        max_value=20000,
        value=1800,
        step=100
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=15,
        value=2,
        step=1
    )

with col2:
    stories = st.number_input(
        "Stories",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

    parking = st.number_input(
        "Parking spaces",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    location = st.selectbox(
        "Location",
        ["Urban", "Suburban", "Rural"]
    )

with col3:
    condition = st.selectbox(
        "Property condition",
        ["Poor", "Average", "Good", "Excellent"]
    )

    age = st.number_input(
        "Property age (years)",
        min_value=0,
        max_value=150,
        value=8,
        step=1
    )

    distance = st.number_input(
        "Distance from city center (km)",
        min_value=0.0,
        max_value=500.0,
        value=5.5,
        step=0.5
    )

# ============================================================
# 5. PREDICTION
# ============================================================

if st.button("🔮 Predict House Price", type="primary"):

        # ବଦଳାଇବା ପରେ ନୂଆ କୋଡ୍ ଏମିତି ହେବ:
    input_data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": stories,       
        "garage": parking,       
        "location": location,
        "condition": condition,
        "yearbuilt": 2026 - age  
    }])


    try:
        # Check the model's expected feature names when available.
        expected_features = getattr(
            model,
            "feature_names_in_",
            None
        )

        if expected_features is not None:
            expected_features = list(expected_features)

            missing_features = [
                feature
                for feature in expected_features
                if feature not in input_data.columns
            ]

            if missing_features:
                st.error(
                    "The input fields do not match the model's "
                    "training features.\n\n"
                    f"Missing features: {missing_features}"
                )

                st.warning(
                    "Update the input form to match the exact "
                    "columns used during model training."
                )

                st.stop()

            input_data = input_data[expected_features]

        prediction = float(model.predict(input_data)[0])

        if not pd.notna(prediction):
            st.error("The model returned an invalid prediction.")

        elif prediction < 0:
            st.warning(
                "The model predicted a negative price. "
                "Check the training data and model performance."
            )

            st.metric(
                "Predicted Price",
                f"{prediction:,.2f}"
            )

        else:
            st.success("✅ Prediction completed successfully!")

            # Display the predicted value without assuming currency.
            st.metric(
                label="Estimated House Price",
                value=f"{prediction:,.2f}"
            )

            st.info(
                "This is a machine-learning estimate, not a "
                "guaranteed market valuation. Accuracy depends "
                "on the training data and model performance."
            )

            with st.expander("View input details"):
                st.dataframe(
                    input_data,
                    use_container_width=True
                )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

        st.info(
            "Check that the input columns, data types, "
            "and category values match those used during training."
        )

# ============================================================
# 6. PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("📊 Project Information")

st.markdown("""
- **Domain:** Real Estate Analytics
- **Task:** Regression
- **Framework:** Streamlit
- **Model:** Trained Machine Learning estimator
- **Application:** Interactive house-price estimation
""")

st.caption(
    "For educational and analytical purposes. "
    "Predictions should be validated before practical use."
)

