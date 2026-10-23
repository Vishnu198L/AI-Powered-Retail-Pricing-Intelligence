
import os
import calendar
import requests
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Retail Pricing Intelligence",
    page_icon="📊",
    layout="wide",
)

# In Docker Compose, API_URL is set to http://backend:8000.
# When running Streamlit directly on Windows, localhost is used.
API_BASE_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

# Keep the existing prediction endpoint.
API_URL = f"{API_BASE_URL}/predict-demand"

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("📊 AI-Powered Retail Pricing Intelligence")
st.write(
    "Predict product demand using your trained "
    "Machine Learning model."
)

st.divider()

# --------------------------------------------------
# Check backend connection
# --------------------------------------------------
try:
    health_response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=5,
    )
    health_response.raise_for_status()
    backend_connected = True

except requests.exceptions.RequestException:
    backend_connected = False

if backend_connected:
    st.success("FastAPI backend is reachable.")
else:
    st.error(
        "Cannot connect to FastAPI. "
        "Check that the backend container is running "
        "and API_URL is configured correctly."
    )
    st.caption(f"Configured backend URL: {API_BASE_URL}")
    st.info(
        "For Docker Compose, API_URL should be "
        "http://backend:8000. Check compose.yaml and "
        "the backend logs if the problem continues."
    )

st.divider()

# --------------------------------------------------
# Input form
# --------------------------------------------------
st.subheader("Enter Product and Pricing Details")

with st.form("demand_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        category = st.selectbox(
            "Category",
            ["Electronics", "Clothing", "Groceries", "Furniture"],
        )

        region = st.selectbox(
            "Region",
            ["North", "South", "East", "West"],
        )

        inventory_level = st.number_input(
            "Inventory Level",
            min_value=0.0,
            value=100.0,
        )

        price = st.number_input(
            "Price",
            min_value=0.01,
            value=500.0,
        )

        discount = st.number_input(
            "Discount (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
        )

        competitor_pricing = st.number_input(
            "Competitor Pricing",
            min_value=0.01,
            value=480.0,
        )

    with col2:
        weather_condition = st.selectbox(
            "Weather Condition",
            ["Sunny", "Rainy", "Cloudy", "Snowy"],
        )

        promotion = st.selectbox(
            "Promotion",
            ["Yes", "No"],
        )

        seasonality = st.selectbox(
            "Seasonality",
            ["Spring", "Summer", "Autumn", "Winter"],
        )

        epidemic = st.selectbox(
            "Epidemic",
            ["No", "Yes"],
        )

        stock_status = st.selectbox(
            "Stock Status",
            ["Low", "Normal", "High"],
        )

        inventory_to_order_ratio = st.number_input(
            "Inventory to Order Ratio",
            min_value=0.0,
            value=1.5,
        )

    with col3:
        year = st.number_input(
            "Year",
            min_value=2000,
            max_value=2100,
            value=2026,
            step=1,
        )

        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=10,
            step=1,
        )

        day = st.number_input(
            "Day",
            min_value=1,
            max_value=31,
            value=13,
            step=1,
        )

        day_of_week = st.selectbox(
            "Day of Week (0 = Monday)",
            options=list(range(7)),
            index=1,
        )

        st.caption(
            "Price difference is calculated as "
            "your price minus competitor price."
        )

    # --------------------------------------------------
    # Derived features
    # --------------------------------------------------
    price_difference = price - competitor_pricing

    price_difference_percent = (
        price_difference / competitor_pricing
    ) * 100

    submitted = st.form_submit_button(
        "Predict Demand",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# Send input to FastAPI
# --------------------------------------------------
if submitted:

    # Validate the selected date
    try:
        days_in_month = calendar.monthrange(
            int(year),
            int(month),
        )[1]

        if int(day) > days_in_month:
            st.error(
                "Please enter a valid day for the selected month."
            )
            st.stop()

    except (ValueError, TypeError):
        st.error("Please enter a valid date.")
        st.stop()

    # Input data sent to FastAPI
    payload = {
        "Category": category,
        "Region": region,
        "Inventory Level": inventory_level,
        "Price": price,
        "Discount": discount,
        "Weather Condition": weather_condition,
        "Promotion": promotion,
        "Competitor Pricing": competitor_pricing,
        "Seasonality": seasonality,
        "Epidemic": epidemic,
        "Year": int(year),
        "Month": int(month),
        "Day": int(day),
        "DayOfWeek": int(day_of_week),
        "Price_Difference": price_difference,
        "Price_Difference_Percent": price_difference_percent,
        "Stock_Status": stock_status,
        "Inventory_to_Order_Ratio": inventory_to_order_ratio,
    }

    try:
        with st.spinner("Predicting product demand..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30,
            )

        if response.ok:
            result = response.json()

            if "predicted_demand" not in result:
                st.error(
                    "The API response does not contain "
                    "'predicted_demand'. Check the response "
                    "format returned by your FastAPI endpoint."
                )
                st.json(result)

            else:
                predicted_demand = result["predicted_demand"]

                st.success("Demand prediction completed!")

                st.subheader("Prediction Result")

                st.metric(
                    label="Predicted Demand",
                    value=f"{float(predicted_demand):,.2f}",
                )

                st.info(
                    "This is the demand estimated by your trained "
                    "model. It is a prediction, not a guarantee "
                    "of actual sales."
                )

                with st.expander("View submitted input"):
                    st.json(payload)

        else:
            st.error(
                f"API error: HTTP {response.status_code}"
            )
            st.code(response.text)

            if response.status_code == 404:
                st.warning(
                    "The endpoint /predict-demand was not found. "
                    "Check the route defined in backend/main.py."
                )

            elif response.status_code == 422:
                st.warning(
                    "The input data does not match the API schema. "
                    "Check the field names and data types expected "
                    "by backend/main.py."
                )

            elif response.status_code >= 500:
                st.warning(
                    "The backend encountered an error. "
                    "Check the backend container logs."
                )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. Verify that the backend "
            "container is running and that API_URL points to "
            "the correct Docker Compose service."
        )
        st.caption(f"Backend URL: {API_BASE_URL}")

    except requests.exceptions.Timeout:
        st.error(
            "The API request timed out. Please try again."
        )

    except (ValueError, KeyError, TypeError) as error:
        st.error(
            f"Could not read the prediction response: {error}"
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")

st.divider()
st.caption(
    "Powered by FastAPI, Streamlit, and Random Forest Regression."
)
