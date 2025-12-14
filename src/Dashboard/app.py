import streamlit as st
import requests

# Define the API URL
# If running locally without Docker: "http://api:8000/predict"
# If running inside Docker Compose: "http://credit-scoring-api:8000/predict"
# We default to localhost for testing, but Docker will need the service name.
API_URL = "http://api:8000/predict"

st.set_page_config(page_title="Credit Risk Scoring", layout="centered")

st.title("💳 Credit score Scoring Dashboard")
st.markdown("Enter customer transaction details below to estimate credit risk.")

# Create a form for user input
with st.form("prediction_form"):
    st.header("Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input("Recency (Days since last Tx)", min_value=0, value=10)
    with col2:
        frequency = st.number_input("Frequency (Tx Count)", min_value=0, value=5)
    with col3:
        monetary = st.number_input("Monetary (Total Spend)", min_value=0.0, value=5000.0)

    # Add other inputs if your model uses them (e.g., Engagement Score)
    # For now, we stick to the basics for simplicity

    submit_button = st.form_submit_button("Assess Risk")

if submit_button:
    # Prepare the payload
    payload = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        # Defaulting others to 0 if not collected in UI
        "Engagement_Score": 0,
        "Channel_Diversity": 0
    }

    try:
        # Send request to API
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            prob = result.get("risk_probability", 0)  # Get probability FIRST

            # manually calculate the label right here
            if prob <= 0.20:  # You can adjust this threshold (e.g. 0.30)
                risk_label = "Good (Low Risk)"
            else:
                risk_label = "High Risk"
            score = result.get("credit_score", "N/A")

            # Display Result
            st.divider()
            st.subheader("Risk Assessment Result")

            # Color logic
            if risk_label == "High Risk":
                st.error(f"⚠️ {risk_label}")
            else:
                st.success(f"✅ {risk_label}")

            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Credit Score", score)
            col_res1.metric("Default Probability", f"{prob:.2%}")

        else:
            st.error(f"Error: {response.text}")

    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.info("Make sure the API is running and accessible.")