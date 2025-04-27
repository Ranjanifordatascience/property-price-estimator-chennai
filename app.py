import streamlit as st
import pandas as pd
import joblib
import os

# Load model
model = joblib.load('xgboost_house_price_model.pkl')

# ---------- Custom Background and Style ----------
page_bg_img = """
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1505691938895-1758d7feb511");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    background-repeat: no-repeat;
    font-family: 'Arial', sans-serif;
}
.stTitle {
    color: black;
    text-align: center;
    font-size: 50px;
    font-weight: bold;
}
.stMarkdown {
    color: black;
    text-align: center;
    font-size: 24px;
}
.stButton > button {
    background-color: #1E90FF;  /* Blue color */
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 18px;
}
.stButton > button:hover {
    background-color: #4682B4;  /* Darker blue on hover */
}
.stSuccess {
    background-color: #4BB543;
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 22px;
    margin-top: 20px;
    font-weight: bold;
}
.stWarning {
    background-color: #ffcc00;
    padding: 10px;
    border-radius: 10px;
    color: black;
    text-align: center;
    font-size: 20px;
    margin-top: 10px;
    font-weight: bold;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------- App Title ----------
st.markdown('<h1 class="stTitle">🏡 Chennai House Price Estimator</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="stMarkdown">🔍 Find out your property\'s sale price by filling out the form below!</h2>',
            unsafe_allow_html=True)

st.markdown("---")

# ---------- Input Form ----------
with (st.container()):
    with st.form(key='form', clear_on_submit=True):

        # Row 1
        col1, col2 = st.columns(2)
        with col1:
            area = st.selectbox("Area",
                                ['Select Area', 'Velachery', 'Anna Nagar', 'Adyar', 'Chrompet', 'T Nagar', 'KK Nagar',
                                 'Karapakkam'])
        with col2:
            sqft = st.number_input("Total Sq. Ft.", min_value=500, max_value=2500, value=1200)

        st.markdown(" ")

        # Row 2
        col3, col4, col5 = st.columns(3)
        with col3:
            bedroom = st.number_input("Bedrooms", min_value=1, max_value=4, value=2)
        with col4:
            rooms = st.number_input("Total Rooms", min_value=2, max_value=6, value=3)
        with col5:
            sale_cond = st.selectbox("Sale Condition",
                                     ['Select Sale Condition', 'Regular Sale', 'quick sale', 'Family member sale', 'divided sale',
                                      'extra land sale'])

        st.markdown(" ")

        # Row 3
        col6, col7, col8 = st.columns(3)
        with col6:
            park_facil = st.selectbox("Parking Facility", ['Select Parking', 'Yes', 'No'])
        with col7:
            buildtype = st.selectbox("Building Type", ['Select Building Type', 'Commercial', 'House', 'Others'])
        with col8:
            utility = st.selectbox("Utility Availability", ['Select Utility', 'electricity, water and drainage', 'electricity only', 'no drainage only septic tank', 'no drainage facility'])

        st.markdown(" ")

        # Row 4
        col9, col10, col11 = st.columns(3)
        with col9:
            street = st.selectbox("Street Type", ['Select Street Type', 'smooth road', 'stone road', 'No road'])
        with col10:
            mz_zone = st.selectbox("Municipal Zone", ['Select Municipal Zone', 'Agricultural zone', 'Commercial Zone','Industrial zone', 'Apartments', 'Villa', 'Townhouse'])
        with col11:
            commis = st.number_input("Commission (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

        st.markdown(" ")


        # Submit Button
        submit_button = st.form_submit_button("🔮 Predict Sale Price")

    if submit_button:
        # Error handling: Check for invalid selections
        if (area == 'Select Area' or bedroom == 'Select Bedrooms' or rooms == 'Select Rooms' or
                sale_cond == 'Select Sale Condition' or park_facil == 'Select Parking' or
                buildtype == 'Select Building Type' or utility == 'Select Utility' or street == 'Select Street Type' or mz_zone == 'Select Municipal Zone'):

            st.markdown('<div class="stWarning">⚠️ Please select valid options for all fields!</div>',
                        unsafe_allow_html=True)
        else:
            # Mapping for 'sale_cond'
            sale_cond_mapping = {
                'Regular Sale': 'Normal Sale',
                'quick sale': 'Abnormal Sale',
                'Family member sale': 'Family',
                'divided sale': 'Partial',
                'extra land sale': 'Adj Land'
            }

            # Apply the mapping to sale_cond
            sale_cond = sale_cond_mapping.get(sale_cond, sale_cond)  # Default to original if no mapping

            # Mapping for utility
            utility_mapping = {
                'electricity, water and drainage': 'Full Utility',
                'electricity only': 'Electricity Only',
                'no drainage only septic tank': 'No Drainage',
                'no drainage facility': 'No Drainage'
            }

            # Apply mapping to utility
            utility = utility_mapping.get(utility, utility)

            # Mapping for street
            street_mapping = {
                'smooth road': 'Smooth',
                'stone road': 'Stone',
                'No road': 'No Road'
            }

            # Apply mapping to street
            street = street_mapping.get(street, street)

            # Mapping for mz_zone
            mz_zone_mapping = {
                'Agricultural zone': 'Agricultural',
                'Commercial Zone': 'Commercial',
                'Industrial zone': 'Industrial',
                'Apartments': 'Apartment',
                'Villa': 'Villa',
                'Townhouse': 'Townhouse'
            }

            # Apply mapping to mz_zone
            mz_zone = mz_zone_mapping.get(mz_zone, mz_zone)

            # Prepare input data for prediction
            input_data = pd.DataFrame([{
                'AREA': area,
                'INT_SQFT': sqft,
                'N_BEDROOM': int(bedroom),
                'N_ROOM': int(rooms),
                'SALE_COND': sale_cond,
                'PARK_FACIL': park_facil,
                'BUILDTYPE': buildtype,
                'UTILITY_AVAIL': utility,
                'STREET': street,
                'MZZONE': mz_zone,
                'COMMIS': commis
            }])

            # One-hot encoding
            input_encoded = pd.get_dummies(input_data)

            # Align columns with model
            model_input_cols = model.get_booster().feature_names
            for col in model_input_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_input_cols]

            # Prediction
            prediction = model.predict(input_encoded)[0]

            # Show Result
            st.markdown(f'<div class="stSuccess">💰 **Estimated Sale Price:** ₹ {prediction:,.2f}</div>',
                        unsafe_allow_html=True)

            # Save the input data along with predicted sale price to a CSV file
            csv_file_path = "house_price_predictions.csv"

            # If the CSV file doesn't exist, create it with the appropriate headers
            if not os.path.exists(csv_file_path):
                df_columns = ['ID', 'area', 'sqft', 'n_bedroom', 'n_room', 'sale_cond', 'park_facil', 'buildtype',
                              'utility', 'street', 'mz_zone', 'commis', 'estimated_sale_price']
                df = pd.DataFrame(columns=df_columns)
                df.to_csv(csv_file_path, index=False)

            # Handle empty file case (read CSV and check if it's empty)
            try:
                df_existing = pd.read_csv(csv_file_path)
                if df_existing.empty:
                    new_id = 1  # If the file is empty, start from ID 1
                else:
                    new_id = df_existing.shape[0] + 1  # Assign the next available ID
            except pd.errors.EmptyDataError:
                new_id = 1  # If CSV is empty, start from ID 1

            # Prepare data to save
            data_to_save = {
                'ID': new_id,
                'area': area,
                'sqft': sqft,
                'n_bedroom': int(bedroom),
                'n_room': int(rooms),
                'sale_cond': sale_cond,
                'park_facil': park_facil,
                'buildtype': buildtype,
                'utility': utility,
                'street': street,
                'mz_zone': mz_zone,
                'commis': commis,
                'estimated_sale_price': prediction
            }

            # Append new data to the CSV file
            df_new_entry = pd.DataFrame([data_to_save])
            df_new_entry.to_csv(csv_file_path, mode='a', header=False, index=False)
