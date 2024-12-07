# Model Prediction Section
elif section == "Model Prediction":
    st.title("🔮 Model Prediction")
    try:
        # Load model and related files
        model = joblib.load('models/vehicle_transmission_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/encoders.pkl')
        le_transmission = joblib.load('models/le_transmission.pkl')
        original_columns = joblib.load('models/original_columns.pkl')
    except Exception as e:
        st.error(f"Error loading files: {e}")
        model, scaler, encoders, le_transmission, original_columns = None, None, None, None, None

    if model:
        # Input for prediction
        st.subheader("Enter Vehicle Details:")

        dealer_type = st.selectbox("Dealer Type", merged_df['dealer_type'].unique())
        stock_type = st.selectbox("Stock Type", merged_df['stock_type'].unique())
        mileage = st.number_input("Mileage", min_value=0)
        price = st.number_input("Price", min_value=0)
        model_year = st.number_input("Model Year", min_value=2000, max_value=2024)
        make = st.selectbox("Make", merged_df['make'].unique())
        model_input = st.selectbox("Model", merged_df['model'].unique())
        certified = st.radio("Certified", ["Yes", "No"])
        fuel_type = st.selectbox("Fuel Type", merged_df['fuel_type_from_vin'].unique())
        price_changes = st.number_input("Number of Price Changes", min_value=0)
    
        input_data = pd.DataFrame(
            [
                {
                    "dealer_type": dealer_type,
                    "stock_type": stock_type,
                    "mileage": mileage,
                    "price": price,
                    "model_year": model_year,
                    "make": make,
                    "model": model_input,
                    "certified": 1 if certified == "Yes" else 0,
                    "fuel_type_from_vin": fuel_type,
                    "number_price_changes": price_changes,
                }
            ]
        )
    
        st.write("Input Data for Prediction:")
        st.write(input_data)

        if st.button("Generate Prediction"):
            try:
                # Reindex input data to match the original columns
                input_df = input_data.reindex(columns=original_columns, fill_value=0)

                # Apply the encoders to categorical features
                for col, encoder in encoders.items():
                    if col in input_df.columns:
                        try:
                            input_df[col] = encoder.transform(input_df[col].astype(str))
                        except KeyError:
                            # If category not seen during training, use a default encoding (e.g., -1 or the most frequent class)
                            input_df[col] = encoder.transform([input_df[col].mode()[0]])[0]

                # Scale the input data
                scaled_input = scaler.transform(input_df)

                # Predict transmission type
                prediction = model.predict(scaled_input)

                # Map predictions to the corresponding transmission type (Manual/Automatic)
                transmission_mapping = {0: "Automatic", 1: "Manual"}
                predicted_transmission = transmission_mapping.get(prediction[0], "Unknown")

                st.write(f"### Predicted Transmission: {predicted_transmission}")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
