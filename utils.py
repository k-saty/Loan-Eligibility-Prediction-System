import pandas as pd
import joblib


def preprocessdata(
    Gender,
    Married,
    Education,
    Self_Employed,
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area,
):
    # Create a dictionary with values for the features
    test_data = {
        "Gender": Gender,
        "Married": Married,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area,
    }

    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([test_data])

    # Load the trained model
    trained_model = joblib.load("model.pkl")

    # Predict using the trained model
    prediction = trained_model.predict(input_df)

    return prediction
