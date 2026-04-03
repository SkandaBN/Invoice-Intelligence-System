import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    """
    Predict whether a new vendor invoice should be flagged for review.

    Parameters
    ----------
    input_data : dict


    Returns
    -------
    pd.DataFrame with predicted flag
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    predictions = model.predict(input_df)
    input_df['Flagged_for_Review'] = predictions
    return input_df