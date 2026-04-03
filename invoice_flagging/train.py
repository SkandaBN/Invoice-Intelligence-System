from model_evaluation import trainrandom_forest, evaluate_model
from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    split_data,
    scale_features,
)
import joblib

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

TARGET = "flag_invoice"

def main():

    # Load Data
    df = load_invoice_data()
    df = apply_labels(df)

    #Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test,'models/scaler.pkl'
        )
    
    #Train and evaluate models
    grid_search = trainrandom_forest(X_train_scaled, y_train)

    evaluate_model(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        'Random Forest Classifier'
        )

    #Save best model
    joblib.dump(grid_search.best_estimator_, 'models/random_forest_model.pkl')

if __name__ == "__main__":
    main()
