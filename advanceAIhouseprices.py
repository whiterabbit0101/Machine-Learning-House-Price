import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, KFold, cross_val_score



# Build a random forest regressor model from cleaned data. Target column is saleprice
def build_random_forest_model(data_path='house_prices_cleaned.csv', target_column='SalePrice',
                              n_estimators=300, max_depth=None, min_samples_split=2,
                              min_samples_leaf=1, random_state=42):

    # Load/Read cleaned data
    df = pd.read_csv(data_path)

    # Separate features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Build the Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )

    return model, X, y


# Function to train the model
def train_model(model, X, y, test_size=0.2, random_state=42):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train the model
    model.fit(X_train, y_train)

    print(f"Training data size: {X_train.shape[0]} samples")
    print(f"Testing data size: {X_test.shape[0]} samples")

    return model, X_train, X_test, y_train, y_test

def perform_kfold_cv(model, X, y, n_folds=5, random_state=42):

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Metrics to track
    r2_scores = []
    rmse_scores = []
    mae_scores = []

    print(f"\n===== {n_folds}-Fold Cross-Validation =====")

    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data for this fold
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        # Train model on this fold
        model.fit(X_train_fold, y_train_fold)

        # Make predictions
        y_pred_fold = model.predict(X_test_fold)

        # Calculate metrics
        r2 = r2_score(y_test_fold, y_pred_fold)
        rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
        mae = mean_absolute_error(y_test_fold, y_pred_fold)

        # Store metrics
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)

        print(f"Fold {fold + 1}: R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")

    # Show K Fold metrics
    print("\nK-Fold Cross-Validation Results:")
    print(f"Mean R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    print(f"Mean RMSE: {np.mean(rmse_scores):.2f} (±{np.std(rmse_scores):.2f})")
    print(f"Mean MAE: {np.mean(mae_scores):.2f} (±{np.std(mae_scores):.2f})")

    return r2_scores, rmse_scores, mae_scores

if __name__ == "__main__":
    # Build the model
    rf_model, X, y = build_random_forest_model()

    print(f"Random Forest model built with {rf_model.n_estimators} trees")
    print(f"Features: {X.shape[1]}")
    print(f"Target: {y.name}")

    # Train the model
    rf_model, X_train, X_test, y_train, y_test = train_model(rf_model, X, y)

    print(f"Model trained successfully")

    # Make predictions
    y_test_pred = rf_model.predict(X_test)
    y_train_pred = rf_model.predict(X_train)

    # Evaluation metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    # Print evaluation metrics
    print("\n===== Model Evaluation =====")
    print(f"R² Score: {test_r2:.4f}")
    print(f"RMSE: {test_rmse:.2f}")
    print(f"MAE: {test_mae:.2f}")
    print(f"MAPE: {test_mape:.2f}%")

    # Perform k-fold cross-validation
    r2_scores, rmse_scores, mae_scores = perform_kfold_cv(rf_model, X, y, n_folds=5)

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features That Affect Sale Price:")
    print(feature_importance.head(10))

