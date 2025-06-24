import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
import time


# Build a random forest regressor model with outlier removal
def build_random_forest_model(data_path='house_prices_cleaned.csv', target_column='SalePrice',
                              n_estimators=100, max_depth=None, min_samples_split=2,
                              min_samples_leaf=1, random_state=42):
    # Load/Read cleaned data
    df = pd.read_csv(data_path)

    # Simple outlier removal based on thresholds
    original_shape = df.shape[0]
    df = df[(df['GrLivArea'] < 4000) & (df['SalePrice'] < 700000)]
    removed_count = original_shape - df.shape[0]
    print(f"Removed {removed_count} outliers based on thresholds")
    print(f"Remaining data points: {df.shape[0]}")

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


# Function to perform k-fold cross-validation
def perform_kfold_cv(model, X, y, n_folds=5, random_state=42):
    # Initialize KFold
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

    # Calculate and display average metrics
    print("\nCross-Validation Results:")
    print(f"Mean R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    print(f"Mean RMSE: {np.mean(rmse_scores):.2f} (±{np.std(rmse_scores):.2f})")
    print(f"Mean MAE: {np.mean(mae_scores):.2f} (±{np.std(mae_scores):.2f})")

    return r2_scores, rmse_scores, mae_scores


# New function to perform randomized search for optimal hyperparameters
def perform_randomized_search(X, y, cv=3, n_iter=10):
    print("\n===== Starting Randomized Search for Optimal Hyperparameters =====")
    start_time = time.time()

    # Define the parameter grid to search
    param_distributions = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # Create a base model
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Initialize the randomized search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,  # Number of parameter settings to try
        cv=cv,  # Reduced CV folds for speed
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1,  # Use all available cores
        random_state=42
    )

    # Perform the randomized search
    random_search.fit(X, y)

    # Get the best parameters and score
    best_params = random_search.best_params_
    best_score = np.sqrt(-random_search.best_score_)  # Convert back to RMSE

    # Get results as a DataFrame
    results = pd.DataFrame(random_search.cv_results_)

    # Calculate execution time
    execution_time = time.time() - start_time

    print(f"\nRandomized Search completed in {execution_time:.2f} seconds")
    print(f"Best RMSE: {best_score:.2f}")
    print(f"Best Parameters: {best_params}")

    # Create a model with the best parameters
    best_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        random_state=42,
        n_jobs=-1
    )

    return best_model, best_params, best_score, results


if __name__ == "__main__":
    # Build the model with outlier removal
    rf_model, X, y = build_random_forest_model()

    print(f"Initial Random Forest model built with {rf_model.n_estimators} trees")
    print(f"Features: {X.shape[1]}")
    print(f"Target: {y.name}")

    # Perform randomized search to find optimal hyperparameters
    # Only try 10 combinations with 3-fold CV for much faster execution
    best_model, best_params, best_score, search_results = perform_randomized_search(X, y, cv=3, n_iter=10)

    print("\n===== Using Optimized Model with Best Parameters =====")

    # Train the optimized model
    best_model, X_train, X_test, y_train, y_test = train_model(best_model, X, y)

    print(f"Optimized model trained successfully")

    # Make predictions
    y_test_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)

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

    # Perform k-fold cross-validation with the optimized model
    r2_scores, rmse_scores, mae_scores = perform_kfold_cv(best_model, X, y, n_folds=5)

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features That Affect Sale Price:")
    print(feature_importance.head(10))

    # Save all results from randomized search
    print("\nAll Parameter Combinations Tried:")
    for i, row in search_results.sort_values('rank_test_score').iterrows():
        params = {key.replace('param_', ''): value for key, value in row.items() if
                  'param_' in key and not pd.isna(value)}
        rmse = np.sqrt(-row['mean_test_score'])
        print(f"Rank {row['rank_test_score']}: RMSE = {rmse:.2f}, Params: {params}")