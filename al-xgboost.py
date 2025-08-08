import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import shap
import os
import warnings
warnings.filterwarnings("ignore")


# === Configuration parameters ===
INIT_TRAIN_SIZE = 32
ACTIVE_ROUNDS = 20
BUDGET_PER_ROUND = 1200
SAMPLE_COST = 10
PATIENCE = 3
LAMBDA_REP = 0.7  # Weight between uncertainty and representativeness

# Bayesian optimization parameter bounds
BOUNDS = {
    'n_estimators': (50, 500),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 10),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'gamma': (0, 5)
}

def bo_evaluate(params, X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        n_estimators=int(params['n_estimators']),
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return r2_score(y_val, preds)

def estimate_uncertainty_shap(model, X_pool):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_pool)
    uncertainty = np.std(shap_values, axis=1)
    return uncertainty

def select_samples_with_budget(
    X_pool: np.ndarray,
    X_train: np.ndarray,
    uncertainty: np.ndarray,
    budget_per_round: int,
    sample_cost: int = 1,
    lambda_rep: float = 0.7
):
    train_center = X_train.mean(axis=0)
    distances = np.linalg.norm(X_pool - train_center, axis=1)
    norm_dist = distances / distances.max() if distances.max() > 0 else distances
    scores = lambda_rep * uncertainty + (1 - lambda_rep) * norm_dist
    sorted_idx = np.argsort(-scores)

    selected_idx = []
    total_cost = 0
    for idx in sorted_idx:
        if total_cost + sample_cost <= budget_per_round:
            selected_idx.append(idx)
            total_cost += sample_cost
        else:
            break
    return selected_idx

def active_learning_xgb(data_path):
    data = pd.read_csv(data_path)
    if data.isnull().values.any():
        data.fillna(data.median(), inplace=True)

    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split test set and remaining data
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # Split validation set and pool from remaining data
    X_pool, X_val, y_pool, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=1)
    # Select initial training set from pool
    X_train, X_pool, y_train, y_pool = train_test_split(X_pool, y_pool, train_size=INIT_TRAIN_SIZE, random_state=0)

    best_val_r2 = -np.inf
    rounds_no_improve = 0

    val_r2_list = []
    test_r2_list = []

    for round_i in range(ACTIVE_ROUNDS):
        print(f"\n=== Active Learning Round {round_i + 1} ===")

        # Bayesian optimization to find hyperparameters
        def bo_target(**params):
            return bo_evaluate(params, X_train, y_train, X_val, y_val)

        optimizer = BayesianOptimization(
            f=bo_target,
            pbounds=BOUNDS,
            random_state=round_i * 10,
            verbose=2
        )
        optimizer.maximize(init_points=3, n_iter=5)

        best_params = optimizer.max['params']
        best_params_int = {
            'n_estimators': int(best_params['n_estimators']),
            'learning_rate': best_params['learning_rate'],
            'max_depth': int(best_params['max_depth']),
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'gamma': best_params['gamma']
        }
        print(f"Best params: {best_params_int}")

        # Train model with best parameters
        model = XGBRegressor(**best_params_int, random_state=42, verbosity=0)
        model.fit(X_train, y_train)

        # Evaluate on validation set
        val_preds = model.predict(X_val)
        val_r2 = r2_score(y_val, val_preds)
        val_r2_list.append(val_r2)
        print(f"Validation R²: {val_r2:.4f}")

        # Evaluate on test set
        test_preds = model.predict(X_test)
        test_r2 = r2_score(y_test, test_preds)
        test_r2_list.append(test_r2)
        print(f"Test R²: {test_r2:.4f}")

        # Early stopping check
        if val_r2 > best_val_r2 + 1e-4:
            best_val_r2 = val_r2
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1
            print(f"No R² improvement for {rounds_no_improve} round(s)")
            if rounds_no_improve >= PATIENCE:
                print(f"Early stopping triggered after round {round_i + 1}. Best val R²: {best_val_r2:.4f}")
                break

        if len(X_pool) == 0:
            print("Pool exhausted. Stopping.")
            break

        # Calculate uncertainty
        uncertainty = estimate_uncertainty_shap(model, X_pool)

        # Unified sampling (uncertainty + representativeness)
        selected_idx = select_samples_with_budget(
            X_pool=X_pool,
            X_train=X_train,
            uncertainty=uncertainty,
            budget_per_round=BUDGET_PER_ROUND,
            sample_cost=SAMPLE_COST,
            lambda_rep=LAMBDA_REP
        )

        print(f"Selected {len(selected_idx)} samples.")

        # Update training set and pool
        X_selected = X_pool[selected_idx]
        y_selected = y_pool.iloc[selected_idx]

        X_train = np.vstack([X_train, X_selected])
        y_train = pd.concat([y_train, y_selected], ignore_index=True)

        mask = np.ones(len(X_pool), dtype=bool)
        mask[selected_idx] = False
        X_pool = X_pool[mask]
        y_pool = y_pool.reset_index(drop=True)[mask]

        print(f"Training set size after round {round_i + 1}: {len(X_train)}")

    # Final training
    final_model = XGBRegressor(**best_params_int, random_state=42, verbosity=0)
    final_model.fit(X_train, y_train)
    final_test_preds = final_model.predict(X_test)
    final_test_r2 = r2_score(y_test, final_test_preds)

    print(f"\nFinal training set size: {len(X_train)}")
    print(f"Final test R²: {final_test_r2:.4f}")

    # Save model and scaler
    os.makedirs("model_output", exist_ok=True)
    joblib.dump(final_model, "model_output/xgb_final_model.joblib")
    joblib.dump(scaler, "model_output/xgb_scaler.joblib")

    # Plot validation and test R² curves
    rounds = np.arange(1, len(val_r2_list) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, val_r2_list, marker='o', label='Validation R²')
    plt.plot(rounds, test_r2_list, marker='s', label='Test R²', color='orange')
    for x, y in zip(rounds, val_r2_list):
        plt.text(x, y + 0.01, f"{y:.3f}", ha='center', va='bottom', fontsize=8)
    for x, y in zip(rounds, test_r2_list):
        plt.text(x, y + 0.01, f"{y:.3f}", ha='center', va='bottom', fontsize=8)
    plt.xlabel('Active Learning Round')
    plt.ylabel('R² Score')
    plt.title('Validation and Test R² over Active Learning')
    plt.grid(True)
    plt.legend()
    plt.savefig("model_output/xgb_r2_curve.png", dpi=300)
    plt.show()

    return final_model, scaler

if __name__ == "__main__":
    model, scaler = active_learning_xgb("Br.csv")
    #model, scaler = active_learning_xgb("Hcj.csv")
