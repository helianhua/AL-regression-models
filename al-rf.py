import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import os

# === Configuration parameters ===
INIT_TRAIN_SIZE = 32
ACTIVE_ROUNDS = 20
BUDGET_PER_ROUND = 1200
SAMPLE_COST = 10
N_ROUNDS_NO_IMPROVE = 3
LAMBDA_REP = 0.7  # Weight between representativeness and uncertainty

# Bayesian optimization parameter bounds
BO_PARAM_BOUNDS = {
    'max_depth': (3, 20),
    'n_estimators': (50, 300)
}

def bo_evaluate(max_depth, n_estimators, X_train, y_train, X_val, y_val):
    max_depth = int(round(max_depth))
    n_estimators = int(round(n_estimators))
    model = RandomForestRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return r2_score(y_val, preds)

def calc_uncertainty(model, X):
    all_preds = np.stack([tree.predict(X) for tree in model.estimators_], axis=0)
    pred_var = np.var(all_preds, axis=0)
    return pred_var

# === Uncertainty + Representativeness combined sampling ===
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

# === Main workflow ===
def active_learning_rf_bayesopt(data_path):
    data = pd.read_csv(data_path)
    if data.isnull().values.any():
        data.fillna(data.median(), inplace=True)

    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_pool, X_val, y_pool, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=1)
    X_train, X_pool, y_train, y_pool = train_test_split(X_pool, y_pool, train_size=INIT_TRAIN_SIZE, random_state=0)

    best_r2 = -np.inf
    rounds_no_improve = 0

    val_r2_list = []
    test_r2_list = []

    for round_i in range(ACTIVE_ROUNDS):
        print(f"\n=== Active Learning Round {round_i + 1} ===")

        def bo_target(max_depth, n_estimators):
            return bo_evaluate(max_depth, n_estimators, X_train, y_train, X_val, y_val)

        optimizer = BayesianOptimization(
            f=bo_target,
            pbounds=BO_PARAM_BOUNDS,
            random_state=round_i * 10,
            verbose=2
        )
        optimizer.maximize(init_points=3, n_iter=5)

        best_params = optimizer.max['params']
        best_max_depth = int(round(best_params['max_depth']))
        best_n_estimators = int(round(best_params['n_estimators']))
        print(f"Best params: max_depth={best_max_depth}, n_estimators={best_n_estimators}")

        model = RandomForestRegressor(
            max_depth=best_max_depth,
            n_estimators=best_n_estimators,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred_val = model.predict(X_val)
        current_r2 = r2_score(y_val, y_pred_val)
        val_r2_list.append(current_r2)

        y_pred_test = model.predict(X_test)
        current_test_r2 = r2_score(y_test, y_pred_test)
        test_r2_list.append(current_test_r2)

        print(f"Validation R²: {current_r2:.4f} | Test R²: {current_test_r2:.4f}")

        if current_r2 > best_r2 + 1e-4:
            best_r2 = current_r2
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1
            print(f"No R² improvement for {rounds_no_improve} round(s)")

        if rounds_no_improve >= N_ROUNDS_NO_IMPROVE:
            print(f"\nEarly stopping triggered after round {round_i + 1}. Best val R²: {best_r2:.4f}")
            break

        if len(X_pool) == 0:
            print("Training pool exhausted. Stopping.")
            break

        # === Active sampling ===
        uncertainty = calc_uncertainty(model, X_pool)
        selected_idx = select_samples_with_budget(
            X_pool=X_pool,
            X_train=X_train,
            uncertainty=uncertainty,
            budget_per_round=BUDGET_PER_ROUND,
            sample_cost=SAMPLE_COST,
            lambda_rep=LAMBDA_REP
        )
        print(f"Selected {len(selected_idx)} samples based on uncertainty + representativeness.")

        X_selected = X_pool[selected_idx]
        y_selected = y_pool.iloc[selected_idx]
        X_train = np.vstack([X_train, X_selected])
        y_train = pd.concat([y_train, y_selected], ignore_index=True)

        mask = np.ones(len(X_pool), dtype=bool)
        mask[selected_idx] = False
        X_pool = X_pool[mask]
        y_pool = y_pool.reset_index(drop=True)[mask]

        print(f"Training set size after round {round_i + 1}: {len(X_train)}")

    # === Final training & saving ===
    final_model = RandomForestRegressor(
        max_depth=best_max_depth,
        n_estimators=best_n_estimators,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)
    y_pred_test = final_model.predict(X_test)
    final_test_r2 = r2_score(y_test, y_pred_test)

    print(f"\nFinal training set size: {len(X_train)}")
    print(f"Final test R²: {final_test_r2:.4f}")

    os.makedirs('model_output', exist_ok=True)
    joblib.dump(final_model, "model_output/rf_final_model.joblib")
    joblib.dump(scaler, "model_output/rf_scaler.joblib")

    # === Combined plot: Validation R² and Test R² ===
    rounds = range(1, len(val_r2_list) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, val_r2_list, marker='o', label='Validation R²', color='blue')
    plt.plot(rounds, test_r2_list, marker='s', label='Test R²', color='orange')
    plt.xlabel('Active Learning Round')
    plt.ylabel('R² Score')
    plt.title('Validation and Test R² over Active Learning Rounds')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("model_output/rf_r2_curve.png", dpi=300)
    plt.show()

    return final_model, scaler

# === Example run ===
if __name__ == "__main__":
    model, scaler = active_learning_rf_bayesopt("Br.csv")
    # model, scaler = active_learning_rf_bayesopt("Hcj.csv")
