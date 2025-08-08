import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVR
from qiskit_aer import AerSimulator
from bayes_opt import BayesianOptimization, UtilityFunction

# === Force FidelityQuantumKernel to use GPU simulator ===
gpu_sim = AerSimulator(method="statevector", device="GPU")
FidelityQuantumKernel._DEFAULT_SIMULATOR = gpu_sim
print("✅ FidelityQuantumKernel now uses GPU backend.")
print("   method:", FidelityQuantumKernel._DEFAULT_SIMULATOR.options.method)
print("   device:", FidelityQuantumKernel._DEFAULT_SIMULATOR.options.device)

# === Configuration Parameters ===
INIT_SIZE = 32
ACTIVE_ROUNDS = 14
PATIENCE = 3
MAX_FEATURES = 6
BUDGET_PER_ROUND = 100
REPS = 1
INIT_POINTS = 2
N_ITER = 5
SAVE_DIR = "models_Hcj"
os.makedirs(SAVE_DIR, exist_ok=True)

def feature_selection(X_raw, y, top_k=MAX_FEATURES):
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=0, importance_type='gain')  # or 'weight'
    xgb_model.fit(X_raw, y)
    importances = xgb_model.feature_importances_

    topk_idx = np.argsort(importances)[-top_k:]
    topk_features = X_raw.columns[topk_idx]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.barh(topk_features, importances[topk_idx], color='darkgreen')
    plt.xlabel('Feature Importance')
    plt.title('Top Features Importance (XGBoost)')
    plt.tight_layout()
    plt.show()

    return topk_features


def standardize(X, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def optimize_qsvr_params(X_train, y_train, X_val, y_val, fmap):
    def bo_func(C, epsilon):
        try:
            start_eval = time.time()  # start of evaluation
            model = QSVR(
                quantum_kernel=FidelityQuantumKernel(feature_map=fmap),
                C=max(C, 0.1),
                epsilon=max(epsilon, 1e-3)
            )
            model.fit(X_train, y_train)
            score = r2_score(y_val, model.predict(X_val))
            end_eval = time.time()
            print(f"  BO Eval: C={C:.3f}, eps={epsilon:.3f}, R2={score:.4f}, time={end_eval - start_eval:.2f} s")
            return score
        except Exception as e:
            print(f"  BO Eval failed: C={C:.3f}, eps={epsilon:.3f}, error: {e}")
            return 0

    print("▶ Starting Bayesian Optimization...")

    start_bo = time.time()  # overall start

    optimizer = BayesianOptimization(
        f=bo_func,
        pbounds={'C': (5, 25), 'epsilon': (0.1, 0.5)},
        random_state=42,
        allow_duplicate_points=True
    )
    utility = UtilityFunction(kind='ucb', kappa=2.5)
    optimizer.maximize(init_points=INIT_POINTS, n_iter=N_ITER, acquisition_function=utility)

    end_bo = time.time()  # overall end
    print(f"✅ Bayesian Optimization total time: {end_bo - start_bo:.2f} s")

    return optimizer.max['params']


def active_learning_qsvr(X_scaled, y, save_model_path=None, save_params_path=None):
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_pool, X_val, y_pool, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=1)
    X_train, X_pool, y_train, y_pool = train_test_split(X_pool, y_pool, train_size=INIT_SIZE, random_state=0)

    best_r2 = -np.inf
    rounds_no_improve = 0
    fmap = PauliFeatureMap(feature_dimension=X_train.shape[1], reps=REPS, entanglement='linear')
    best_params = {'C': 10, 'epsilon': 0.1}

    for round_i in range(ACTIVE_ROUNDS):
        print(f"\n=== Round {round_i+1}/{ACTIVE_ROUNDS} ===")
        best_params = optimize_qsvr_params(X_train, y_train, X_val, y_val, fmap)

        start_train = time.time()
        model = QSVR(quantum_kernel=FidelityQuantumKernel(feature_map=fmap),
                     C=best_params['C'], epsilon=best_params['epsilon'])
        model.fit(X_train, y_train)
        end_train = time.time()

        start_pred = time.time()
        val_score = r2_score(y_val, model.predict(X_val))
        test_score = r2_score(y_test, model.predict(X_test))
        end_pred = time.time()

        print(f"Round {round_i+1}: Val R2={val_score:.4f}, Test R2={test_score:.4f}")
        print(f"    Training time: {end_train - start_train:.2f} s, Prediction time: {end_pred - start_pred:.2f} s")

        if val_score > best_r2:
            best_r2 = val_score
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1
            if rounds_no_improve >= PATIENCE:
                break

        if len(X_pool) == 0:
            break

        preds = []
        for _ in range(3):
            boot_idx = np.random.choice(len(X_train), size=min(64, len(X_train)), replace=True)
            boot_model = QSVR(quantum_kernel=FidelityQuantumKernel(feature_map=fmap),
                              C=best_params['C'], epsilon=best_params['epsilon'])
            boot_model.fit(X_train[boot_idx], y_train.iloc[boot_idx])
            try:
                preds.append(boot_model.predict(X_pool))
            except:
                continue

        if not preds:
            break

        preds = np.array(preds)
        std = preds.std(axis=0)
        train_center = X_train.mean(axis=0)
        distances = np.linalg.norm(X_pool - train_center, axis=1)
        scores = 0.7 * std + 0.3 * (distances / distances.max())
        selected_idx = np.argsort(-scores)[:min(BUDGET_PER_ROUND, len(scores))]

        X_train = np.vstack([X_train, X_pool[selected_idx]])
        y_train = pd.concat([y_train, y_pool.iloc[selected_idx]], ignore_index=True)
        mask = np.ones(len(X_pool), dtype=bool)
        mask[selected_idx] = False
        X_pool, y_pool = X_pool[mask], y_pool.reset_index(drop=True)[mask]
        print(f"Selected {len(selected_idx)} new samples. Train size: {len(X_train)}")

    # === Measure final training and inference time ===
    start_train = time.time()
    final_model = QSVR(quantum_kernel=FidelityQuantumKernel(feature_map=fmap),
                       C=best_params['C'], epsilon=best_params['epsilon'])
    final_model.fit(X_train, y_train)
    end_train = time.time()

    start_pred = time.time()
    y_test_pred = final_model.predict(X_test)
    end_pred = time.time()

    final_train_time_sec = end_train - start_train
    final_pred_time_sec = end_pred - start_pred

    if save_model_path:
        joblib.dump(final_model, save_model_path)
    if save_params_path:
        with open(save_params_path, "w") as f:
            json.dump(best_params, f, indent=4)

    return {
        'model': final_model,
        'r2': r2_score(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'n_train': len(X_train),
        'final_train_time_sec': final_train_time_sec,
        'final_pred_time_sec': final_pred_time_sec
    }

# === Main Entry Point ===
if __name__ == "__main__":
    data_path = "Hcj.csv"
    data = pd.read_csv(data_path)
    X_raw, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_raw = X_raw.fillna(X_raw.median())

    topk_features = feature_selection(X_raw, y)
    joblib.dump(list(topk_features), os.path.join(SAVE_DIR, "selected_features.joblib"))
    X = X_raw[topk_features]
    X_scaled, scaler = standardize(X)
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.joblib"))

    print("\n== Active Learning QSVR ==")
    al_results = active_learning_qsvr(
        X_scaled, y,
        save_model_path=os.path.join(SAVE_DIR, "active_learning_qsvr_model.joblib"),
        save_params_path=os.path.join(SAVE_DIR, "active_learning_qsvr_params.json")
    )

    print("\n=== Results ===")
    print(f"Active Learning - Test R2: {al_results['r2']:.4f}, MAE: {al_results['mae']:.4f}, RMSE: {al_results['rmse']:.4f}, Training samples: {al_results['n_train']}, "
          f"Training time: {al_results['final_train_time_sec']:.2f} s, Inference time: {al_results['final_pred_time_sec']:.2f} s")
