import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# データ読み込み
# =========================
train_data = pd.read_csv('llm-class/dataset/train_num.csv')
validation_data = pd.read_csv('llm-class/dataset/validation_num.csv')

X_train = train_data.drop(columns=['label']).astype(np.float32)
y_train = train_data['label']

X_val = validation_data.drop(columns=['label']).astype(np.float32)
y_val = validation_data['label']

# =========================
# 評価関数
# =========================
def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n■{name}")
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)

    return mse, mae, rmse


# =========================================================
# ① RandomForest
# =========================================================
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_val)

evaluate("RandomForest", y_val, pred_rf)

pd.DataFrame({
    'label': y_val,
    'predicted_label': pred_rf,
    **X_val.to_dict('series')
}).to_csv("llm-class/results/regression/results_num_rf.csv", index=False)


# =========================================================
# ② XGBoost
# =========================================================
xgb = XGBRegressor(
    random_state=42,
    objective='reg:squarederror'
)

xgb.fit(X_train, y_train)

pred_xgb = xgb.predict(X_val)

evaluate("XGBoost", y_val, pred_xgb)

pd.DataFrame({
    'label': y_val,
    'predicted_label': pred_xgb,
    **X_val.to_dict('series')
}).to_csv("llm-class/results/regression/results_num_xgb.csv", index=False)


# =========================================================
# ③ LightGBM
# =========================================================
lgb = LGBMRegressor(
    random_state=42,
    verbose=-1
)

lgb.fit(X_train, y_train)

pred_lgb = lgb.predict(X_val)

evaluate("LightGBM", y_val, pred_lgb)

pd.DataFrame({
    'label': y_val,
    'predicted_label': pred_lgb,
    **X_val.to_dict('series')
}).to_csv("llm-class/results/regression/results_num_lgb.csv", index=False)