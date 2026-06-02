import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, X, y, cv, stratify_labels):
    """
    Thực hiện Cross-Validation và trả về DataFrame kết quả MAE, RMSE từng Fold.
    Đảm bảo tính tái lập và đánh giá khách quan.
    """
    results = []
    
    # Duyệt qua từng Fold của StratifiedKFold
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, stratify_labels)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # Huấn luyện
        model.fit(X_train, y_train)
        
        # Dự đoán
        preds = model.predict(X_valid)
        
        # Tính metrics
        mae = mean_absolute_error(y_valid, preds)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        
        results.append({
            "fold": fold + 1,
            "mae": mae,
            "rmse": rmse
        })
        
    return pd.DataFrame(results)

def get_model_summary(results_df):
    """
    Trả về giá trị Mean và Std của Metrics sau Cross-validation.
    """
    return results_df[["mae", "rmse"]].agg(["mean", "std"])