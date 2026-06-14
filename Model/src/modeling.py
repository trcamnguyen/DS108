import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from src.evaluation import evaluate_model

def tune_ridge_alpha(
    alphas,
    X,
    y,
    cv,
    stratify_labels,
    preprocessor
):
    """
    Tuning alpha cho Ridge Regression bằng Cross Validation.

    Parameters
    ----------
    alphas : list
        Danh sách alpha cần thử.

    X : pd.DataFrame
        Feature matrix.

    y : pd.Series
        Target.

    cv : CV object
        StratifiedKFold.

    stratify_labels : pd.Series
        Labels dùng cho stratification.

    preprocessor : ColumnTransformer
        Pipeline preprocessing.

    Returns
    -------
    pd.DataFrame
        Bảng kết quả alpha tuning.
    """
    alpha_results = []
    for alpha in alphas:
        ridge_model = Pipeline(
            steps=[
                ("preprocessor",preprocessor),
                ("model",Ridge(alpha=alpha))
            ]
        )
        cv_result = evaluate_model(
            model=ridge_model,
            X=X,
            y=y,
            cv=cv,
            stratify_labels=stratify_labels
        )
        alpha_results.append(
            {
                "alpha": alpha,
                "mae_mean": cv_result["mae"].mean(),
                "mae_std": cv_result["mae"].std(),
                "rmse_mean": cv_result["rmse"].mean(),
                "rmse_std": cv_result["rmse"].std()
            }
        )
    return pd.DataFrame(alpha_results)

def build_ridge_pipeline(
    preprocessor,
    alpha
):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                Ridge(alpha=alpha)
            )
        ]
    )

def build_lgbm_pipeline(
    preprocessor,
    random_state=42,
    n_estimators=300
):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LGBMRegressor(
                    random_state=random_state,
                    n_estimators=n_estimators
                )
            )
        ]
    )