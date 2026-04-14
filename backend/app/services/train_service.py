import os
import uuid
import base64
import io
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycaret.classification import setup as cls_setup, create_model as cls_create, pull as cls_pull, save_model as cls_save, interpret_model, predict_model
from pycaret.regression import setup as reg_setup, create_model as reg_create, pull as reg_pull, save_model as reg_save
from pycaret.clustering import setup as clu_setup, create_model as clu_create, pull as clu_pull, save_model as clu_save
from app.config import UPLOAD_DIR, MODEL_DIR

logger = logging.getLogger(__name__)

os.environ["OMP_NUM_THREADS"] = "14"

MODEL_NAME_MAP = {
    'lr': 'lr', 'ridge': 'ridge', 'lasso': 'lasso', 'rf': 'rf',
    'et': 'et', 'gbc': 'gbc', 'gbr': 'gbr', 'lightgbm': 'lightgbm',
    'xgb': 'xgboost', 'dt': 'dt', 'knn': 'knn', 'svm': 'svm',
    'ada': 'ada', 'lda': 'lda', 'qda': 'qda', 'nb': 'nb',
}


def _extract_mean_row(pulled_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    PyCaret's pull() after create_model() returns a DataFrame with one row per CV fold
    plus a 'Mean' row and an 'SD' row at the end. We only keep the Mean row as the
    per-model summary so the metrics table has exactly one row per model.

    Typical index structure:  0, 1, 2, ..., 9, 'Mean', 'SD'
    """
    if 'Mean' in pulled_df.index:
        mean_row = pulled_df.loc[['Mean']].copy()
    elif len(pulled_df) >= 2:
        # Fallback: second-to-last row is typically the Mean
        mean_row = pulled_df.iloc[[-2]].copy()
    else:
        mean_row = pulled_df.iloc[[-1]].copy()

    mean_row['Model'] = model_name.upper()
    return mean_row


def run_automl(
    filename: str,
    task_type: str,
    target_column: str,
    selected_models: list = None,
    ignore_columns: list = None,
    use_smote: bool = False,
    progress_callback=None,
    stop_event=None,
) -> dict:
    filepath = os.path.join(UPLOAD_DIR, filename)
    df = pd.read_csv(filepath)

    model_id = uuid.uuid4().hex[:12]
    feature_importance = {}
    shap_plot = ""
    misclassified_samples = []
    completed_models = []

    ignore_cols = ignore_columns if ignore_columns else []
    fix_imbalance = use_smote if task_type == "classification" else False

    def check_stop():
        if stop_event and stop_event.is_set():
            raise Exception("Training stopped by user")

    def report_progress(progress: int, message: str):
        if progress_callback:
            should_stop = progress_callback(progress, message, completed_models.copy())
            if should_stop:
                raise Exception("Training stopped by user")

    if task_type == "classification":
        report_progress(5, "数据预处理中...")
        check_stop()

        cls_setup(
            data=df, target=target_column, session_id=42, n_jobs=14, verbose=False,
            ignore_features=ignore_cols if ignore_cols else None, fix_imbalance=fix_imbalance,
        )

        report_progress(10, "数据预处理完成")
        check_stop()

        models_to_train = selected_models if selected_models else ['lr', 'rf', 'gbc', 'et', 'xgb']
        mapped_models = [MODEL_NAME_MAP.get(m, m) for m in models_to_train]
        metrics_list = []
        best_model = None
        best_score = -1
        total_models = len(mapped_models)

        for i, model_name in enumerate(mapped_models):
            check_stop()
            report_progress(10 + int(60 * i / total_models), f"训练模型: {model_name.upper()}")
            try:
                model = cls_create(model=model_name, verbose=False)
                # BUG FIX: pull() returns ALL CV folds + Mean + SD rows.
                # Extract only the Mean row so metrics_table has one row per model.
                all_folds = cls_pull()
                mean_row = _extract_mean_row(all_folds, model_name)
                metrics_list.append(mean_row)
                completed_models.append(model_name.upper())

                # BUG FIX: read accuracy from the mean_row (not fold 0 via iloc[0])
                acc = float(mean_row['Accuracy'].iloc[0]) if 'Accuracy' in mean_row.columns else 0
                if acc > best_score:
                    best_score = acc
                    best_model = model

                report_progress(10 + int(60 * (i + 1) / total_models), f"完成: {model_name.upper()}")
            except Exception as e:
                # BUG FIX: was `str(Exception)` (the class object), must be `str(e)` (the instance)
                if "stopped by user" in str(e):
                    raise
                continue

        check_stop()
        if metrics_list:
            metrics_df = pd.concat(metrics_list, ignore_index=True)
        else:
            all_folds = cls_pull()
            metrics_df = _extract_mean_row(all_folds, "unknown")

        report_progress(75, "模型训练完成")
        check_stop()

        if best_model is not None:
            cls_save(best_model, os.path.join(MODEL_DIR, model_id))

            report_progress(80, "生成 SHAP 分析...")
            check_stop()
            try:
                # BUG FIX: interpret_model() does not return a figure object.
                # Capture the matplotlib figure via plt.gcf() after the call.
                plt.close('all')
                interpret_model(best_model, verbose=False)
                fig = plt.gcf()
                if fig and fig.get_axes():
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    shap_plot = base64.b64encode(buf.read()).decode('utf-8')
                    buf.close()
                plt.close('all')
            except Exception:
                plt.close('all')

            check_stop()
            report_progress(90, "分析错误样本...")
            try:
                predictions = predict_model(best_model, data=df)
                if 'prediction_label' in predictions.columns and target_column in predictions.columns:
                    wrong = predictions[predictions['prediction_label'] != predictions[target_column]]
                    misclassified_samples = wrong.head(10).fillna("NaN").to_dict(orient="records")
            except Exception:
                pass

        check_stop()
        report_progress(95, "提取特征重要性...")
        try:
            if best_model and hasattr(best_model, "feature_importances_"):
                import numpy as np
                fis = best_model.feature_importances_
                feature_names = getattr(best_model, "feature_names_in_", None)
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(len(fis))]
                sorted_idx = np.argsort(fis)[::-1][:20]
                for idx in sorted_idx:
                    feature_importance[str(feature_names[idx])] = round(float(fis[idx]), 6)
        except Exception:
            pass

    elif task_type == "regression":
        report_progress(5, "数据预处理中...")
        check_stop()

        reg_setup(
            data=df, target=target_column, session_id=42, n_jobs=14, verbose=False,
            ignore_features=ignore_cols if ignore_cols else None,
        )

        report_progress(10, "数据预处理完成")
        check_stop()

        models_to_train = selected_models if selected_models else ['lr', 'ridge', 'rf', 'gbr', 'et']
        mapped_models = [MODEL_NAME_MAP.get(m, m) for m in models_to_train]
        metrics_list = []
        best_model = None
        best_score = -float('inf')
        total_models = len(mapped_models)

        for i, model_name in enumerate(mapped_models):
            check_stop()
            report_progress(10 + int(60 * i / total_models), f"训练模型: {model_name.upper()}")
            try:
                model = reg_create(model=model_name, verbose=False)
                # BUG FIX: extract Mean row only
                all_folds = reg_pull()
                mean_row = _extract_mean_row(all_folds, model_name)
                metrics_list.append(mean_row)
                completed_models.append(model_name.upper())

                # BUG FIX: use mean value not fold[0]
                r2 = float(mean_row['R2'].iloc[0]) if 'R2' in mean_row.columns else -float('inf')
                if r2 > best_score:
                    best_score = r2
                    best_model = model

                report_progress(10 + int(60 * (i + 1) / total_models), f"完成: {model_name.upper()}")
            except Exception as e:
                # BUG FIX: was `str(Exception)`, must be `str(e)`
                if "stopped by user" in str(e):
                    raise
                continue

        check_stop()
        if metrics_list:
            metrics_df = pd.concat(metrics_list, ignore_index=True)
        else:
            all_folds = reg_pull()
            metrics_df = _extract_mean_row(all_folds, "unknown")

        report_progress(80, "模型训练完成")
        check_stop()

        if best_model is not None:
            reg_save(best_model, os.path.join(MODEL_DIR, model_id))

        check_stop()
        report_progress(95, "提取特征重要性...")
        try:
            if best_model and hasattr(best_model, "feature_importances_"):
                import numpy as np
                fis = best_model.feature_importances_
                feature_names = getattr(best_model, "feature_names_in_", None)
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(len(fis))]
                sorted_idx = np.argsort(fis)[::-1][:20]
                for idx in sorted_idx:
                    feature_importance[str(feature_names[idx])] = round(float(fis[idx]), 6)
        except Exception:
            pass

    elif task_type == "clustering":
        report_progress(5, "数据预处理中...")
        check_stop()

        clu_setup(
            data=df, session_id=42, n_jobs=14, verbose=False,
            ignore_features=ignore_cols if ignore_cols else None,
        )

        report_progress(10, "数据预处理完成")
        check_stop()

        models_to_train = selected_models if selected_models else ['kmeans', 'hclust', 'meanshift']
        metrics_list = []
        best_model = None
        best_score = -1
        total_models = len(models_to_train)

        for i, model_name in enumerate(models_to_train):
            check_stop()
            report_progress(10 + int(70 * i / total_models), f"训练模型: {model_name.upper()}")
            try:
                model = clu_create(model=model_name, num_clusters=4, verbose=False)
                # BUG FIX: extract Mean row from pull()
                all_folds = clu_pull()
                mean_row = _extract_mean_row(all_folds, model_name)
                metrics_list.append(mean_row)
                completed_models.append(model_name.upper())

                if hasattr(model, 'score'):
                    score = model.score(df)
                    if score > best_score:
                        best_score = score
                        best_model = model

                report_progress(10 + int(70 * (i + 1) / total_models), f"完成: {model_name.upper()}")
            except Exception as e:
                # BUG FIX: was `str(Exception)`, must be `str(e)`
                if "stopped by user" in str(e):
                    raise
                continue

        check_stop()
        if metrics_list:
            metrics_df = pd.concat(metrics_list, ignore_index=True)
        else:
            all_folds = clu_pull()
            metrics_df = _extract_mean_row(all_folds, "unknown")

        report_progress(90, "模型训练完成")
        check_stop()

        if best_model is not None:
            clu_save(best_model, os.path.join(MODEL_DIR, model_id))

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    report_progress(98, "生成报告...")

    metrics_table = metrics_df.reset_index(drop=True).to_dict(orient="records")
    metrics_table_serializable = _make_serializable(metrics_table)

    logger.info(f"Training completed. Metrics table: {metrics_table_serializable}")
    logger.info(f"Feature importance: {feature_importance}")
    logger.info(f"Misclassified samples count: {len(misclassified_samples)}")

    return {
        "model_id": model_id,
        "metrics_table": metrics_table_serializable,
        "feature_importance": feature_importance,
        "shap_plot": shap_plot,
        "misclassified_samples": misclassified_samples,
    }


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    else:
        return str(obj)
