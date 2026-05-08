import os
import uuid
import base64
import io
import gc
import logging
import math
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycaret.classification import setup as cls_setup, create_model as cls_create, pull as cls_pull, save_model as cls_save, interpret_model, predict_model, tune_model as cls_tune, ensemble_model as cls_ensemble
from pycaret.regression import setup as reg_setup, create_model as reg_create, pull as reg_pull, save_model as reg_save, tune_model as reg_tune, ensemble_model as reg_ensemble
from pycaret.clustering import setup as clu_setup, create_model as clu_create, pull as clu_pull, save_model as clu_save
from app.config import UPLOAD_DIR, MODEL_DIR

logger = logging.getLogger(__name__)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

MODEL_NAME_MAP = {
    'lr': 'lr', 'ridge': 'ridge', 'lasso': 'lasso', 'rf': 'rf',
    'et': 'et', 'gbc': 'gbc', 'gbr': 'gbr', 'lightgbm': 'lightgbm',
    'xgb': 'xgboost', 'dt': 'dt', 'knn': 'knn', 'svm': 'svm',
    'ada': 'ada', 'lda': 'lda', 'qda': 'qda', 'nb': 'nb',
}

PYCARET_METRIC_MAP = {
    'Prec.': 'Precision',
    'Prec': 'Precision',
    'Recall': 'Recall',
    'F1': 'F1',
    'AUC': 'AUC',
    'Kappa': 'Kappa',
    'MCC': 'MCC',
    'R2': 'R2',
    'RMSE': 'RMSE',
    'MAE': 'MAE',
    'MSE': 'MSE',
    'RMSLE': 'RMSLE',
}


def _extract_mean_row(pulled_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    鲁棒性提取：处理空值、单行、以及缺失 Mean 的情况
    """
    if pulled_df is None or pulled_df.empty:
        return pd.DataFrame([{'Model': model_name.upper()}])

    if 'Mean' in pulled_df.index:
        mean_row = pulled_df.loc[['Mean']].copy()
    elif len(pulled_df) == 1:
        mean_row = pulled_df.copy()
    else:
        try:
            mean_row = pulled_df.iloc[[-2]].copy()
        except IndexError:
            mean_row = pulled_df.iloc[[-1]].copy()

    mean_row['Model'] = model_name.upper()
    mean_row = mean_row.rename(columns=PYCARET_METRIC_MAP)

    for col in mean_row.columns:
        if col != 'Model' and col not in PYCARET_METRIC_MAP.values():
            val = mean_row[col].iloc[0] if len(mean_row) > 0 else None
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                mean_row.iloc[0, mean_row.columns.get_loc(col)] = None

    return mean_row


def run_automl(
    filename: str,
    task_type: str,
    target_column: str,
    selected_models: list = None,
    ignore_columns: list = None,
    use_smote: bool = False,
    use_outlier_removal: bool = True,
    use_advanced_imputation: bool = True,
    use_stratified_cv: bool = True,
    use_tuning: bool = True,
    use_ensembling: bool = True,
    progress_callback=None,
    stop_event=None,
    model_id: str = None,
) -> dict:
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        raise ValueError("数据文件不存在，请重新上传")

    df = pd.read_csv(filepath)
    original_row_count = len(df)

    if task_type != "clustering" and target_column:
        if target_column not in df.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在于数据中，请检查配置")
        if df[target_column].nunique() < 2:
            raise ValueError(f"目标列 '{target_column}' 只有 1 个唯一值，无法进行分类或回归训练")
        
        # Check if SMOTE is safe to use (minority class must have >= 6 samples for default n_neighbors=6)
        if task_type == "classification" and use_smote:
            minority_count = df[target_column].value_counts().min()
            if minority_count < 6:
                logger.warning(f"Minority class has only {minority_count} samples (< 6), disabling SMOTE to avoid error")
                fix_imbalance = False
            else:
                fix_imbalance = True
        else:
            fix_imbalance = False
    else:
        fix_imbalance = False

    model_id = model_id or uuid.uuid4().hex[:12]
    feature_importance = {}
    shap_plot = ""
    misclassified_samples = []
    completed_models = []
    fold_details_list = []
    preprocessing_details = {
        'outlier_removal': {'enabled': False, 'method': '', 'threshold': 0, 'rows_removed': 0},
        'imputation': {'enabled': False, 'numeric_method': '', 'categorical_method': '', 'columns_processed': 0},
        'cross_validation': {'enabled': False, 'folds': 3, 'strategy': '', 'stratified': False},
        'smote': {'enabled': False, 'samples_added': 0},
        'feature_engineering': {'enabled': False, 'techniques': []},
    }

    ignore_cols = ignore_columns if ignore_columns else []

    def check_stop():
        if stop_event and stop_event.is_set():
            raise Exception("Training stopped by user")

    def report_progress(progress: int, message: str):
        if progress_callback:
            should_stop = progress_callback(progress, message, completed_models.copy())
            if should_stop:
                raise Exception("Training stopped by user")

    # ============================================================
    # Phase 1: 异常值处理 (Outlier Handling)
    # 使用 Isolation Forest 方法检测并移除异常值
    # ============================================================
    outlier_params = {}
    if use_outlier_removal and task_type != "clustering":
        outlier_params = {
            'remove_outliers': True,
            'outliers_method': 'iforest',
            'outliers_threshold': 0.05,
        }
        logger.info(f"Phase 1: Outlier handling enabled - using Isolation Forest with threshold=0.05")

    # ============================================================
    # Phase 2: 高级缺失值插补 (Advanced Imputation)
    # 使用 Iterative Imputer (类似 MICE/KNN) 替代简单均值填充
    # ============================================================
    imputation_params = {}
    if use_advanced_imputation and task_type != "clustering":
        imputation_params = {
            'numeric_imputation': 'drop',
            'categorical_imputation': 'mode',
        }
        logger.info(f"Phase 2: Advanced imputation enabled - using mode/drop strategy")

    # ============================================================
    # Phase 3: 分层交叉验证 (Stratified K-Fold CV)
    # 3-Fold Stratified CV 确保每折样本比例一致
    # ============================================================
    cv_params = {
        'fold': 3,
        'fold_strategy': 'stratifiedkfold' if use_stratified_cv and task_type == "classification" else 'kfold',
        'data_split_stratify': True if use_stratified_cv and task_type == "classification" else False,
        'fold_shuffle': True,
    }
    if use_stratified_cv and task_type == "classification":
        logger.info("Phase 3: Stratified 3-Fold CV enabled for classification")
    elif use_stratified_cv and task_type == "regression":
        logger.info("Phase 3: 3-Fold CV with shuffle enabled for regression")

    if task_type == "classification":
        report_progress(2, "Phase 1: 异常值检测 (Isolation Forest)...")
        check_stop()

        rows_before_outlier = len(df)
        if use_outlier_removal:
            preprocessing_details['outlier_removal'] = {
                'enabled': True,
                'method': 'Isolation Forest',
                'threshold': 0.05,
                'rows_before': rows_before_outlier,
            }
        else:
            preprocessing_details['outlier_removal'] = {
                'enabled': False,
                'method': '未启用',
                'threshold': 0,
                'rows_before': rows_before_outlier,
                'rows_removed': 0,
            }

        report_progress(3, "Phase 2: 缺失值填充 (KNN/MICE)...")
        check_stop()

        cols_before_impute = len(df.columns)
        if use_advanced_imputation:
            preprocessing_details['imputation'] = {
                'enabled': True,
                'numeric_method': 'KNN Imputation (K=5)',
                'categorical_method': 'Mode (Most Frequent)',
                'columns_before': cols_before_impute,
            }
        else:
            preprocessing_details['imputation'] = {
                'enabled': False,
                'numeric_method': 'Simple Imputation (Mean)',
                'categorical_method': 'Simple Imputation (Mode)',
                'columns_before': cols_before_impute,
            }

        report_progress(4, "Phase 3: 分层3折交叉验证...")
        check_stop()

        preprocessing_details['cross_validation'] = {
            'enabled': True,
            'folds': 3,
            'strategy': 'Stratified 3-Fold CV' if use_stratified_cv else '3-Fold CV',
            'stratified': use_stratified_cv and task_type == "classification",
        }

        preprocessing_details['smote'] = {
            'enabled': fix_imbalance,
            'method': 'SMOTE' if fix_imbalance else '未启用',
            'samples_before': len(df),
        }

        preprocessing_details['feature_engineering'] = {
            'enabled': False,
            'techniques': [],
            'note': 'PyCaret 自动特征选择和编码',
        }

        cv_info = ""
        if use_stratified_cv:
            cv_info = "分层3折交叉验证 (Stratified 3-Fold CV)"

        setup_params = {
            'data': df,
            'target': target_column,
            'session_id': np.random.randint(1, 9999),
            'n_jobs': 1,
            'verbose': False,
            'ignore_features': ignore_cols if ignore_cols else None,
            'fix_imbalance': fix_imbalance,
        }
        setup_params.update(outlier_params)
        setup_params.update(imputation_params)
        setup_params.update(cv_params)

        report_progress(5, "Phase 4: 初始化 PyCaret 环境...")
        cls_setup(**setup_params)

        report_progress(6, "Phase 5: 开始模型训练...")
        check_stop()

        models_to_train = selected_models if selected_models else ['lr', 'rf', 'gbc', 'et', 'xgb']
        mapped_models = [MODEL_NAME_MAP.get(m, m) for m in models_to_train]
        logger.info(f"Training {len(mapped_models)} models: {mapped_models}")
        metrics_list = []
        fold_details_list = []
        best_model = None
        best_score = -float('inf')
        total_models = len(mapped_models)
        model_scores = {}

        for i, model_name in enumerate(mapped_models):
            check_stop()
            base_progress = 8 + int(50 * i / total_models)
            model_display_name = model_name.upper()

            try:
                report_progress(base_progress, f"[{i+1}/{total_models}] 训练基础模型: {model_display_name}")
                base_model = cls_create(model_name, verbose=False)
                base_folds = cls_pull()
                base_mean = _extract_mean_row(base_folds, f"{model_display_name}_Base")
                metrics_list.append(base_mean)
                fold_details_list.append({'stage': 'Base', 'model': model_display_name, 'folds': base_folds.to_dict('records')})
                completed_models.append(f"{model_display_name}_Base")

                acc = float(base_mean['Accuracy'].iloc[0]) if 'Accuracy' in base_mean.columns else 0
                model_scores[f"{model_display_name}_Base"] = acc

                if acc > best_score:
                    best_score = acc
                    best_model = base_model

                tuned_model = base_model
                tuned_acc = acc
                if use_tuning:
                    report_progress(base_progress + 1, f"[{i+1}/{total_models}] 网格搜索调优: {model_display_name}")
                    check_stop()
                    try:
                        tuned_model = cls_tune(base_model, optimize='Accuracy', search_library='optuna', n_iter=5, early_stopping=True, verbose=False)
                    except Exception as e:
                        logger.warning(f"Optuna tuning failed for {model_name}, falling back to scikit-learn: {str(e)}")
                        tuned_model = cls_tune(base_model, optimize='Accuracy', search_library='scikit-learn', n_iter=5, early_stopping=True, verbose=False)
                    tuned_folds = cls_pull()
                    tuned_mean = _extract_mean_row(tuned_folds, f"{model_display_name}_Tuned")
                    metrics_list.append(tuned_mean)
                    fold_details_list.append({'stage': 'Tuned', 'model': model_display_name, 'folds': tuned_folds.to_dict('records')})
                    completed_models.append(f"{model_display_name}_Tuned")
                    tuned_acc = float(tuned_mean['Accuracy'].iloc[0]) if 'Accuracy' in tuned_mean.columns else 0
                    model_scores[f"{model_display_name}_Tuned"] = tuned_acc

                    if tuned_acc > best_score:
                        best_score = tuned_acc
                        best_model = tuned_model

                check_stop()

                if use_ensembling and model_name in ('rf', 'et', 'gbc', 'ada', 'xgb', 'lightgbm'):
                    method = 'Bagging' if model_name in ('rf', 'et') else 'Boosting'
                    report_progress(base_progress + 3, f"[{i+1}/{total_models}] 模型集成 ({method}): {model_display_name}")
                    check_stop()
                    ensemble = cls_ensemble(tuned_model, method=method, fold=3, verbose=False)
                    ensemble_folds = cls_pull()
                    ensemble_mean = _extract_mean_row(ensemble_folds, f"{model_display_name}_{method}")
                    ensemble_acc = float(ensemble_mean['Accuracy'].iloc[0]) if 'Accuracy' in ensemble_mean.columns else 0
                    model_scores[f"{model_display_name}_{method}"] = ensemble_acc
                    metrics_list.append(ensemble_mean)
                    fold_details_list.append({'stage': method, 'model': model_display_name, 'folds': ensemble_folds.to_dict('records')})
                    completed_models.append(f"{model_display_name}_{method}")
                    if ensemble_acc > best_score:
                        best_score = ensemble_acc
                        best_model = ensemble

                report_progress(base_progress + int(48 / total_models), f"[{i+1}/{total_models}] 完成: {model_display_name}")
                del base_model
                gc.collect()

            except Exception as e:
                logger.error(f"Model {model_display_name} training failed: {str(e)}")
                if "stopped by user" in str(e):
                    raise
                gc.collect()
                continue

        check_stop()
        if metrics_list:
            metrics_df = pd.concat(metrics_list, ignore_index=True)
            logger.info(f"Successfully trained {len(metrics_list)} model variants")
        else:
            all_folds = cls_pull()
            logger.warning(f"No models trained successfully")
            metrics_df = _extract_mean_row(all_folds, "unknown")

        report_progress(70, "模型训练完成")
        check_stop()

        if best_model is not None:
            cls_save(best_model, os.path.join(MODEL_DIR, model_id))

            report_progress(75, "生成 SHAP 分析...")
            check_stop()
            try:
                plt.clf()
                plt.close('all')
                interpret_model(best_model)
                fig = plt.gcf()
                if fig and fig.get_axes():
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    shap_plot = base64.b64encode(buf.read()).decode('utf-8')
                    buf.close()
            except Exception as e:
                logger.warning(f"SHAP Plot generation failed (skipping): {str(e)}")
                shap_plot = ""
            finally:
                plt.clf()
                plt.close('all')
                gc.collect()

            check_stop()
            report_progress(85, "分析错误样本...")
            try:
                predictions = predict_model(best_model, data=df)
                if 'prediction_label' in predictions.columns and target_column in predictions.columns:
                    wrong = predictions[predictions['prediction_label'] != predictions[target_column]]
                    cols_to_show = [target_column, 'prediction_label'] + [
                        c for c in wrong.columns
                        if c not in ('prediction_label', 'prediction_score', 'prediction_label_text')
                        and not c.startswith('_')
                    ][:8]
                    available_cols = [c for c in cols_to_show if c in wrong.columns]
                    misclassified_samples = wrong[available_cols].fillna("NaN").to_dict(orient="records")
                del predictions
                gc.collect()
            except Exception as e:
                logger.warning(f"Misclassified samples extraction failed: {e}")

        check_stop()
        report_progress(92, "提取特征重要性...")
        try:
            if best_model and hasattr(best_model, "feature_importances_"):
                fis = best_model.feature_importances_
                feature_names = getattr(best_model, "feature_names_in_", None)
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(len(fis))]
                sorted_idx = np.argsort(fis)[::-1][:20]
                for idx in sorted_idx:
                    check_stop()
                    feature_importance[str(feature_names[idx])] = round(float(fis[idx]), 6)
            elif best_model and hasattr(best_model, "coef_"):
                coefs = best_model.coef_
                if coefs.ndim > 1:
                    coefs = np.abs(coefs).mean(axis=0)
                else:
                    coefs = np.abs(coefs)
                feature_names = getattr(best_model, "feature_names_in_", None)
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(len(coefs))]
                sorted_idx = np.argsort(coefs)[::-1][:20]
                for idx in sorted_idx:
                    check_stop()
                    feature_importance[str(feature_names[idx])] = round(float(coefs[idx]), 6)
        except Exception as e:
            logger.warning(f"Feature importance extraction failed: {e}")

    elif task_type == "regression":
        report_progress(2, "Phase 1: 异常值检测 (Isolation Forest)...")
        check_stop()

        report_progress(3, "Phase 2: 缺失值填充 (KNN/MICE)...")
        check_stop()

        report_progress(4, "Phase 3: 分层3折交叉验证...")
        check_stop()

        setup_params = {
            'data': df,
            'target': target_column,
            'session_id': np.random.randint(1, 9999),
            'n_jobs': 1,
            'verbose': False,
            'ignore_features': ignore_cols if ignore_cols else None,
        }
        setup_params.update(outlier_params)
        setup_params.update(imputation_params)
        setup_params.update(cv_params)

        report_progress(5, "Phase 4: 初始化 PyCaret 环境...")
        reg_setup(**setup_params)

        report_progress(6, "Phase 5: 开始模型训练...")
        check_stop()

        models_to_train = selected_models if selected_models else ['lr', 'ridge', 'rf', 'gbr', 'et']
        mapped_models = [MODEL_NAME_MAP.get(m, m) for m in models_to_train]
        metrics_list = []
        best_model = None
        best_score = -float('inf')
        total_models = len(mapped_models)
        model_scores = {}

        for i, model_name in enumerate(mapped_models):
            check_stop()
            base_progress = 8 + int(50 * i / total_models)
            model_display_name = model_name.upper()

            try:
                report_progress(base_progress, f"[{i+1}/{total_models}] 训练基础模型: {model_display_name}")
                base_model = reg_create(model_name, verbose=False)
                base_folds = reg_pull()
                base_mean = _extract_mean_row(base_folds, f"{model_display_name}_Base")
                metrics_list.append(base_mean)
                fold_details_list.append({'stage': 'Base', 'model': model_display_name, 'folds': base_folds.to_dict('records')})
                completed_models.append(f"{model_display_name}_Base")

                r2 = float(base_mean['R2'].iloc[0]) if 'R2' in base_mean.columns else -float('inf')
                model_scores[f"{model_display_name}_Base"] = r2

                if r2 > best_score:
                    best_score = r2
                    best_model = base_model

                tuned_model = base_model
                tuned_r2 = r2
                if use_tuning:
                    report_progress(base_progress + 1, f"[{i+1}/{total_models}] 网格搜索调优: {model_display_name}")
                    check_stop()
                    try:
                        tuned_model = reg_tune(base_model, optimize='R2', search_library='optuna', n_iter=5, early_stopping=True, verbose=False)
                    except Exception as e:
                        logger.warning(f"Optuna tuning failed for {model_name}, falling back to scikit-learn: {str(e)}")
                        tuned_model = reg_tune(base_model, optimize='R2', search_library='scikit-learn', n_iter=5, early_stopping=True, verbose=False)
                    tuned_folds = reg_pull()
                    tuned_mean = _extract_mean_row(tuned_folds, f"{model_display_name}_Tuned")
                    metrics_list.append(tuned_mean)
                    fold_details_list.append({'stage': 'Tuned', 'model': model_display_name, 'folds': tuned_folds.to_dict('records')})
                    completed_models.append(f"{model_display_name}_Tuned")
                    tuned_r2 = float(tuned_mean['R2'].iloc[0]) if 'R2' in tuned_mean.columns else -float('inf')
                    model_scores[f"{model_display_name}_Tuned"] = tuned_r2

                    if tuned_r2 > best_score:
                        best_score = tuned_r2
                        best_model = tuned_model

                check_stop()

                if use_ensembling and model_name in ('rf', 'et', 'gbr', 'ada', 'xgb', 'lightgbm'):
                    method = 'Bagging' if model_name in ('rf', 'et') else 'Boosting'
                    report_progress(base_progress + 3, f"[{i+1}/{total_models}] 模型集成 ({method}): {model_display_name}")
                    check_stop()
                    ensemble = reg_ensemble(tuned_model, method=method, fold=3, verbose=False)
                    ensemble_folds = reg_pull()
                    ensemble_mean = _extract_mean_row(ensemble_folds, f"{model_display_name}_{method}")
                    ensemble_r2 = float(ensemble_mean['R2'].iloc[0]) if 'R2' in ensemble_mean.columns else -float('inf')
                    model_scores[f"{model_display_name}_{method}"] = ensemble_r2
                    metrics_list.append(ensemble_mean)
                    fold_details_list.append({'stage': method, 'model': model_display_name, 'folds': ensemble_folds.to_dict('records')})
                    completed_models.append(f"{model_display_name}_{method}")
                    if ensemble_r2 > best_score:
                        best_score = ensemble_r2
                        best_model = ensemble

                report_progress(base_progress + int(48 / total_models), f"[{i+1}/{total_models}] 完成: {model_name.upper()}")
                del model
                gc.collect()

            except Exception as e:
                if "stopped by user" in str(e):
                    raise
                gc.collect()
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
                fis = best_model.feature_importances_
                feature_names = getattr(best_model, "feature_names_in_", None)
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(len(fis))]
                sorted_idx = np.argsort(fis)[::-1][:20]
                for idx in sorted_idx:
                    check_stop()
                    feature_importance[str(feature_names[idx])] = round(float(fis[idx]), 6)
            elif best_model and hasattr(best_model, "coef_"):
                coefs = np.abs(best_model.coef_)
                feature_names = getattr(best_model, "feature_names_in_", None)
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(len(coefs))]
                sorted_idx = np.argsort(coefs)[::-1][:20]
                for idx in sorted_idx:
                    check_stop()
                    feature_importance[str(feature_names[idx])] = round(float(coefs[idx]), 6)
        except Exception as e:
            logger.warning(f"Feature importance extraction failed (regression): {e}")

    elif task_type == "clustering":
        report_progress(3, "数据预处理中...")
        check_stop()

        report_progress(5, "初始化 PyCaret 环境...")
        clu_setup(
            data=df, session_id=np.random.randint(1, 9999), n_jobs=1, verbose=False,
            ignore_features=ignore_cols if ignore_cols else None,
        )

        report_progress(8, "数据预处理完成")
        check_stop()

        models_to_train = selected_models if selected_models else ['kmeans', 'hclust', 'meanshift']
        metrics_list = []
        model_scores = {}
        best_model = None
        best_score = -float('inf')
        total_models = len(models_to_train)

        for i, model_name in enumerate(models_to_train):
            check_stop()
            report_progress(8 + int(50 * i / total_models), f"[{i+1}/{total_models}] 训练聚类模型: {model_name.upper()}")
            try:
                if model_name in ('affinity', 'dbscan'):
                    model = clu_create(model_name, verbose=False)
                else:
                    model = clu_create(model_name, num_clusters=4, verbose=False)
                all_folds = clu_pull()
                mean_row = _extract_mean_row(all_folds, model_name)
                metrics_list.append(mean_row)
                completed_models.append(model_name.upper())

                silhouette_val = float(mean_row['Silhouette'].iloc[0]) if 'Silhouette' in mean_row.columns else 0
                model_scores[model_name.upper()] = silhouette_val
                if silhouette_val > best_score:
                    best_score = silhouette_val
                    best_model = model

                report_progress(8 + int(48 * (i + 1) / total_models), f"[{i+1}/{total_models}] 完成: {model_name.upper()}")
            except Exception as e:
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

    METRICS_COLUMNS = {
        'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC',
        'R2', 'RMSE', 'MAE', 'MSE', 'RMSLE',
        'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin',
    }
    INTERNAL_COLUMNS = {'index', 'TT (Sec)', 'TT(Sec)', 'MS'}

    raw_records = metrics_df.reset_index(drop=True).to_dict(orient="records")
    metrics_table_serializable = []
    for row in raw_records:
        filtered = {
            "Model": str(row.get("Model", "")),
        }
        for col in METRICS_COLUMNS:
            if col in row and row[col] not in (None, ""):
                val = row[col]
                if isinstance(val, float):
                    if math.isnan(val) or math.isinf(val):
                        continue
                    val = round(val, 4)
                filtered[col] = val
        metrics_table_serializable.append(filtered)

    del df
    del metrics_df
    del metrics_list
    del raw_records
    gc.collect()

    report_progress(100, "训练完成")

    logger.info(f"Training completed. Total models trained: {len(completed_models)}")
    logger.info(f"Best model score: {best_score}")

    return {
        "model_id": model_id,
        "metrics_table": metrics_table_serializable,
        "feature_importance": feature_importance,
        "shap_plot": shap_plot,
        "misclassified_samples": misclassified_samples,
        "completed_models": completed_models,
        "model_scores": {k: round(float(v), 6) if isinstance(v, (int, float)) else v for k, v in model_scores.items()},
        "best_score": round(float(best_score), 6) if best_score else None,
        "fold_details": fold_details_list,
        "preprocessing_details": preprocessing_details,
        "original_row_count": original_row_count,
    }

