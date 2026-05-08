# VibeMine

**AutoML Intelligence Platform — From Data to Best Model, Automatically**

---

## Overview

VibeMine is an end-to-end intelligent automated machine learning platform built on PyCaret 3.x, providing a fully automated pipeline from data upload, feature engineering, model training, performance evaluation to model export.

Users can upload a CSV data file and the platform automatically completes data preprocessing, algorithm selection, hyperparameter tuning, model ensembling, and outputs a comprehensive evaluation report for the best model — zero code required.

### Key Features

- **Zero-Code AutoML**: Upload data → Configure → One-click train → Get best model
- **AI-Powered Insights**: Google Gemini 2.0 Flash driven model evaluation & misclassification analysis
- **Bilingual UI**: Complete i18n support (Chinese / English)
- **Multi-dimensional Visualization**: Bar charts, radar charts, SHAP plots, feature importance charts
- **6 Advanced Options**: Outlier removal, imputation, stratified CV, tuning, ensembling, SMOTE — all independently controllable
- **One-Click Export**: Standard PyCaret .pkl model + Python deployment code + Markdown report

---

## How VibeMine Ensures the Best Model

Traditional ML requires manual data cleaning, feature engineering, algorithm tuning — time-consuming and experience-dependent. VibeMine ensures optimal model performance through six systematic dimensions:

1. **Data Quality**: Outlier detection + robust missing value imputation
2. **Evaluation Reliability**: Stratified cross-validation prevents evaluation bias
3. **Algorithm Diversity**: Up to 27 algorithms compete in parallel, covering all mainstream models
4. **Hyperparameter Intelligence**: Bayesian optimization (Optuna) with scikit-learn fallback automatically searches for optimal parameters
5. **Model Ensembling**: Bagging + Boosting dual strategy for stability improvement
6. **Multi-dimensional Evaluation**: 5+ metrics + visualization + AI interpretation for trustworthy results

---

## End-to-End Pipeline

```
Upload → Analysis → Configure → Train → Evaluate → Export
   ↓         ↓          ↓         ↓         ↓         ↓
 CSV    Type detect  Adv. opts  Multi-model  Charts   .pkl model
        Imbalance    SMOTE     Bayes tuning  SHAP     Deploy code
        AI insight   Ensemble  Ensembling    AI read  Markdown report
```

---

## Three Task Types

### Classification

Supports binary and multi-class with 12 algorithms:

| Algorithm | ID | Type |
|-----------|-----|------|
| Logistic Regression | lr | Linear |
| Random Forest | rf | Ensemble - Bagging |
| Gradient Boosting Classifier | gbc | Ensemble - Boosting |
| Extra Trees Classifier | et | Ensemble - Bagging |
| XGBoost | xgb | Ensemble - Boosting |
| LightGBM | lightgbm | Ensemble - Boosting |
| Decision Tree | dt | Tree-based |
| K-Nearest Neighbors | knn | Distance-based |
| AdaBoost | ada | Ensemble - Boosting |
| Linear Discriminant Analysis | lda | Linear |
| Naive Bayes | nb | Probabilistic |
| Quadratic Discriminant Analysis | qda | Probabilistic |

**Metrics**: Accuracy, AUC, Recall, Precision, F1, Kappa, MCC

### Regression

Supports 10 regression algorithms:

| Algorithm | ID | Type |
|-----------|-----|------|
| Linear Regression | lr | Linear |
| Ridge Regression | ridge | Linear - Regularized |
| Lasso Regression | lasso | Linear - Regularized |
| Elastic Net | en | Linear - Regularized |
| Random Forest Regressor | rf | Ensemble - Bagging |
| Extra Trees Regressor | et | Ensemble - Bagging |
| Gradient Boosting Regressor | gbr | Ensemble - Boosting |
| LightGBM Regressor | lightgbm | Ensemble - Boosting |
| K-Nearest Neighbors Regressor | knn | Distance-based |
| Decision Tree Regressor | dt | Tree-based |

**Metrics**: R2, RMSE, MAE, MSE, RMSLE

### Clustering

Supports 5 clustering algorithms:

| Algorithm | ID |
|-----------|-----|
| K-Means | kmeans |
| Hierarchical Clustering | hclust |
| DBSCAN | dbscan |
| Mean Shift | meanshift |
| Affinity Propagation | affinity |

**Metrics**: Silhouette, Calinski-Harabasz, Davies-Bouldin

---

## Data Preprocessing

### Outlier Detection & Removal

Uses Isolation Forest algorithm to automatically detect outliers in data:

```python
# Configured via PyCaret setup params
remove_outliers=True, outliers_method='iforest', outliers_threshold=0.05
```

### Missing Value Imputation

Robust two-stage imputation strategy:

1. **Numeric columns**: Drop rows with missing values (`numeric_imputation='drop'`) to avoid imputation error propagation
2. **Categorical columns**: Fill with mode (most frequent value) via `categorical_imputation='mode'`

### Feature Type Inference

Automatically identifies and distinguishes:

- **Numeric features** (int64, float64) → for KNN, linear models, tree models
- **Categorical features** (object, category) → auto-encoded for modeling
- **Target column** → excluded from features

---

## Model Training

### Cross-Validation Strategy

- **Classification**: 3-Fold Stratified CV, ensuring class proportions match the overall distribution in each fold
- **Regression**: 3-Fold CV with shuffle
- Stratified CV prevents evaluation bias from class imbalance in individual folds

### Hyperparameter Tuning

Uses Optuna framework for Bayesian optimization with automatic fallback:

- **Iterations**: 5 (balances effectiveness and speed)
- **Optimization target**: Accuracy (classification) / R2 (regression)
- **Fallback**: If Optuna fails, automatically falls back to scikit-learn search
- **Early stopping**: Enabled to prevent overfitting

### Model Ensembling

Automatically applies Bagging or Boosting ensemble to well-performing base models:

| Base Model | Ensemble Method | Strategy |
|-----------|---------------|----------|
| Random Forest, Extra Trees | Bagging | Reduce variance |
| GBC, AdaBoost, XGBoost, LightGBM | Boosting | Reduce bias |

### SMOTE for Class Imbalance

- Auto-detects when any class ratio exceeds 2:1
- User can manually enable SMOTE for minority class oversampling
- Safety check: SMOTE auto-disabled if minority class has fewer than 6 samples
- **Classification only**, default off

---

## Multi-Dimensional Evaluation

### Metrics

| Task Type | Metrics |
|-----------|--------|
| Classification | Accuracy, AUC, Recall, Precision, F1, Kappa, MCC |
| Regression | R2, RMSE, MAE, MSE, RMSLE |
| Clustering | Silhouette, Calinski-Harabasz, Davies-Bouldin |

### Visualization

- **Bar Chart**: Multi-metric horizontal comparison across models (multi-select + dual Y-axis for mixed direction metrics)
- **Radar Chart**: Single model multi-dimensional capability display
- **Feature Importance**: Top 15-20 important features (gradient color indicating importance level)
- **SHAP Summary Plot**: SHAP feature impact analysis (for LightGBM/XGBoost)
- **Misclassified Samples**: Classification task prediction error display (true vs predicted comparison)

### AI-Powered Interpretation

Powered by Google Gemini 2.0 Flash API, called directly from frontend (API Key stored locally in browser only):

- **Data Insights**: Analyzes data distribution, feature correlations, potential issues
- **Model Evaluation**: Explains best model performance, provides improvement suggestions
- **Misclassification Analysis**: Analyzes patterns in incorrectly classified samples
- **Independent Loading States**: Evaluation and misclassification analysis have separate loading indicators
- **Safe Rendering**: ReactMarkdown used instead of dangerouslySetInnerHTML

---

## Report Generation

After training, a complete Markdown format data report can be generated, including:

| Report Section | Content |
|---------------|---------|
| Data Overview | Filename, row/column count, column type statistics |
| Configuration | Task type, target column, ignored columns, selected models, advanced option states |
| Preprocessing Results | Outlier handling, missing value imputation, feature encoding |
| Training Results | Per-model metrics comparison table, best model and score |
| Feature Importance | Top 20 important features and weights |
| AI Evaluation | Gemini-generated model analysis and improvement suggestions |
| Misclassified Samples | Statistics on incorrectly classified samples (classification tasks) |

---

## Frontend

### Tech Stack

- React 19 + TypeScript
- Zustand state management
- ECharts 6 charts
- Tailwind CSS 4 (Apple glassmorphism style)
- Custom I18n Context (Chinese / English)
- ReactMarkdown safe rendering
- Axios API calls

### Page Flow

```
Step 1: Upload → Step 2: Config → Step 3: Train → Step 4: Evaluate → Step 5: Export
```

### Main Components

| Component | Function |
|-----------|----------|
| StepUpload | CSV upload, data preview (Data/Info/Stats tabs), AI insight |
| StepConfig | Task type, target column, model selection, ignore columns, 6 advanced option toggles |
| StepTrain | Training progress ring, model logs, real-time status polling |
| StepEvaluate | Metrics bar chart, radar chart, feature importance, SHAP, AI interpretation (independent loading) |
| StepExport | Model download, view/download Markdown report, Python deploy code, restart |
| SettingsModal | Gemini API Key configuration (local storage) |
| Header | Brand, language switch, GitHub link, settings |

---

## Backend Architecture

### Tech Stack

- FastAPI (async API + GZip compression + request timing middleware)
- PyCaret 3.x (AutoML engine)
- Pandas + NumPy (data processing)
- Threading (concurrent training with lock-based session management)
- Logging (structured logging + millisecond request duration)

### Core Services

| Service | File | Responsibility |
|---------|------|----------------|
| train_service | train_service.py | Training orchestration, PyCaret wrapping, SHAP generation |
| data_service | data_service.py | CSV parsing, data statistics, data profiling, class distribution detection |
| train.py | routes/train.py | FastAPI routes, session management, training control |
| upload.py | routes/upload.py | File upload, security validation, size/row/column limits |
| download.py | routes/download.py | Model download, path traversal prevention |
| ai_service.py | services/ai_service.py | AI service (reserved, AI calls moved to frontend) |

### Concurrency Control

- `threading.Lock` protects session dictionary; file I/O executed outside lock to avoid blocking
- Session timeout auto-cleanup (3600 seconds)
- Maximum 50 concurrent training sessions
- Random 12-char session_id prevents collision
- User-initiated stop via `threading.Event`

### Process Safety

- `multiprocessing.set_start_method("spawn")` prevents Broken Pipe errors
- `n_jobs=1` prevents multi-process contention
- SHAP generation fails gracefully (training continues without interruption)
- matplotlib resource auto-cleanup (`plt.clf` + `plt.close` + `gc.collect`)
- Environment variables set: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`

### Performance Optimization

- GZip middleware (>1KB responses auto-compressed)
- Request duration logging (millisecond precision)
- Tuning fallback: Optuna → scikit-learn automatic fallback on failure
- SMOTE safety check: auto-disabled when minority class < 6 samples
- NaN/Inf values filtered from metrics output

---

## Security

| Security Measure | Description |
|-----------------|-------------|
| Path traversal prevention | `realpath` validation + regex whitelist (`^[a-zA-Z0-9_-]+$`) |
| Filename safety | UUID prefix + original filename |
| Error message sanitization | Internal exceptions not exposed to users; friendly error messages |
| Session isolation | Each training task has independent random session_id |
| Resource limits | File size (100MB), rows (50K), columns (200), min rows (3) |
| XSS prevention | ReactMarkdown rendering instead of dangerouslySetInnerHTML |
| API Key local storage | Gemini Key stored only in browser localStorage |
| Upload validation | CSV-only, empty file check, parse failure auto-cleanup |

---

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| Outlier Removal | On | Isolation Forest 0.05 threshold |
| Advanced Imputation | On | Robust strategy (drop numeric + mode categorical) |
| Stratified CV | On | 3-Fold Stratified |
| Model Tuning | On | Bayesian optimization (Optuna with scikit-learn fallback) |
| Model Ensembling | On | Bagging + Boosting |
| SMOTE Balance | Off | User manually enables (classification only) |

> All advanced options can be independently toggled on the configuration page. States are passed to the backend training pipeline.

---

## Project Structure

```
VibeMine/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry, CORS/GZip/request timing middleware
│   │   ├── config.py             # Environment variable config (dirs, Gemini key, proxy, CORS)
│   │   ├── routes/
│   │   │   ├── train.py          # Training API routes (session management, stop, status polling)
│   │   │   ├── upload.py         # Upload API routes (validation, profiling)
│   │   │   └── download.py       # Download API routes (path safety)
│   │   └── services/
│   │       ├── train_service.py  # Core training service (PyCaret orchestration, SHAP, features)
│   │       ├── data_service.py   # Data processing service (CSV parse, profile, class distribution)
│   │       └── ai_service.py     # AI service (reserved, calls moved to frontend)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/index.ts          # API calls + Gemini frontend direct call
│   │   ├── store/index.ts        # Zustand state management (6 advanced option states, classDistributions)
│   │   ├── i18n/index.tsx        # I18n config (complete zh/en, 170+ keys)
│   │   ├── App.tsx               # Main application
│   │   └── components/
│   │       ├── StepUpload.tsx     # Upload component (i18n tabs: Data/Info/Stats)
│   │       ├── StepConfig.tsx    # Config component (6 advanced options, class balance display)
│   │       ├── StepTrain.tsx     # Training component (advanced options passthrough)
│   │       ├── StepEvaluate.tsx  # Evaluation component (independent AI loading, ReactMarkdown)
│   │       ├── StepExport.tsx    # Export component (model download, report, deploy code)
│   │       ├── SettingsModal.tsx # Settings modal (API Key config)
│   │       ├── Header.tsx        # Top navigation
│   │       └── Stepper.tsx       # Step indicator
│   ├── package.json
│   └── vite.config.ts
├── docs/
│   ├── README_ZH.md              # Chinese documentation
│   └── TEST_REPORT.md            # Test report
└── README.md
```

---

## Summary

VibeMine delivers optimal models through six systematic stages — from data quality assurance, algorithm diversity, intelligent hyperparameter search, model ensembling, to multi-dimensional evaluation — ensuring the final model achieves the best balance of accuracy, stability, and interpretability. Users without ML expertise can achieve results approaching those of professional data scientists.

### Quality Assurance Mechanisms

| Stage | Mechanism | Goal |
|-------|-----------|------|
| Data Preprocessing | Outlier detection + robust imputation | Data quality |
| Cross-Validation | Stratified 3-Fold CV | Evaluation reliability |
| Algorithm Selection | 27 algorithms competing | Find optimal algorithm |
| Hyperparameter Optimization | Bayesian search + fallback | Parameter optimization |
| Model Ensembling | Bagging + Boosting | Generalization stability |
| Result Verification | Multi-metric + AI interpretation | Result interpretability |

---

## Quick Start

```bash
# Clone the project
git clone https://github.com/willvibe/VibeMine.git
cd VibeMine

# Start backend
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Start frontend
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173 to start using VibeMine!

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | empty | Backend Gemini API key (optional, AI calls run on frontend) |
| `GEMINI_PROXY` | empty | HTTPS proxy for Gemini API (useful in China) |

Create a `.env` file in the `backend/` directory to configure these variables.
