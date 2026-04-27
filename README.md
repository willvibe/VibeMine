# 🚀 VibeMine 智能 AutoML 平台

> **AutoML Intelligence Platform — From Data to Best Model, Automatically**

![React](https://img.shields.io/badge/React-18.x-20232A?style=flat-square&logo=react&logoColor=61DAFB) ![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=flat-square&logo=typescript&logoColor=white) ![ECharts](https://img.shields.io/badge/ECharts-5.x-E43961?style=flat-square&logo=apacheecharts&logoColor=white) ![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.x-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-Async-009485?style=flat-square&logo=fastapi&logoColor=white) ![PyCaret](https://img.shields.io/badge/PyCaret-3.x-F9A03F?style=flat-square&logo=python&logoColor=white) ![AI](https://img.shields.io/badge/AI-Gemini_Driven-8E44AD?style=flat-square&logo=googlegemini&logoColor=white)



## 🌟 一、平台概述

VibeMine 是一款端到端的智能自动化机器学习平台，基于 **PyCaret 3.x** 构建。平台提供从数据上传、特征工程、模型训练、性能评估到模型导出的全流程自动化服务。

用户无需编写任何代码，只需上传 CSV 数据文件，平台即可自动完成数据预处理、算法选择、超参数调优、模型集成，并最终输出最优模型的完整评估报告。

### 🌐 在线体验 

- **主站 (Primary)**: http://45.113.2.167:5173/

### ✨ 核心特性

* 🚀 **零代码 AutoML**：上传数据 → 配置 → 一键训练 → 获得最优模型。
* 🧠 **AI 智能解读**：由 Google Gemini 驱动的模型评估与错误样本分析。
* 🌍 **多语言支持**：完整的 i18n 中英文国际化切换。
* 📊 **多维可视化**：内置柱状图、雷达图、SHAP 图及特征重要性图。
* 🔧 **6 大高级选项**：支持异常检测、缺失值填充、分层交叉验证 (CV)、模型调优、模型集成与 SMOTE 采样。
* 📦 **一键交付**：导出标准 PyCaret `.pkl` 模型文件、Python 部署代码及 Markdown 分析报告。

---

## 🏆 二、核心优势：如何保障最优模型？

传统机器学习依赖人工经验且耗时。VibeMine 通过以下六个维度的系统化操作，确保您的模型达到最优表现：

1.  **数据质量保障**：结合异常值检测与稳定的缺失值填充策略。
2.  **评估可靠性**：采用分层交叉验证（Stratified CV），避免评估偏差。
3.  **算法多样性**：支持 28 种算法并行竞速，覆盖当前所有主流模型。
4.  **超参数智能化**：利用贝叶斯优化，自动搜索并锁定最优参数。
5.  **模型集成**：运用 Bagging + Boosting 双策略，全面提升模型稳定性。
6.  **多维度评估**：结合 5 类指标、可视化图表与 AI 智能解读，确保结果真实可靠。

---

## 🔄 三、端到端工作流

> **CSV文件** → **数据预处理** (列类型/不平衡检测/AI建议) → **参数配置** (高级选项/SMOTE/集成) → **模型训练** (多模型竞速/贝叶斯调优) → **多维评估** (图表/SHAP/AI解读) → **成果导出** (.pkl模型/代码/报告)

---

## 🎯 四、支持的三大任务类型

### 1. 分类任务 (Classification)

支持二分类和多分类，涵盖 12 种算法，并提供 Accuracy、AUC、Recall、Precision、F1、Kappa、MCC 等评估指标：

| 算法全名                        | 算法简称 | 算法类型        |
| :------------------------------ | :------- | :-------------- |
| Logistic Regression             | lr       | 线性            |
| Linear Discriminant Analysis    | lda      | 线性            |
| Random Forest                   | rf       | 集成 - Bagging  |
| Extra Trees Classifier          | et       | 集成 - Bagging  |
| Gradient Boosting Classifier    | gbc      | 集成 - Boosting |
| XGBoost                         | xgb      | 集成 - Boosting |
| LightGBM                        | lightgbm | 集成 - Boosting |
| AdaBoost                        | ada      | 集成 - Boosting |
| Decision Tree                   | dt       | 基于树          |
| K-Nearest Neighbors             | knn      | 基于距离        |
| Naive Bayes                     | nb       | 概率            |
| Quadratic Discriminant Analysis | qda      | 概率            |

### 2. 回归任务 (Regression)

支持 10 种回归算法，评估指标包括 R2、RMSE、MAE、MSE、RMSLE：

| 算法全名                      | 算法简称 | 算法类型        |
| :---------------------------- | :------- | :-------------- |
| Linear Regression             | lr       | 线性            |
| Ridge Regression              | ridge    | 线性 - 正则化   |
| Lasso Regression              | lasso    | 线性 - 正则化   |
| Elastic Net                   | en       | 线性 - 正则化   |
| Random Forest Regressor       | rf       | 集成 - Bagging  |
| Extra Trees Regressor         | et       | 集成 - Bagging  |
| Gradient Boosting Regressor   | gbr      | 集成 - Boosting |
| LightGBM Regressor            | lightgbm | 集成 - Boosting |
| Decision Tree Regressor       | dt       | 基于树          |
| K-Nearest Neighbors Regressor | knn      | 基于距离        |

### 3. 聚类任务 (Clustering)

支持 5 种无监督聚类算法，评估指标包括 Silhouette、Calinski-Harabasz、Davies-Bouldin：

| 算法全名                | 算法简称  |
| :---------------------- | :-------- |
| K-Means                 | kmeans    |
| Hierarchical Clustering | hclust    |
| DBSCAN                  | dbscan    |
| Mean Shift              | meanshift |
| Affinity Propagation    | affinity  |

---

## ⚙️ 五、数据预处理与特征工程

* **异常值处理**：使用 Isolation Forest 算法（contamination=0.05）自动检测并隔离异常值。
* **缺失值填充**：采用两阶段插补策略。首先删除缺失严重的行列，随后对数值列使用中位数填充，类别列使用众数填充。
* **特征智能推断**：自动区分数值特征（int64, float64）与类别特征（object, category），并在建模前自动完成编码，同时安全排除目标列。

---

## 🧠 六、模型训练与优化机制

* **交叉验证**：分类任务使用 3-Fold Stratified CV 保障类别分布一致性；回归任务使用带有 shuffle 的 3-Fold CV。
* **贝叶斯优化**：集成 Optuna 框架进行 5 次迭代寻优，分类以 Accuracy 为目标，回归以 R2 为目标，智能搜索树深度、学习率等参数空间。
* **自动模型集成**：对表现优异的基础模型应用集成策略。Random Forest 和 Extra Trees 使用 Bagging 降低方差；GBC、AdaBoost、XGBoost、LightGBM 使用 Boosting 降低偏差。

---

## 📈 七、评估、可视化与 AI 解读

* **全景可视化**：提供各模型多指标横向对比的柱状图、展示单个模型多维能力的雷达图、标记 Top 15 特征重要性的渐变图，以及支持 LightGBM/XGBoost 的 SHAP Summary Plot。
* **混淆矩阵**：专为分类任务提供详尽的误差分析。
* **Gemini AI 洞察**：通过前端直调 Google Gemini API（Key 仅存本地），自动生成数据分布分析、最优模型解读、改进建议及被错误分类样本的深度剖析。

---

## 💻 八、系统架构与技术栈

### 前端应用 (Frontend)

* **核心栈**：React 18 + TypeScript + Vite。
* **状态与 UI**：Zustand 状态管理，Tailwind CSS 样式，ECharts 图表渲染。
* **主要组件**：包含 StepUpload（上传与预览）、StepConfig（任务与参数配置）、StepTrain（状态轮询与日志）、StepEvaluate（可视化与 AI）及 StepExport（下载与导出）。

### 后端服务 (Backend)

* **核心栈**：FastAPI 提供异步 API，PyCaret 3.x 驱动 AutoML 引擎，Pandas 与 NumPy 处理数据。
* **并发控制**：最多同时运行 50 个训练会话，使用 `threading.Lock` 保护会话字典，超时（3600 秒）自动清理。
* **进程安全**：采用 `spawn` 启动方法避免 Broken Pipe，设置 `n_jobs=1` 防止多进程竞争，并包含 matplotlib 资源自动清理机制（plt.clf/close）。

---

## 🛡️ 九、安全与默认配置

平台内置严格的安全机制，包括路径遍历防护（realpath + 正则白名单）、文件名安全校验、内部异常隐藏、会话隔离（独立随机 session_id）、防 XSS 注入及 Gemini API Key 本地存储隔离。

**默认高级配置概览：**

| 功能模块   | 默认状态 | 配置详情                            |
| :--------- | :------- | :---------------------------------- |
| 异常值处理 | ✅ 启用   | Isolation Forest (阈值 0.05)        |
| 缺失值填充 | ✅ 启用   | 稳定两阶段插补 (Drop + Mode/Median) |
| 交叉验证   | ✅ 启用   | 3-Fold Stratified CV                |
| 模型调优   | ✅ 启用   | Optuna 贝叶斯优化                   |
| 模型集成   | ✅ 启用   | Bagging + Boosting 双策略           |
| SMOTE 平衡 | ⏸ 关闭   | 按需手动开启（仅限分类任务）        |

---

## 🚀 十、快速开始

```bash
# 1. 克隆项目
git clone [https://github.com/willvibe/VibeMine.git](https://github.com/willvibe/VibeMine.git)
cd VibeMine

# 2. 启动后端服务
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. 启动前端服务
cd ../frontend
npm install
npm run dev
```
