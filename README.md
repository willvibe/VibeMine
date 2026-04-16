# VibeMine 智能 AutoML 平台

**AutoML Intelligence Platform — From Data to Best Model, Automatically**

---

## 一、平台概述

VibeMine 是一款端到端的智能自动化机器学习平台，基于 PyCaret 3.x 构建，提供从数据上传、特征工程、模型训练、性能评估到模型导出的全流程自动化服务。

用户无需编写一行代码，只需上传 CSV 数据文件，平台即可自动完成数据预处理、算法选择、超参数调优、模型集成，并输出最优模型的完整评估报告。

### 核心特性

- 🚀 **零代码 AutoML**：上传数据 → 配置 → 一键训练 → 获得最优模型
- 🧠 **AI 智能解读**：Google Gemini 驱动的模型评估与错误样本分析
- 🌍 **中英文切换**：完整 i18n 国际化支持
- 📊 **多维可视化**：柱状图、雷达图、SHAP 图、特征重要性图
- 🔧 **6 大高级选项**：异常检测、缺失值填充、分层CV、调优、集成、SMOTE
- 📦 **一键导出**：标准 PyCaret .pkl 模型文件 + Python 部署代码

---

## 二、核心优势：如何保障训练出最好的模型

传统机器学习需要人工处理数据清洗、特征工程、算法调参等多个环节，耗时且依赖经验。VibeMine 通过以下六个维度的系统化操作，确保模型达到最优表现：

1. **数据质量保障**：异常值检测 + 稳定缺失值填充
2. **评估可靠性**：分层交叉验证避免评估偏差
3. **算法多样性**：15 种算法并行竞速，覆盖所有主流模型
4. **超参数智能化**：贝叶斯优化自动搜索最优参数
5. **模型集成**：Bagging + Boosting 双策略提升稳定性
6. **多维度评估**：5 类指标 + 可视化 + AI 解读，确保模型真实可靠

---

## 三、端到端流程

```
数据上传 → 智能分析 → 参数配置 → 模型训练 → 多维评估 → 模型导出
   ↓           ↓           ↓           ↓           ↓           ↓
 CSV文件    列类型检测    高级选项    多模型竞速   柱状/雷达图   .pkl模型
            不平衡检测    SMOTE      贝叶斯调优   SHAP分析     部署代码
            AI建议       集成策略    模型集成     AI解读
```

---

## 四、三大任务类型

### 4.1 分类任务（Classification）

支持二分类和多分类，涵盖 13 种算法：

| 算法 | ID | 类型 |
|-----|-----|-----|
| Logistic Regression | lr | 线性 |
| Random Forest | rf | 集成- Bagging |
| Gradient Boosting | gbc | 集成- Boosting |
| Extra Trees | et | 集成- Bagging |
| XGBoost | xgb | 集成- Boosting |
| LightGBM | lightgbm | 集成- Boosting |
| Decision Tree | dt | 基于树 |
| KNN | knn | 基于距离 |
| AdaBoost | ada | 集成- Boosting |
| Linear Discriminant | lda | 线性 |
| Naive Bayes | nb | 概率 |
| Quadratic Discriminant | qda | 概率 |

**评估指标**：Accuracy、AUC（AUC_OVR/AUC_OVO 多分类）、Recall、Precision、F1、Kappa、MCC

### 4.2 回归任务（Regression）

支持 10 种回归算法：

| 算法 | ID | 类型 |
|-----|-----|-----|
| Linear Regression | lr | 线性 |
| Ridge Regression | ridge | 线性-正则化 |
| Lasso Regression | lasso | 线性-正则化 |
| ElasticNet | en | 线性-正则化 |
| Random Forest | rf | 集成- Bagging |
| Extra Trees | et | 集成- Bagging |
| Gradient Boosting | gbr | 集成- Boosting |
| LightGBM | lightgbm | 集成- Boosting |
| KNN | knn | 基于距离 |
| Decision Tree | dt | 基于树 |

**评估指标**：R2、RMSE、MAE、MSE、RMSLE

### 4.3 聚类任务（Clustering）

支持 5 种聚类算法：

| 算法 | ID |
|-----|-----|
| K-Means | kmeans |
| Hierarchical | hclust |
| DBSCAN | dbscan |
| Mean Shift | meanshift |
| Affinity Propagation | affinity |

**评估指标**：Silhouette、Calinski-Harabasz、Davies-Bouldin

---

## 五、数据预处理阶段

### 5.1 异常值检测与处理

使用 Isolation Forest 算法自动检测数据中的异常值：

```python
from sklearn.ensemble import IsolationForest

# 阈值 0.05（5% 异常值比例上限）
iso = IsolationForest(contamination=0.05, random_state=42)
outlier_mask = iso.fit_predict(data) != -1
```

### 5.2 缺失值填充

采用稳定的两阶段插补策略：

1. **drop**：删除含有过多缺失值的行或列
2. **mode/median**：对数值列使用中位数，对类别列使用众数填充剩余缺失值

### 5.3 特征类型推断

自动识别并区分：

- **数值特征**（int64、float64）→ 用于 KNN、线性模型、树模型
- **类别特征**（object、category）→ 自动编码后参与建模
- **目标列** → 排除在特征之外

---

## 六、模型训练阶段

### 6.1 交叉验证策略

- **分类任务**：3-Fold Stratified CV，确保每折中各类别比例与整体分布一致
- **回归任务**：3-Fold CV with shuffle
- Stratified CV 可避免某折因类别不均衡导致评估结果失真

### 6.2 超参数调优

使用 Optuna 框架进行贝叶斯优化：

- **迭代次数**：5 次（平衡效果与速度）
- **优化目标**：Accuracy（R2 for regression）
- **搜索空间**：树深度、学习率、特征比例等

### 6.3 模型集成

对表现良好的基础模型（RF、ET、GBC、ADA、XGB、LightGBM）自动进行 Bagging 或 Boosting 集成：

| 基础模型 | 集成方法 | 策略 |
|---------|---------|------|
| Random Forest、Extra Trees | Bagging | 降低方差 |
| GBC、AdaBoost、XGB、LightGBM | Boosting | 降低偏差 |

---

## 七、多维评估体系

### 7.1 指标计算

训练完成后，对所有模型计算完整的指标体系：

| 任务类型 | 指标 |
|---------|------|
| 分类 | Accuracy, AUC, Recall, Precision, F1, Kappa, MCC |
| 回归 | R2, RMSE, MAE, MSE, RMSLE |
| 聚类 | Silhouette, Calinski-Harabasz, Davies-Bouldin |

### 7.2 可视化

- **柱状图**：各模型多指标横向对比
- **雷达图**：单个模型多维度能力展示
- **特征重要性**：Top 15 重要特征
- **SHAP**：SHAP Summary Plot（支持 LightGBM/XGBoost）
- **混淆矩阵**：分类任务误差分析

### 7.3 AI 智能解读

基于 Google Gemini API，自动生成：

- **数据洞察**：分析数据分布、特征相关性、潜在问题
- **模型评估**：解释最优模型表现，给出改进建议
- **错误样本分析**：分析被错误分类的样本特征

---

## 八、前端界面

### 8.1 技术栈

- React 18 + TypeScript
- Zustand 状态管理
- ECharts 图表
- Tailwind CSS
- i18next 国际化
- Axios API 调用

### 8.2 页面流程

```
Step 1: 上传 → Step 2: 配置 → Step 3: 训练 → Step 4: 评估 → Step 5: 导出
```

### 8.3 主要组件

| 组件 | 功能 |
|-----|------|
| StepUpload | CSV 上传、数据预览、AI 洞察 |
| StepConfig | 任务类型、目标列、模型选择、SMOTE、高级选项 |
| StepTrain | 训练进度、模型日志、实时状态 |
| StepEvaluate | 指标对比、雷达图、特征重要性、AI 解读 |
| StepExport | 模型下载、部署代码生成 |

---

## 九、后端架构

### 9.1 技术栈

- FastAPI（异步 API）
- PyCaret 3.x（AutoML 引擎）
- Pandas + NumPy（数据处理）
- Google Generative AI（AI 洞察）
- Threading（并发训练）
- Logging（结构化日志）

### 9.2 核心服务

| 服务 | 文件 | 职责 |
|-----|------|------|
| train_service | 训练流程编排、PyCaret 封装 | |
| data_service | CSV 解析、数据统计 | |
| ai_service | Google Gemini API 调用 | |
| train.py | FastAPI 路由、会话管理 | |
| upload.py | 文件上传、安全校验 | |
| download.py | 模型下载、路径安全 | |

### 9.3 并发控制

- 使用 `threading.Lock` 保护会话字典
- 会话超时自动清理（3600 秒）
- 最多同时运行 50 个训练会话

---

## 十、安全机制

| 安全措施 | 说明 |
|---------|------|
| 路径遍历防护 | realpath 验证 + 白名单校验 |
| 文件名安全 | 仅允许字母数字和特定字符 |
| 错误信息隐藏 | 内部异常不暴露给用户 |
| 会话隔离 | 每个训练任务独立 session_id |
| 资源限制 | 文件大小、行数列数限制 |

---

## 十一、配置项总览

| 功能 | 默认 | 说明 |
|------|------|------|
| 异常值处理 | ✅ 启用 | Isolation Forest 0.05 阈值 |
| 高级缺失值填充 | ✅ 启用 | 稳定插补策略（drop + mode） |
| 分层交叉验证 | ✅ 启用 | 3-Fold Stratified |
| 模型调优 | ✅ 启用 | 贝叶斯优化（Optuna） |
| 模型集成 | ✅ 启用 | Bagging + Boosting |
| SMOTE 平衡 | ⏸ 关闭 | 需用户手动开启（仅分类） |

---

## 十二、总结

VibeMine 通过六个阶段的系统化处理，从数据质量保障、算法多样性、超参数智能化搜索、模型集成、到多维度评估，确保最终交付的模型在准确率、稳定性、可解释性上达到最优水平。用户无需机器学习专业知识，即可获得接近专业数据科学家调优效果的模型。

### 关键质量保障机制

| 阶段 | 机制 | 保障目标 |
|------|------|----------|
| 数据预处理 | 异常值检测 + 稳定插补 | 数据质量 |
| 交叉验证 | 分层 3-Fold CV | 评估可靠性 |
| 算法选择 | 15 种算法竞速 | 找到最优算法 |
| 超参数优化 | 贝叶斯搜索 | 参数最优化 |
| 模型集成 | Bagging + Boosting | 泛化稳定性 |
| 结果验证 | 多指标 + AI 解读 | 结果可解释性 |

---

## 快速开始

```bash
# 克隆项目
git clone https://github.com/willvibe/VibeMine.git
cd VibeMine

# 启动后端
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 启动前端
cd frontend
npm install
npm run dev
```

访问 http://localhost:5173 开始使用 VibeMine！
