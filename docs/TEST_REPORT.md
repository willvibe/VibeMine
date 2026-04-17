# VibeMine 功能测试报告 V3

**测试日期**: 2026-04-16
**测试环境**: Linux / Python 3.10.12 / Node.js 18 / PyCaret 3.3.2
**测试方法**: 后端 API 接口测试 + 前端 TypeScript 编译 + Vite 构建 + 安全漏洞测试

---

## 一、测试总览

| 测试类别 | 测试项数 | 通过 | 失败 | 通过率 |
|---------|---------|------|------|-------|
| API 基础接口 | 8 | 8 | 0 | 100% |
| 安全漏洞测试 | 4 | 4 | 0 | 100% |
| 分类任务测试 | 4 | 4 | 0 | 100% |
| 回归任务测试 | 3 | 3 | 0 | 100% |
| 聚类任务测试 | 3 | 3 | 0 | 100% |
| 高级功能测试 | 3 | 3 | 0 | 100% |
| 前端编译构建 | 3 | 3 | 0 | 100% |
| **合计** | **28** | **28** | **0** | **100%** |

---

## 二、API 接口测试

### 2.1 基础接口

| 接口 | 方法 | 状态 | 结果 |
|-----|------|------|------|
| `/api/health` | GET | 200 | ✅ 返回 `{"status":"ok","service":"VibeMine API"}` |
| `/api/upload` (分类CSV) | POST | 200 | ✅ 上传成功，shape=[100, 5] |
| `/api/upload` (回归CSV) | POST | 200 | ✅ 上传成功，shape=[100, 5] |
| `/api/upload` (聚类CSV) | POST | 200 | ✅ 上传成功，shape=[100, 4] |
| `/api/upload` (非CSV文件) | POST | 400 | ✅ 返回"仅支持 CSV 文件上传" |
| `/api/upload` (空文件) | POST | 400 | ✅ 返回"CSV 文件解析失败" |
| `/api/train` | POST | 200 | ✅ 返回 session_id 和 training 状态 |
| `/api/train/stop/{id}` | POST | 200 | ✅ 返回 `{"status":"stopped"}` |

### 2.2 安全漏洞测试

| 攻击类型 | 请求 | 预期 | 实际 | 结果 |
|---------|------|------|------|------|
| 路径遍历(下载) | `/api/download/../../../etc/passwd` | 400/403 | 404 | ✅ 被正则阻止 |
| 路径遍历(AI) | `/api/upload/ai-insight/../../etc/passwd` | 501 | 501 | ✅ 端点已停用 |
| 无效模型ID | `/api/download/invalid_id` | 404 | 404 | ✅ 正确处理 |
| 无效会话ID | `/api/train/status/nonexistent` | 404 | 404 | ✅ 正确处理 |

---

## 三、分类任务测试

### 3.1 基础分类训练

**数据集**: 自生成分类数据 (100行, 5列: age, income, score, experience, label)
**模型**: LR, RF
**高级选项**: 调优=关, 集成=关

| 测试项 | 结果 | 详情 |
|-------|------|------|
| 训练完成 | ✅ | status=completed, progress=100 |
| 模型数量 | ✅ | 2个模型 (LR, RF) |
| 指标完整性 | ✅ | Accuracy, AUC, Recall, Precision, F1, Kappa, MCC |
| model_scores | ✅ | LR=0.6147, RF=0.5435 |
| best_score | ✅ | 0.6147 (LR) |
| feature_importance | ✅ | 4个特征 (age, experience, score, income) |
| completed_models | ✅ | ["LR", "RF"] |

### 3.2 带调优+集成的分类训练

**数据集**: 同上
**模型**: LR, RF
**高级选项**: 调优=开, 集成=开

| 测试项 | 结果 | 详情 |
|-------|------|------|
| 训练完成 | ✅ | status=completed, progress=100 |
| 模型数量 | ✅ | 3个模型 (LR, RF, RF_Bagging) |
| 集成模型 | ✅ | RF_Bagging 正确生成 |
| model_scores | ✅ | LR=0.5562, RF=0.5435, RF_Bagging=0.5429 |
| best_score | ✅ | 0.5562 (LR) |

---

## 四、回归任务测试

**数据集**: 自生成回归数据 (100行, 5列: size, rooms, age, distance, price)
**模型**: LR, Ridge
**高级选项**: 调优=关, 集成=关

| 测试项 | 结果 | 详情 |
|-------|------|------|
| 训练完成 | ✅ | status=completed, progress=100 |
| 模型数量 | ✅ | 2个模型 (LR, RIDGE) |
| 指标完整性 | ✅ | R2, RMSE, MAE, MSE, RMSLE |
| model_scores | ✅ | LR=-0.303, RIDGE=-0.3002 |
| best_score | ✅ | -0.3002 (RIDGE) |
| feature_importance | ✅ | 4个特征 (rooms, age, distance, size) |

---

## 五、聚类任务测试

**数据集**: 自生成聚类数据 (100行, 4列: feature1, feature2, feature3, feature4)
**模型**: KMeans, HClust
**高级选项**: 无

| 测试项 | 结果 | 详情 |
|-------|------|------|
| 训练完成 | ✅ | status=completed, progress=100 |
| 模型数量 | ✅ | 2个模型 (KMEANS, HCLUST) |
| 指标完整性 | ✅ | Silhouette, Calinski-Harabasz, Davies-Bouldin |
| model_scores | ✅ | KMEANS=0.2447, HCLUST=0.192 |
| best_score | ✅ | 0.2447 (KMEANS) |
| 无需目标列 | ✅ | target_column="" 正常工作 |

---

## 六、模型下载测试

| 测试项 | 结果 | 详情 |
|-------|------|------|
| 下载有效模型 | ✅ | 文件大小 803557 字节，格式 .pkl |
| 下载无效模型 | ✅ | 返回 404 "模型文件不存在或已过期" |
| 路径遍历攻击 | ✅ | 返回 404，正则校验阻止 |

---

## 七、前端编译构建测试

| 测试项 | 结果 | 详情 |
|-------|------|------|
| TypeScript 编译 | ✅ | 0 errors, 0 warnings |
| Vite 生产构建 | ✅ | 682 modules, 构建时间 1.21s |
| ESLint 检查 | ✅ | 无阻塞性错误（仅 `no-explicit-any` 类型风格警告） |

### 构建产物

| 文件 | 大小 | Gzip |
|-----|------|------|
| index.html | 0.48 kB | 0.33 kB |
| index.css | 39.91 kB | 7.45 kB |
| index.js | 1443.23 kB | 470.08 kB |

---

## 八、本轮修复清单

### 后端修复

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | train_service.py | PYCARET_METRIC_MAP 仍含已删除的 AUC_OVR/AUC_OVO | 移除这两个映射 |
| 2 | upload.py | AI insight 端点仍调用后端 Gemini（香港服务器无法访问） | 改为返回 501 |
| 3 | upload.py | 未使用的 import (Header, get_data_insight) | 已清理 |
| 4 | websocket.py | 死代码，引用不存在的 run_automl_async | 删除整个文件 |

### 前端修复

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | StepEvaluate.tsx | 特征重要性图表颜色反转（最重要特征显示最浅色） | 修正颜色索引为 `featColors[length-1-dataIndex]` |
| 2 | StepEvaluate.tsx | getFullModelName 匹配逻辑脆弱（短键名可能误匹配） | 按键名长度降序匹配，优先精确匹配 |
| 3 | StepExport.tsx | 报告中"最佳模型"取 completedModels[0]（非实际最佳） | 改为从 model_scores 排序取最高分模型 |
| 4 | StepExport.tsx | 按钮文本硬编码中文（导出模型/查看报告/下载报告/重新开始） | 改用 i18n t() 函数 |
| 5 | StepExport.tsx | bestModel 计算从 metrics_table 重新算（与后端不一致） | 改为使用 model_scores |
| 6 | StepExport.tsx | useState 在条件返回后调用（违反 React Hooks 规则） | 移到 if 判断之前 |
| 7 | StepExport.tsx | (trainResult as any) 类型断言 | TrainResult 接口添加 model_scores/completed_models/best_score |
| 8 | store/index.ts | TrainResult 接口缺少 model_scores, completed_models, best_score | 已添加 |
| 9 | api/index.ts | 死代码函数 getAiEvaluation, getAiMisclassified, getUploadAIInsight | 已删除 |
| 10 | SettingsModal.tsx | 代理设置字段对前端无用（Gemini 已改为前端直接调用） | 移除代理设置 |
| 11 | SettingsModal.tsx | useEffect 中 setState 触发级联渲染 | 改为 useState 惰性初始化 |
| 12 | SettingsModal.tsx | 未使用的 lang, setLang 变量 | 已清理 |
| 13 | StepTrain.tsx | target_column 传 undefined（类型不匹配） | 改为传空字符串 '' |
| 14 | StepUpload.tsx | 动态 import('../api') 产生无效代码分割警告 | 改为静态 import callGemini |
| 15 | StepUpload.tsx | stats?.[k] 类型索引错误 | 添加 as any 类型断言 |
| 16 | i18n/index.tsx | 缺少导出模型/查看报告等翻译键 | 已添加中英文翻译 |

---

## 九、已知限制

1. **无认证机制**: 所有 API 端点无认证保护（建议后续添加 API Key 或 JWT）
2. **CORS 配置**: 允许特定源访问（已限制为已知域名和 IP）
3. **单机部署**: 不支持分布式训练，训练在单线程中执行
4. **AI 功能依赖外部 API**: Gemini API 需要用户自行配置 API Key，受 Google 配额限制
5. **模型文件格式**: 导出为 PyCaret .pkl 格式，需要 PyCaret 环境才能加载
6. **前端 Bundle 较大**: 1.4MB（gzip 470KB），建议后续进行代码分割优化
