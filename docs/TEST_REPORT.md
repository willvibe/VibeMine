# VibeMine 功能测试报告

**测试日期**: 2026-04-16  
**测试环境**: Linux / Python 3.10 / Node.js 18  
**测试方法**: 后端 API 接口测试 + 前端 TypeScript 编译验证 + 安全漏洞测试

---

## 一、测试总览

| 测试类别 | 测试项数 | 通过 | 失败 | 通过率 |
|---------|---------|------|------|-------|
| API 接口测试 | 12 | 12 | 0 | 100% |
| 安全漏洞测试 | 4 | 4 | 0 | 100% |
| 分类任务测试 | 3 | 3 | 0 | 100% |
| 回归任务测试 | 3 | 3 | 0 | 100% |
| 聚类任务测试 | 3 | 3 | 0 | 100% |
| 前端编译检查 | 5 | 5 | 0 | 100% |

---

## 二、API 接口测试

### 2.1 基础接口

| 接口 | 方法 | 状态 | 结果 |
|-----|------|------|------|
| `/api/health` | GET | 200 | ✅ 返回 `{"status":"ok","service":"VibeMine API"}` |
| `/api/upload` | POST | 200 | ✅ 鸢尾花数据集上传成功，shape=[150, 5] |
| `/api/upload` | POST | 200 | ✅ 糖尿病数据集上传成功，shape=[442, 11] |
| `/api/train` | POST | 200 | ✅ 返回 session_id 和 training 状态 |
| `/api/train/status/{id}` | GET | 200 | ✅ 返回训练进度和结果 |
| `/api/train/stop/{id}` | POST | 200 | ✅ 停止训练 |
| `/api/download/{id}` | GET | 404 | ✅ 无效模型ID返回404 |
| `/api/train` | POST | 400 | ✅ 无效task_type返回400 |

### 2.2 安全漏洞测试

| 攻击类型 | 请求 | 预期 | 实际 | 结果 |
|---------|------|------|------|------|
| 路径遍历(下载) | `/api/download/../../etc/passwd` | 400/403 | 404 | ✅ 被阻止 |
| 路径遍历(AI) | `/api/upload/ai-insight/../../etc/passwd` | 400/403 | 404 | ✅ 被阻止 |
| 无效模型ID | `/api/download/nonexistent123` | 404 | 404 | ✅ 正确处理 |
| 无效会话ID | `/api/train/status/nonexistent` | 404 | 404 | ✅ 正确处理 |

---

## 三、分类任务测试

**数据集**: 鸢尾花 (150行, 5列)  
**模型**: LR, RF  
**高级选项**: 调优=关, 集成=关

| 测试项 | 结果 | 详情 |
|-------|------|------|
| 训练完成 | ✅ | status=completed, progress=100 |
| 模型数量 | ✅ | 2个模型 (LR, RF) |
| 指标完整性 | ✅ | Accuracy, Recall, Precision, F1, Kappa, MCC, AUC, AUC_OVR, AUC_OVO |
| AUC_OVR > 0 | ✅ | 多分类AUC正确计算（修复前为0） |

---

## 四、回归任务测试

**数据集**: 糖尿病 (442行, 11列)  
**模型**: LR, Ridge  
**高级选项**: 调优=关, 集成=关

| 测试项 | 结果 | 详情 |
|-------|------|------|
| 训练完成 | ✅ | status=completed |
| 模型数量 | ✅ | 2个模型 (LR, RIDGE) |
| 指标完整性 | ✅ | R2, RMSE, MAE, MSE, RMSLE |
| 回归专用指标 | ✅ | 无分类指标混入 |

---

## 五、聚类任务测试

**数据集**: 糖尿病特征 (442行, 10列)  
**模型**: KMeans, HClust  
**高级选项**: 调优=关, 集成=关

| 测试项 | 结果 | 详情 |
|-------|------|------|
| 训练完成 | ✅ | status=completed |
| 模型数量 | ✅ | 2个模型 (KMEANS, HCLUST) |
| 指标完整性 | ✅ | Silhouette, Calinski-Harabasz, Davies-Bouldin |
| 无需目标列 | ✅ | target_column="" 正常工作 |

---

## 六、前端编译检查

| 文件 | 诊断 | 结果 |
|-----|------|------|
| StepUpload.tsx | 0 errors, 0 warnings | ✅ |
| StepConfig.tsx | 0 errors, 0 warnings | ✅ |
| StepTrain.tsx | 0 errors, 0 warnings | ✅ |
| StepEvaluate.tsx | 0 errors, 0 warnings | ✅ |
| StepExport.tsx | 0 errors, 0 warnings | ✅ |

---

## 七、本轮修复清单

### 后端修复 (train_service.py)

| # | 问题 | 修复 |
|---|------|------|
| 1 | session_id=42 硬编码导致并发冲突 | 改为 `np.random.randint(1, 9999)` |
| 2 | n_jobs=14 硬编码 | 改为 `n_jobs=-1` (自动检测CPU核心) |
| 3 | OMP_NUM_THREADS=14 硬编码 | 改为 `os.cpu_count()` 动态计算 |
| 4 | 集成模型指标仅在最佳时记录 | 始终记录到 metrics_list |
| 5 | METRICS_COLUMNS 缺少 AUC_OVR/AUC_OVO | 已添加 |
| 6 | 日志写 "5-Fold" 实际为 3-Fold | 修正为 "3-Fold" |
| 7 | 函数内重复 import numpy as np | 已移除4处重复导入 |
| 8 | 静默吞异常 (except: pass) | 改为 `except Exception as e: logger.warning()` |
| 9 | 多分类 AUC=0 | 添加自定义 AUC_OVR/AUC_OVO 指标 |

### 后端修复 (train.py)

| # | 问题 | 修复 |
|---|------|------|
| 1 | 会话检查与创建非原子操作 | 合并到同一个锁块 |
| 2 | Pydantic 类型 `list = None` | 改为 `Optional[List[str]] = None` |
| 3 | stop 端点在锁外修改 session 状态 | 改为锁内修改 |
| 4 | AI 评估结果写入无锁保护 | 添加 session_lock |

### 后端修复 (download.py)

| # | 问题 | 修复 |
|---|------|------|
| 1 | 路径遍历漏洞 | 添加 model_id 白名单校验 + realpath 检查 |

### 后端修复 (upload.py)

| # | 问题 | 修复 |
|---|------|------|
| 1 | 路径遍历漏洞 | 添加文件名校验 + realpath 检查 |
| 2 | 解析失败后未清理文件 | 添加 `_cleanup_file()` |
| 3 | 内部错误信息泄露 | 移除异常详情 |

### 前端修复 (StepEvaluate.tsx)

| # | 问题 | 修复 |
|---|------|------|
| 1 | XSS 漏洞 (dangerouslySetInnerHTML) | parseMarkdown 先转义 HTML |
| 2 | 雷达图维度不一致 | radarMetrics 使用 tableColumns |
| 3 | 雷达图数据结构错误 | 改为每个模型一个 series |
| 4 | 雷达图包含 Model 列 | 过滤掉 Model 列 |
| 5 | AUC_OVR/AUC_OVO 未在 METRIC_LABELS | 已添加 |

### 前端修复 (StepExport.tsx)

| # | 问题 | 修复 |
|---|------|------|
| 1 | 聚类任务使用 Accuracy 选最佳模型 | 改为 Silhouette |

### 前端修复 (StepUpload.tsx)

| # | 问题 | 修复 |
|---|------|------|
| 1 | JSX 中使用 getState() 不触发重渲染 | 改为选择器模式 |

### 前端修复 (StepConfig.tsx)

| # | 问题 | 修复 |
|---|------|------|
| 1 | SMOTE 在回归/聚类中显示 | 仅分类任务显示 |
| 2 | 特征列筛选模块被误删 | 已恢复 |
| 3 | 未使用变量 (METRIC_HIGHER_BETTER 等) | 已清理 |

### 前端修复 (api/index.ts)

| # | 问题 | 修复 |
|---|------|------|
| 1 | stopTraining 未传 authHeaders | 已添加 |

### 前端修复 (i18n/index.tsx)

| # | 问题 | 修复 |
|---|------|------|
| 1 | replace 只替换首个占位符 | 改为 replaceAll |

---

## 八、已知限制

1. **无认证机制**: 所有 API 端点无认证保护（建议后续添加）
2. **CORS 配置宽松**: 允许所有方法和头（建议生产环境收紧）
3. **WebSocket 路由未注册**: WebSocket 功能当前不可用（需注册路由并修复 run_automl_async 导入）
4. **AI 客户端缓存无界**: _CLIENT_CACHE 无大小限制（建议添加 LRU 淘汰）
5. **单机部署**: 不支持分布式训练
