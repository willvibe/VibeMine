import { useState } from 'react';
import { useAppStore } from '../store';
import ReactMarkdown from 'react-markdown';

const ALL_MODELS: Record<string, { id: string; name: string }[]> = {
  classification: [
    { id: 'lr', name: 'Logistic Regression' },
    { id: 'rf', name: 'Random Forest' },
    { id: 'gbc', name: 'Gradient Boosting' },
    { id: 'et', name: 'Extra Trees' },
    { id: 'xgb', name: 'Extreme Gradient Boosting (XGBoost)' },
    { id: 'lightgbm', name: 'Light GBM' },
    { id: 'dt', name: 'Decision Tree' },
    { id: 'knn', name: 'K Neighbors' },
    { id: 'ada', name: 'Ada Boost' },
    { id: 'lda', name: 'Linear Discriminant Analysis' },
    { id: 'qda', name: 'Quadratic Discriminant Analysis' },
    { id: 'nb', name: 'Naive Bayes' },
    { id: 'svm', name: 'SVM (Linear)' },
    { id: 'ridge', name: 'Ridge Classifier' },
  ],
  regression: [
    { id: 'lr', name: 'Linear Regression' },
    { id: 'ridge', name: 'Ridge' },
    { id: 'lasso', name: 'Lasso' },
    { id: 'rf', name: 'Random Forest' },
    { id: 'et', name: 'Extra Trees' },
    { id: 'gbr', name: 'Gradient Boosting' },
    { id: 'lightgbm', name: 'Light GBM' },
    { id: 'xgb', name: 'Extreme Gradient Boosting' },
    { id: 'dt', name: 'Decision Tree' },
    { id: 'knn', name: 'K Neighbors' },
    { id: 'svm', name: 'SVM' },
  ],
  clustering: [
    { id: 'kmeans', name: 'K-Means' },
    { id: 'hclust', name: 'Hierarchical Clustering' },
    { id: 'meanshift', name: 'Mean Shift' },
    { id: 'dbscan', name: 'DBSCAN' },
    { id: 'affinity', name: 'Affinity Propagation' },
  ],
};

const DEFAULT_MODELS: Record<string, string[]> = {
  classification: ['lr', 'rf', 'gbc', 'et', 'xgb'],
  regression: ['lr', 'ridge', 'rf', 'gbr', 'et'],
  clustering: ['kmeans', 'hclust', 'meanshift'],
};

export default function StepConfig() {
  const columns = useAppStore((s) => s.columns);
  const shape = useAppStore((s) => s.shape);
  const preview = useAppStore((s) => s.preview);
  const aiInsight = useAppStore((s) => s.aiInsight);
  const columnDetails = useAppStore((s) => s.columnDetails);
  const taskType = useAppStore((s) => s.taskType);
  const targetColumn = useAppStore((s) => s.targetColumn);
  const selectedModels = useAppStore((s) => s.selectedModels);
  const ignoreColumns = useAppStore((s) => s.ignoreColumns);
  const useSmote = useAppStore((s) => s.useSmote);
  const imbalanceDetected = useAppStore((s) => s.imbalanceDetected);
  const imbalanceRatio = useAppStore((s) => s.imbalanceRatio);
  const setTaskType = useAppStore((s) => s.setTaskType);
  const setTargetColumn = useAppStore((s) => s.setTargetColumn);
  const setSelectedModels = useAppStore((s) => s.setSelectedModels);
  const setIgnoreColumns = useAppStore((s) => s.setIgnoreColumns);
  const setUseSmote = useAppStore((s) => s.setUseSmote);
  const setStep = useAppStore((s) => s.setStep);

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showIgnoreSelect, setShowIgnoreSelect] = useState(false);

  const handleTaskTypeChange = (type: string) => {
    setTaskType(type);
    if (type === 'clustering') {
      setTargetColumn('');
    }
    setSelectedModels(DEFAULT_MODELS[type] || []);
  };

  const toggleModel = (modelId: string) => {
    if (selectedModels.includes(modelId)) {
      if (selectedModels.length > 1) {
        setSelectedModels(selectedModels.filter((m) => m !== modelId));
      }
    } else {
      setSelectedModels([...selectedModels, modelId]);
    }
  };

  const toggleIgnoreColumn = (col: string) => {
    if (ignoreColumns.includes(col)) {
      setIgnoreColumns(ignoreColumns.filter((c) => c !== col));
    } else {
      setIgnoreColumns([...ignoreColumns, col]);
    }
  };

  const selectAllModels = () => {
    setSelectedModels(ALL_MODELS[taskType].map((m) => m.id));
  };

  const selectDefaultModels = () => {
    setSelectedModels(DEFAULT_MODELS[taskType] || []);
  };

  const canProceed = taskType === 'clustering' || targetColumn !== '';

  const selectableColumns = columns.filter((col) => col !== targetColumn);

  return (
    <div className="max-w-6xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">数据概览与任务配置</h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-700 mb-3 flex items-center gap-2">
              <span className="text-indigo-500">📋</span> 数据预览
              <span className="text-sm font-normal text-gray-400">
                {shape[0]} 行 × {shape[1]} 列
              </span>
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50">
                    {columns.map((col) => (
                      <th key={col} className="px-3 py-2 text-left font-medium text-gray-600 whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.map((row, i) => (
                    <tr key={i} className="border-t border-gray-100">
                      {columns.map((col) => (
                        <td key={col} className="px-3 py-2 text-gray-500 whitespace-nowrap max-w-[150px] truncate">
                          {String(row[col] ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-700 mb-3 flex items-center gap-2">
              <span className="text-indigo-500">📊</span> 列统计信息
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="px-3 py-2 text-left font-medium text-gray-600">列名</th>
                    <th className="px-3 py-2 text-left font-medium text-gray-600">类型</th>
                    <th className="px-3 py-2 text-left font-medium text-gray-600">缺失</th>
                    <th className="px-3 py-2 text-left font-medium text-gray-600">唯一值</th>
                  </tr>
                </thead>
                <tbody>
                  {columnDetails.map((col) => (
                    <tr key={col.name} className="border-t border-gray-100">
                      <td className="px-3 py-2 font-medium text-gray-700">{col.name}</td>
                      <td className="px-3 py-2 text-gray-500">{col.dtype}</td>
                      <td className="px-3 py-2">
                        {col.missing_count > 0 ? (
                          <span className="text-red-500 font-medium">
                            {col.missing_count} ({(col.missing_ratio * 100).toFixed(1)}%)
                          </span>
                        ) : (
                          <span className="text-green-500">0</span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-gray-500">{col.unique_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl border border-indigo-100 p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-indigo-700 mb-3 flex items-center gap-2">
              <span>🤖</span> AI 智能建议
            </h3>
            <div className="prose prose-sm prose-indigo max-w-none text-gray-700">
              <ReactMarkdown>{aiInsight}</ReactMarkdown>
            </div>
          </div>

          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-700 mb-4 flex items-center gap-2">
              <span className="text-indigo-500">⚙️</span> 任务配置
            </h3>

            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-gray-600 mb-2">任务类型</label>
                <div className="flex gap-3">
                  {[
                    { value: 'classification', label: '分类', icon: '🏷️' },
                    { value: 'regression', label: '回归', icon: '📈' },
                    { value: 'clustering', label: '聚类', icon: '🎯' },
                  ].map((opt) => (
                    <button
                      key={opt.value}
                      onClick={() => handleTaskTypeChange(opt.value)}
                      className={`flex-1 py-3 px-4 rounded-xl border-2 font-medium transition-all duration-200 ${
                        taskType === opt.value
                          ? 'border-indigo-500 bg-indigo-50 text-indigo-700 shadow-sm'
                          : 'border-gray-200 bg-white text-gray-500 hover:border-gray-300'
                      }`}
                    >
                      <span className="mr-2">{opt.icon}</span>
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              {taskType !== 'clustering' && (
              <div>
                <label className="block text-sm font-medium text-gray-600 mb-2">目标列 (Target)</label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  className="w-full py-3 px-4 rounded-xl border-2 border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 outline-none transition-all text-gray-700 bg-white"
                >
                  <option value="">-- 请选择目标列 --</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
              )}

              {taskType === 'clustering' && (
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
                <p className="text-sm text-amber-700">
                  <span className="font-medium">💡 聚类说明：</span>聚类是一种无监督学习任务，无需指定目标列。系统将自动发现数据中的自然分组。
                </p>
              </div>
              )}

              {/* BUG FIX: only show SMOTE block when imbalance is actually detected */}
              {taskType === 'classification' && imbalanceDetected && (
              <div className="bg-orange-50 border border-orange-200 rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-orange-700 font-medium">⚠️ 样本不平衡检测</p>
                    <p className="text-xs text-orange-600 mt-1">
                      系统检测到类别比例约为 {imbalanceRatio}:1
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={useSmote}
                      onChange={(e) => setUseSmote(e.target.checked)}
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-orange-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-orange-500"></div>
                  </label>
                </div>
                <p className="text-xs text-orange-600 mt-2">
                  {useSmote ? '✓ 已开启 SMOTE 过采样' : '开启后将使用 SMOTE 自动平衡样本'}
                </p>
              </div>
              )}

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-gray-600">
                    忽略的列 ({ignoreColumns.length}个)
                  </label>
                  <button
                    onClick={() => setShowIgnoreSelect(!showIgnoreSelect)}
                    className="text-sm text-indigo-600 hover:text-indigo-700"
                  >
                    {showIgnoreSelect ? '收起 ▲' : '设置 ▼'}
                  </button>
                </div>

                {showIgnoreSelect && (
                  <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 max-h-40 overflow-y-auto">
                    <div className="flex flex-wrap gap-2">
                      {selectableColumns.map((col) => (
                        <label
                          key={col}
                          className={`flex items-center gap-1 px-2 py-1 rounded cursor-pointer text-sm transition-colors ${
                            ignoreColumns.includes(col)
                              ? 'bg-red-100 text-red-700 border border-red-200'
                              : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={ignoreColumns.includes(col)}
                            onChange={() => toggleIgnoreColumn(col)}
                            className="sr-only"
                          />
                          {col}
                        </label>
                      ))}
                    </div>
                    {ignoreColumns.length > 0 && (
                      <button
                        onClick={() => setIgnoreColumns([])}
                        className="mt-2 text-xs text-gray-500 hover:text-gray-700"
                      >
                        清除全部
                      </button>
                    )}
                  </div>
                )}
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-gray-600">
                    参与训练的算法 ({selectedModels.length}个)
                  </label>
                  <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="text-sm text-indigo-600 hover:text-indigo-700"
                  >
                    {showAdvanced ? '收起 ▲' : '展开 ▼'}
                  </button>
                </div>

                {showAdvanced && (
                  <div className="space-y-3 mt-3 p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="flex gap-2 mb-2">
                      <button
                        onClick={selectDefaultModels}
                        className="px-3 py-1 text-xs bg-indigo-100 text-indigo-700 rounded-lg hover:bg-indigo-200"
                      >
                        默认5个
                      </button>
                      <button
                        onClick={selectAllModels}
                        className="px-3 py-1 text-xs bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
                      >
                        全选
                      </button>
                    </div>
                    <div className="grid grid-cols-2 gap-2 max-h-60 overflow-y-auto">
                      {ALL_MODELS[taskType]?.map((model) => (
                        <label
                          key={model.id}
                          className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-colors ${
                            selectedModels.includes(model.id)
                              ? 'bg-indigo-50 border border-indigo-200'
                              : 'bg-white border border-gray-200 hover:bg-gray-50'
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={selectedModels.includes(model.id)}
                            onChange={() => toggleModel(model.id)}
                            className="w-4 h-4 text-indigo-600 rounded"
                          />
                          <span className="text-sm text-gray-700 truncate">{model.name}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => setStep(0)}
              className="flex-1 py-3 px-6 rounded-xl border-2 border-gray-200 text-gray-600 font-medium hover:bg-gray-50 transition-all"
            >
              ← 重新上传
            </button>
            <button
              onClick={() => setStep(2)}
              disabled={!canProceed}
              className={`flex-1 py-3 px-6 rounded-xl font-medium transition-all ${
                canProceed
                  ? 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-md shadow-indigo-200'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              }`}
            >
              开始挖掘 🚀
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
