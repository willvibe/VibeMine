import { useState } from 'react';
import { useAppStore } from '../store';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts/core';
import { BarChart } from 'echarts/charts';
import {
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent,
  VisualMapComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import ReactMarkdown from 'react-markdown';

echarts.use([
  BarChart,
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

const METRIC_OPTIONS: Record<string, { label: string; yAxisMin: number; yAxisMax?: number; higherIsBetter: boolean }> = {
  Accuracy:             { label: '准确率 (Accuracy)',       yAxisMin: 0.5,  higherIsBetter: true  },
  AUC:                  { label: 'AUC',                     yAxisMin: 0.5, yAxisMax: 1.0, higherIsBetter: true  },
  Recall:               { label: '召回率 (Recall)',          yAxisMin: 0.5,  higherIsBetter: true  },
  Precision:            { label: '精确率 (Precision)',       yAxisMin: 0.5,  higherIsBetter: true  },
  F1:                   { label: 'F1 分数',                  yAxisMin: 0.5,  higherIsBetter: true  },
  R2:                   { label: 'R² 分数',                  yAxisMin: 0.0, yAxisMax: 1.0, higherIsBetter: true  },
  RMSE:                 { label: 'RMSE',                    yAxisMin: 0.0,  higherIsBetter: false },
  MAE:                  { label: 'MAE',                     yAxisMin: 0.0,  higherIsBetter: false },
  Silhouette:           { label: '轮廓系数 (Silhouette)',    yAxisMin: -1.0, yAxisMax: 1.0, higherIsBetter: true  },
  'Calinski-Harabasz':  { label: 'Calinski-Harabasz',       yAxisMin: 0.0,  higherIsBetter: true  },
  'Davies-Bouldin':     { label: 'Davies-Bouldin',          yAxisMin: 0.0,  higherIsBetter: false },
};

const DEFAULT_METRICS: Record<string, string[]> = {
  classification: ['Accuracy', 'AUC', 'F1', 'Recall', 'Precision'],
  regression: ['R2', 'RMSE', 'MAE'],
  clustering: ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'],
};

export default function StepEvaluate() {
  const trainResult = useAppStore((s) => s.trainResult);
  const taskType = useAppStore((s) => s.taskType);
  const targetColumn = useAppStore((s) => s.targetColumn);
  const setStep = useAppStore((s) => s.setStep);

  const [primaryMetric, setPrimaryMetric] = useState<string>(
    DEFAULT_METRICS[taskType]?.[0] || 'Accuracy'
  );
  const [showShap, setShowShap] = useState(false);

  if (!trainResult) return null;

  const { metrics_table, feature_importance, ai_evaluation, shap_plot, misclassified_samples } = trainResult;

  // BUG FIX: filter out internal columns ('index', 'TT (Sec)') that are not meaningful for display
  const DISPLAY_EXCLUDE = new Set(['index', 'TT (Sec)', 'TT(Sec)']);

  const modelNames = metrics_table.map((row) => String(row['Model'] || ''));

  const availableMetrics = metrics_table.length > 0
    ? Object.keys(metrics_table[0]).filter(k => !DISPLAY_EXCLUDE.has(k) && k !== 'Model' && METRIC_OPTIONS[k])
    : [];

  const getMetricValues = (metric: string) => {
    return metrics_table.map((row) => {
      const v = row[metric];
      return typeof v === 'number' ? v : parseFloat(String(v)) || 0;
    });
  };

  const effectiveMetric = availableMetrics.includes(primaryMetric)
    ? primaryMetric
    : (availableMetrics[0] || 'Accuracy');

  const primaryValues = getMetricValues(effectiveMetric);
  const metricConfig = METRIC_OPTIONS[effectiveMetric] || { yAxisMin: 0, higherIsBetter: true };

  // BUG FIX: determine best model by sorting on the primary metric (respecting direction)
  const bestModelIndex = (() => {
    if (primaryValues.length === 0) return 0;
    if (metricConfig.higherIsBetter) {
      return primaryValues.indexOf(Math.max(...primaryValues));
    } else {
      return primaryValues.indexOf(Math.min(...primaryValues));
    }
  })();
  const bestModelName = modelNames[bestModelIndex] || '-';

  const barChartOption = {
    tooltip: { trigger: 'axis' as const, axisPointer: { type: 'shadow' as const } },
    legend: { data: [effectiveMetric], top: 0 },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: {
      type: 'category' as const,
      data: modelNames,
      axisLabel: { rotate: 30, fontSize: 11 },
    },
    yAxis: {
      type: 'value' as const,
      min: metricConfig.yAxisMin,
      max: metricConfig.yAxisMax,
      splitNumber: 5,
    },
    animationDuration: 1500,
    animationEasing: 'cubicOut' as const,
    series: [
      {
        name: effectiveMetric,
        type: 'bar' as const,
        data: primaryValues.map((v, i) => ({
          value: v,
          itemStyle: {
            color: i === bestModelIndex
              ? new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                  { offset: 0, color: '#10b981' },
                  { offset: 1, color: '#34d399' },
                ])
              : new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                  { offset: 0, color: '#6366f1' },
                  { offset: 1, color: '#818cf8' },
                ]),
            borderRadius: [4, 4, 0, 0],
          },
        })),
      },
    ],
  };

  // BUG FIX: avoid mutating the original arrays with .reverse() — use spread copies
  const featureNames = Object.keys(feature_importance).slice(0, 15);
  const featureValues = featureNames.map((k) => feature_importance[k]);
  const featureNamesReversed = [...featureNames].reverse();
  const featureValuesReversed = [...featureValues].reverse();

  const featureChartOption = {
    tooltip: { trigger: 'axis' as const, axisPointer: { type: 'shadow' as const } },
    grid: { left: '3%', right: '8%', bottom: '3%', containLabel: true },
    xAxis: { type: 'value' as const },
    yAxis: {
      type: 'category' as const,
      data: featureNamesReversed,
      axisLabel: { fontSize: 11 },
    },
    animationDuration: 1500,
    animationEasing: 'cubicOut' as const,
    series: [
      {
        type: 'bar' as const,
        data: featureValuesReversed,
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: '#06b6d4' },
            { offset: 1, color: '#22d3ee' },
          ]),
          borderRadius: [0, 4, 4, 0],
        },
      },
    ],
  };

  const taskTypeLabel = {
    classification: '分类',
    regression: '回归',
    clustering: '聚类',
  }[taskType] || taskType;

  const misclassifiedColumns = misclassified_samples && misclassified_samples.length > 0
    ? Object.keys(misclassified_samples[0])
    : [];

  // BUG FIX: filter out internal/redundant columns from the displayed detail table
  const tableColumns = metrics_table.length > 0
    ? Object.keys(metrics_table[0]).filter(k => !DISPLAY_EXCLUDE.has(k))
    : [];

  return (
    <div className="max-w-7xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">模型评估与智能解读</h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
                <span className="text-indigo-500">📊</span> 模型性能对比
                <span className="text-xs font-normal text-green-600 ml-1">🏆 绿色为最优</span>
              </h3>
              <div className="flex items-center gap-3">
                <label className="text-sm text-gray-600">选择指标:</label>
                <select
                  value={effectiveMetric}
                  onChange={(e) => setPrimaryMetric(e.target.value)}
                  className="px-3 py-1.5 border border-gray-200 rounded-lg text-sm focus:border-indigo-500 outline-none"
                >
                  {availableMetrics.map((m) => (
                    <option key={m} value={m}>
                      {METRIC_OPTIONS[m]?.label || m}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <ReactECharts
              echarts={echarts}
              option={barChartOption}
              style={{ height: 350 }}
              notMerge={true}
              lazyUpdate={true}
            />
          </div>

          {shap_plot && taskType === 'classification' && (
            <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
                  <span className="text-purple-500">🧬</span> SHAP 特征影响分析
                </h3>
                <button
                  onClick={() => setShowShap(!showShap)}
                  className="text-sm text-indigo-600 hover:text-indigo-700"
                >
                  {showShap ? '收起' : '展开'}
                </button>
              </div>
              {showShap && (
                <div className="mt-4">
                  <img
                    src={`data:image/png;base64,${shap_plot}`}
                    alt="SHAP Summary Plot"
                    className="w-full rounded-lg shadow-sm"
                  />
                  <p className="text-xs text-gray-500 mt-2 text-center">
                    SHAP 摘要图展示了每个特征对模型预测的影响程度
                  </p>
                </div>
              )}
            </div>
          )}

          {featureNames.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
              <h3 className="text-lg font-semibold text-gray-700 mb-4 flex items-center gap-2">
                <span className="text-indigo-500">🔍</span> 特征重要性 Top 15
              </h3>
              <ReactECharts
                echarts={echarts}
                option={featureChartOption}
                style={{ height: 400 }}
                notMerge={true}
                lazyUpdate={true}
              />
            </div>
          )}

          {misclassified_samples && misclassified_samples.length > 0 && (
            <div className="bg-white rounded-xl border border-red-200 p-5 shadow-sm">
              <h3 className="text-lg font-semibold text-red-600 mb-4 flex items-center gap-2">
                <span>⚠️</span> 预测错误的样本 (Top 10)
              </h3>
              <p className="text-xs text-gray-500 mb-3">
                以下是模型预测与真实标签不一致的样本，这些案例可以帮助您分析模型的薄弱环节
              </p>
              <div className="overflow-x-auto max-h-80 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-gray-50">
                    <tr>
                      {misclassifiedColumns.map((col) => (
                        <th key={col} className="px-3 py-2 text-left font-medium text-gray-600 whitespace-nowrap">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {misclassified_samples.map((row, i) => (
                      <tr key={i} className="border-t border-gray-100 bg-red-50/30">
                        {misclassifiedColumns.map((col) => {
                          const isWrong = col === 'prediction_label' || col === targetColumn;
                          return (
                            <td key={col} className={`px-3 py-2 whitespace-nowrap ${isWrong ? 'font-medium text-red-600' : 'text-gray-600'}`}>
                              {String(row[col] ?? '')}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-700 mb-4 flex items-center gap-2">
              <span className="text-indigo-500">📋</span> 详细指标表
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-50">
                    {tableColumns.map((key) => (
                      <th key={key} className="px-3 py-2 text-left font-medium text-gray-600 whitespace-nowrap">
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics_table.map((row, i) => (
                    <tr
                      key={i}
                      className={`border-t border-gray-100 ${i === bestModelIndex ? 'bg-green-50 ring-1 ring-inset ring-green-200' : ''}`}
                    >
                      {tableColumns.map((key) => {
                        const val = row[key];
                        return (
                          <td key={key} className={`px-3 py-2 whitespace-nowrap ${key === 'Model' ? 'font-semibold text-gray-800' : 'text-gray-600'}`}>
                            {typeof val === 'number' ? val.toFixed(4) : String(val ?? '')}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl border border-indigo-100 p-5 shadow-sm sticky top-6">
            <h3 className="text-lg font-semibold text-indigo-700 mb-3 flex items-center gap-2">
              <span>🤖</span> AI 深度解读
            </h3>
            <div className="prose prose-sm prose-indigo max-w-none text-gray-700">
              <ReactMarkdown>{ai_evaluation}</ReactMarkdown>
            </div>
          </div>

          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <div className="text-sm text-gray-500 mb-3">任务信息</div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-500">任务类型</span>
                <span className="font-medium text-gray-700">{taskTypeLabel}</span>
              </div>
              {taskType !== 'clustering' && (
                <div className="flex justify-between">
                  <span className="text-gray-500">目标列</span>
                  <span className="font-medium text-gray-700">{targetColumn}</span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-gray-500">参与模型</span>
                <span className="font-medium text-gray-700">{metrics_table.length} 个</span>
              </div>
              {/* BUG FIX: show actual best model by metric, not always first model */}
              <div className="flex justify-between">
                <span className="text-gray-500">最优模型</span>
                <span className="font-medium text-green-600">🏆 {bestModelName}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">评估指标</span>
                <span className="font-medium text-indigo-600">
                  {METRIC_OPTIONS[effectiveMetric]?.label || effectiveMetric}
                </span>
              </div>
              {shap_plot && taskType === 'classification' && (
                <div className="flex justify-between">
                  <span className="text-gray-500">SHAP 分析</span>
                  <span className="font-medium text-green-600">可用</span>
                </div>
              )}
            </div>
          </div>

          <button
            onClick={() => setStep(4)}
            className="w-full py-3 px-6 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700 shadow-md shadow-indigo-200 transition-all"
          >
            导出模型 💾
          </button>
        </div>
      </div>
    </div>
  );
}
