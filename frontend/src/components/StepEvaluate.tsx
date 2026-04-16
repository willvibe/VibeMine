import { useState, useRef } from 'react';
import { useAppStore } from '../store';
import { useI18n } from '../i18n/index';
import { callGemini } from '../api';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts/core';
import { BarChart, RadarChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([BarChart, RadarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

const MODEL_FULL_NAMES: Record<string, string> = {
  'lr': 'Logistic Regression', 'rf': 'Random Forest', 'gbc': 'Gradient Boosting Classifier',
  'et': 'Extra Trees Classifier', 'xgb': 'XGBoost', 'lightgbm': 'LightGBM', 'dt': 'Decision Tree',
  'knn': 'K-Nearest Neighbors', 'ada': 'AdaBoost', 'lda': 'Linear Discriminant Analysis',
  'qda': 'Quadratic Discriminant Analysis', 'nb': 'Naive Bayes', 'svm': 'SVM',
  'ridge': 'Ridge Regression', 'lasso': 'Lasso Regression', 'gbr': 'Gradient Boosting Regressor',
  'kmeans': 'K-Means', 'hclust': 'Hierarchical Clustering', 'meanshift': 'Mean Shift',
  'dbscan': 'DBSCAN', 'affinity': 'Affinity Propagation',
};

const METRIC_LABELS: Record<string, string> = {
  Accuracy: 'accuracy', AUC: 'auc', Recall: 'recall', Precision: 'precision', F1: 'f1',
  R2: 'r2', RMSE: 'rmse', MAE: 'mae', 'Silhouette': 'silhouette',
  'Calinski-Harabasz': 'calinskiHarabasz', 'Davies-Bouldin': 'daviesBouldin',
};

const HIGHER_IS_BETTER_METRICS = new Set(['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'R2', 'Silhouette', 'Calinski-Harabasz']);

const DISPLAY_EXCLUDE = new Set(['index', 'TT (Sec)', 'TT(Sec)', 'MS', 'prediction_label', 'prediction_score']);

const GRADIENT_COLORS = [
  ['#4f46e5', '#818cf8'],
  ['#10b981', '#34d399'],
  ['#f59e0b', '#fbbf24'],
  ['#ef4444', '#f87171'],
];

const RADAR_COLORS = ['#4f46e5', '#7c3aed', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

function getFullModelName(name: string): string {
  const upper = name.toUpperCase();
  const found = Object.keys(MODEL_FULL_NAMES).find(k => upper.includes(k.toUpperCase()) || MODEL_FULL_NAMES[k].toUpperCase().includes(upper));
  if (found) return MODEL_FULL_NAMES[found];
  return MODEL_FULL_NAMES[name] || name;
}

function fmt(v: unknown): string {
  if (v === null || v === undefined || v === '') return '-';
  const n = typeof v === 'number' ? v : parseFloat(String(v));
  if (isNaN(n)) return String(v);
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(3);
}

function downloadChart(chartRef: React.RefObject<any>, filename: string) {
  const instance = chartRef?.current?.getEchartsInstance();
  if (!instance) return;
  const url = instance.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' });
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
}

function downloadCSV(rows: Record<string, unknown>[], cols: string[], filename: string) {
  if (!rows || rows.length === 0) return;
  const header = cols.map(c => METRIC_LABELS[c] || c).join(',');
  const dataRows = rows.map(row =>
    cols.map(col => {
      const val = row[col];
      if (val === null || val === undefined) return '';
      const str = String(val);
      return str.includes(',') ? `"${str}"` : str;
    }).join(',')
  );
  const csv = [header, ...dataRows].join('\n');
  const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function parseMarkdown(text: string) {
  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
  return escaped
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n\n/g, '<br/><br/>')
    .replace(/\n- /g, '<br/>• ')
    .replace(/\n## /g, '<br/><strong>')
    .replace(/:\*\*/g, ':</strong>')
    .replace(/`([^`]+)`/g, '<code style="background:#f3f4f6;padding:1px 4px;border-radius:4px;font-size:0.85em">$1</code>');
}

export default function StepEvaluate() {
  const { t } = useI18n();
  const trainResult = useAppStore((s) => s.trainResult);
  const taskType = useAppStore((s) => s.taskType);
  const targetColumn = useAppStore((s) => s.targetColumn);
  const aiEvaluation = useAppStore((s) => s.aiEvaluation);
  const misclassifiedAnalysis = useAppStore((s) => s.misclassifiedAnalysis);
  const setStep = useAppStore((s) => s.setStep);
  const setAiEvaluation = useAppStore((s) => s.setAiEvaluation);
  const setMisclassifiedAnalysis = useAppStore((s) => s.setMisclassifiedAnalysis);

  const defaultMetric = taskType === 'regression' ? 'R2' : taskType === 'clustering' ? 'Silhouette' : 'Accuracy';
  const [primaryMetrics, setPrimaryMetrics] = useState<string[]>([defaultMetric]);
  const [showShap, setShowShap] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);

  const barChartRef = useRef<any>(null);
  const radarChartRef = useRef<any>(null);
  const featChartRef = useRef<any>(null);

  if (!trainResult) return null;

  const { metrics_table, feature_importance, shap_plot, misclassified_samples } = trainResult;

  const allMetrics = metrics_table.length > 0
    ? Object.keys(metrics_table[0]).filter(k => !DISPLAY_EXCLUDE.has(k) && k !== 'Model' && METRIC_LABELS[k])
    : [];

  const tableColumns = metrics_table.length > 0
    ? Object.keys(metrics_table[0]).filter(k => !DISPLAY_EXCLUDE.has(k))
    : [];

  const firstMetric = primaryMetrics[0] || allMetrics[0];
  const firstCfg = HIGHER_IS_BETTER_METRICS.has(firstMetric) ? { higherIsBetter: true } : { higherIsBetter: false };

  let bestIdx = 0;
  if (metrics_table.length > 0) {
    const vals = metrics_table.map(r => {
      const v = parseFloat(String(r[firstMetric] ?? 'NaN'));
      return isNaN(v) ? null : v;
    }).filter((v): v is number => v !== null);
    if (vals.length > 0) {
      bestIdx = firstCfg.higherIsBetter
        ? metrics_table.findIndex(r => parseFloat(String(r[firstMetric] ?? 'NaN')) === Math.max(...vals))
        : metrics_table.findIndex(r => parseFloat(String(r[firstMetric] ?? 'NaN')) === Math.min(...vals));
    }
  }

  const primaryValues = (metric: string) => metrics_table.map(row => {
    const v = row[metric];
    if (v === undefined || v === null) return null;
    const parsed = parseFloat(String(v));
    return isNaN(parsed) ? null : parsed;
  });

  const bestModelName = String(metrics_table[bestIdx]?.Model || '');

  const needsDualAxis = primaryMetrics.length >= 2 && (() => {
    const higher = primaryMetrics.filter(m => HIGHER_IS_BETTER_METRICS.has(m));
    const lower = primaryMetrics.filter(m => !HIGHER_IS_BETTER_METRICS.has(m));
    return higher.length > 0 && lower.length > 0;
  })();

  const barSeries = primaryMetrics.map((metric, seriesIdx) => {
    const vals = primaryValues(metric);
    const seriesYAxisIndex = needsDualAxis ? (HIGHER_IS_BETTER_METRICS.has(metric) ? 0 : 1) : 0;
    const [c0, c1] = GRADIENT_COLORS[seriesIdx % GRADIENT_COLORS.length];

    return {
      name: METRIC_LABELS[metric] || metric,
      type: 'bar' as const,
      yAxisIndex: seriesYAxisIndex,
      data: vals.map((v, i) => ({
        value: v,
        label: {
          show: v !== null && v !== undefined,
          position: 'insideBottom' as const,
          offset: [0, -4],
          formatter: (params: any) => {
            const val = params.value;
            if (val === null || val === undefined) return '';
            return typeof val === 'number' ? val.toFixed(3) : String(val);
          },
          fontSize: 9,
          color: '#fff',
        },
        itemStyle: {
          color: i === bestIdx
            ? { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: c0 }, { offset: 1, color: c1 }] }
            : { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: `${c0}66` }, { offset: 1, color: `${c1}66` }] },
          borderRadius: [4, 4, 0, 0],
        },
      })),
      barMaxWidth: 52,
      animationDuration: 1200,
      animationEasing: 'cubicOut' as const,
    };
  });

  const barYAxes = (() => {
    if (!needsDualAxis) {
      const vals = primaryValues(firstMetric).filter((v): v is number => v !== null);
      const lo = vals.length > 0 ? Math.min(...vals) : 0;
      const hi = vals.length > 0 ? Math.max(...vals) : 1;
      const pad = hi > lo ? (hi - lo) * 0.2 : 0.1;
      const isHigher = HIGHER_IS_BETTER_METRICS.has(firstMetric);
      return [{
        type: 'value' as const,
        min: isHigher ? Math.max(0, lo - pad) : 0,
        max: isHigher ? hi + pad : hi * 1.2,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: '#9ca3af', fontSize: 9, fontFamily: 'ui-sans-serif', formatter: (v: number) => v.toFixed(2) },
        splitLine: { lineStyle: { color: '#f3f4f6', type: 'dashed' } },
      }];
    }

    const axes = [];
    const leftMetrics = primaryMetrics.filter(m => HIGHER_IS_BETTER_METRICS.has(m));
    const rightMetrics = primaryMetrics.filter(m => !HIGHER_IS_BETTER_METRICS.has(m));

    if (leftMetrics.length > 0) {
      const vals = leftMetrics.flatMap(m => primaryValues(m).filter((v): v is number => v !== null));
      const lo = vals.length > 0 ? Math.min(...vals) : 0;
      const hi = vals.length > 0 ? Math.max(...vals) : 1;
      const pad = hi > lo ? (hi - lo) * 0.2 : 0.1;
      axes.push({
        type: 'value' as const,
        name: leftMetrics.map(m => METRIC_LABELS[m] || m).join('+'),
        nameTextStyle: { color: '#6366f1', fontSize: 9, width: 80, overflow: 'truncate' },
        min: Math.max(0, lo - pad),
        max: hi + pad,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: '#9ca3af', fontSize: 9, fontFamily: 'ui-sans-serif', formatter: (v: number) => v.toFixed(2) },
        splitLine: { lineStyle: { color: '#f3f4f6', type: 'dashed' } },
        position: 'left' as const,
      });
    }

    if (rightMetrics.length > 0) {
      const vals = rightMetrics.flatMap(m => primaryValues(m).filter((v): v is number => v !== null));
      const hi = vals.length > 0 ? Math.max(...vals) : 1;
      const pad = hi > 0 ? hi * 0.2 : 0.1;
      axes.push({
        type: 'value' as const,
        name: rightMetrics.map(m => METRIC_LABELS[m] || m).join('+'),
        nameTextStyle: { color: '#10b981', fontSize: 9, width: 80, overflow: 'truncate' },
        min: 0,
        max: hi + pad,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: '#9ca3af', fontSize: 9, fontFamily: 'ui-sans-serif', formatter: (v: number) => v.toFixed(2) },
        splitLine: { show: false },
        position: 'right' as const,
      });
    }

    return axes;
  })();

  const barOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      axisPointer: { type: 'shadow' },
      backgroundColor: 'rgba(255,255,255,0.96)',
      borderColor: 'rgba(0,0,0,0.08)',
      borderRadius: 12,
      padding: [10, 14],
      textStyle: { color: '#1d1d1f', fontSize: 12, fontFamily: 'ui-sans-serif, system-ui' },
      formatter: (params: any[]) => {
        const idx = params[0]?.dataIndex ?? 0;
        const modelName = getFullModelName(String(metrics_table[idx]?.Model || ''));
        const lines = params.map((p: any) => {
          const color = p.color?.colorStops ? p.color.colorStops[0].color : p.color;
          return `<div style="color:${color};font-size:12px">${p.seriesName}: <b>${fmt(p.value)}</b></div>`;
        }).join('');
        return `<div style="font-weight:600;margin-bottom:4px">${modelName}</div>${lines}`;
      },
    },
    legend: {
      data: primaryMetrics.map((m, i) => ({
        name: METRIC_LABELS[m] || m,
        itemStyle: { color: GRADIENT_COLORS[i % GRADIENT_COLORS.length][0],
          borderRadius: [2, 2, 0, 0], width: 12, height: 12 },
      })),
      bottom: 0,
      icon: 'roundRect',
      itemWidth: 14,
      itemHeight: 14,
      textStyle: { color: '#6b7280', fontSize: 11, fontFamily: 'ui-sans-serif', padding: [0, 4, 0, 0] },
    },
    grid: { top: 12, right: needsDualAxis ? 70 : 24, bottom: 52, left: 48, containLabel: true },
    xAxis: {
      type: 'category' as const,
      data: metrics_table.map((r, i) => i === bestIdx ? `🏆 ${getFullModelName(String(r.Model || ''))}` : getFullModelName(String(r.Model || ''))),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: '#6b7280', fontSize: 9, rotate: 30, fontFamily: 'ui-sans-serif',
        interval: 0,
        formatter: (v: string) => {
          const name = v.replace('🏆 ', '');
          return name.length > 12 ? name.substring(0, 10) + '..' : name;
        },
      },
      splitLine: { show: false },
    },
    yAxis: barYAxes,
    series: barSeries,
  };

  const featNames = Object.keys(feature_importance).slice(0, 15);
  const featVals = featNames.map(k => feature_importance[k]);
  const featColors = featVals.map((_, i) => `rgba(99,102,241,${Math.max(0.3, 0.9 - i * 0.04)})`);

  const featOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      axisPointer: { type: 'shadow' },
      backgroundColor: 'rgba(255,255,255,0.96)',
      borderColor: 'rgba(0,0,0,0.08)',
      borderRadius: 12,
      padding: [10, 14],
      textStyle: { color: '#1d1d1f', fontSize: 12, fontFamily: 'ui-sans-serif, system-ui' },
    },
    grid: { top: 16, right: 80, bottom: 24, left: 140, containLabel: false },
    xAxis: {
      type: 'value' as const,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { show: false },
      splitLine: { lineStyle: { color: '#f3f4f6', type: 'dashed' } },
    },
    yAxis: {
      type: 'category' as const,
      data: [...featNames].reverse(),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: '#6b7280', fontSize: 10, fontFamily: 'ui-sans-serif' },
    },
    series: [{
      type: 'bar' as const,
      data: [...featVals].reverse(),
      itemStyle: { color: (params: any) => featColors[params.dataIndex], borderRadius: [0, 4, 4, 0] },
      barMaxWidth: 12,
      animationDuration: 1200,
      animationEasing: 'cubicOut' as const,
    }],
  };

  const radarMetrics = tableColumns.filter(c => c !== 'Model').slice(0, 6);

  const radarIndicator = radarMetrics.map(m => {
    const vals = metrics_table.slice(0, 8).map(row => {
      const raw = row[m];
      if (raw === null || raw === undefined) return null;
      const n = parseFloat(String(raw));
      return isNaN(n) ? null : n;
    }).filter((v): v is number => v !== null);
    if (vals.length === 0) return { name: METRIC_LABELS[m] || m, max: 1, min: 0 };
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const pad = (hi - lo) * 0.15 || 0.05;
    return {
      name: METRIC_LABELS[m] || m,
      max: hi + pad,
      min: Math.max(0, lo - pad),
    };
  });

  const radarSeries = metrics_table.slice(0, 8).map((row, idx) => {
    const values = radarMetrics.map(metric => {
      const v = row[metric];
      if (v === undefined || v === null) return null;
      const parsed = parseFloat(String(v));
      return isNaN(parsed) ? null : parsed;
    });
    return {
      name: String(row.Model || ''),
      value: values,
      lineStyle: { color: RADAR_COLORS[idx % RADAR_COLORS.length], width: 2 },
      itemStyle: { color: RADAR_COLORS[idx % RADAR_COLORS.length] },
      areaStyle: { color: RADAR_COLORS[idx % RADAR_COLORS.length] + '33' },
    };
  });

  const radarOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item' as const,
      backgroundColor: 'rgba(255,255,255,0.96)',
      borderColor: 'rgba(0,0,0,0.08)',
      borderRadius: 12,
      padding: [10, 14],
      textStyle: { color: '#1d1d1f', fontSize: 12, fontFamily: 'ui-sans-serif, system-ui' },
    },
    legend: {
      data: radarSeries.map(s => s.name),
      bottom: 0,
      icon: 'circle',
      itemWidth: 8,
      itemHeight: 8,
      textStyle: { color: '#6b7280', fontSize: 10, fontFamily: 'ui-sans-serif' },
    },
    radar: {
      indicator: radarIndicator,
      shape: 'polygon' as const,
      splitNumber: 5,
      radius: '65%',
      axisName: { color: '#6b7280', fontSize: 10, fontFamily: 'ui-sans-serif' },
      splitLine: { lineStyle: { color: '#e5e7eb' } },
      splitArea: { show: false },
      axisLine: { lineStyle: { color: '#e5e7eb' } },
    },
    series: [{ type: 'radar' as const, data: radarSeries, animationDuration: 1200, animationEasing: 'cubicOut' as const }],
  };

  const taskLabel = taskType === 'classification' ? t('classification') : taskType === 'regression' ? t('regression') : t('clustering');
  const misclassifiedCols = misclassified_samples && misclassified_samples.length > 0
    ? Object.keys(misclassified_samples[0]).filter(c => !DISPLAY_EXCLUDE.has(c))
    : [];

  const toggleMetric = (metric: string) => {
    setPrimaryMetrics(prev => {
      if (prev.includes(metric)) {
        if (prev.length === 1) return prev;
        return prev.filter(m => m !== metric);
      }
      return [...prev, metric];
    });
  };

  const handleAiEvaluation = async () => {
    if (!trainResult || trainResult.metrics_table.length === 0) {
      setAiEvaluation(t('aiEvalUnavailable'));
      return;
    }
    setAiLoading(true);
    try {
      const mt = trainResult.metrics_table;
      const bestIdx = mt.findIndex((r: any) => {
        const mk = taskType === 'regression' ? 'R2' : taskType === 'clustering' ? 'Silhouette' : 'Accuracy';
        const val = parseFloat(r[mk]);
        return !isNaN(val) && val === Math.max(...mt.map((row: any) => parseFloat(row[mk]) || 0));
      });
      const bestModel = bestIdx >= 0 ? String(mt[bestIdx].Model) : '';
      const table = JSON.stringify(mt, null, 2);
      const prompt = `你是一个资深的数据科学家。用户完成了机器学习训练，以下是所有模型的指标对比：

任务类型：${taskType}
目标列：${targetColumn || '无'}
模型指标数据：
${table}

最佳模型是：${bestModel}

请分析：
1. 最佳模型的表现如何
2. 各模型之间的差异
3. 有什么优化建议
请使用 Markdown 格式，控制在 500 字以内。`;
      const result = await callGemini(prompt);
      setAiEvaluation(result);
    } catch (err) {
      setAiEvaluation(t('aiEvalFailed', { msg: String(err) }));
    } finally {
      setAiLoading(false);
    }
  };

  const handleAiMisclassified = async () => {
    if (!trainResult) {
      setMisclassifiedAnalysis(t('aiMisclassUnavailable'));
      return;
    }
    setAiLoading(true);
    try {
      const samples = (trainResult as any).misclassified_samples || [];
      const featImp = trainResult.feature_importance || {};
      const topFeatures = Object.entries(featImp).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([n, v]) => `${n} (${v})`);
      const prompt = `你是一个资深的数据科学家。以下是分类模型中被错误分类的样本（共 ${samples.length} 个）：

目标列：${targetColumn}
最重要特征：${topFeatures.join(', ')}
错误样本：
${JSON.stringify(samples.slice(0, 20), null, 2)}

请分析：
1. 错误样本的共同特征
2. 可能的原因
3. 改进建议
请使用 Markdown 格式，控制在 400 字以内。`;
      const result = await callGemini(prompt);
      setMisclassifiedAnalysis(result);
    } catch (err) {
      setMisclassifiedAnalysis(t('aiMisclassFailed', { msg: String(err) }));
    } finally {
      setAiLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-semibold tracking-tight text-gray-900">{t('modelEvaluation')}</h2>
        <p className="text-sm text-gray-500 mt-1">{t('performanceDesc')}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="glass-card rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-semibold text-gray-900">{t('performanceComparison')}</h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  {needsDualAxis
                    ? t('metricsHintDual')
                    : t('metricsHint')}
                </p>
              </div>
              <div className="flex items-center gap-3">
                <div className="flex flex-wrap gap-1 max-w-[220px] justify-end">
                  {allMetrics.map(m => {
                    const isActive = primaryMetrics.includes(m);
                    const isHigher = HIGHER_IS_BETTER_METRICS.has(m);
                    return (
                      <button
                        key={m}
                        onClick={() => toggleMetric(m)}
                        className={`px-2 py-1 rounded-md text-xs font-medium transition-all ${
                          isActive
                            ? isHigher
                              ? 'bg-indigo-100 text-indigo-700 border border-indigo-300'
                              : 'bg-emerald-100 text-emerald-700 border border-emerald-300'
                            : 'bg-gray-50 text-gray-400 border border-gray-200 hover:border-gray-300'
                        }`}
                      >
                        {METRIC_LABELS[m] || m}
                      </button>
                    );
                  })}
                </div>
                <button
                  onClick={() => downloadChart(barChartRef, 'vibemine_bar_chart.png')}
                  title={t('downloadBarChart')}
                  className="p-2 rounded-lg text-gray-400 hover:text-indigo-600 hover:bg-indigo-50 transition-all shrink-0"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M16 12l-4 4-4-4M12 16V4" />
                  </svg>
                </button>
              </div>
            </div>
            <ReactECharts ref={barChartRef} echarts={echarts} option={barOption} style={{ height: 320, width: '100%' }} notMerge={true} lazyUpdate={true} />
          </div>

          {shap_plot && taskType === 'classification' && (
            <div className="glass-card rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-base font-semibold text-gray-900">{t('shapFeatureImpact')}</h3>
                <button onClick={() => setShowShap(!showShap)} className="text-xs text-indigo-500 hover:text-indigo-600 font-medium transition-colors">
                    {showShap ? t('shapHide') : t('shapShow')}
                </button>
              </div>
              {showShap && <img src={`data:image/png;base64,${shap_plot}`} alt="SHAP" className="w-full rounded-xl" />}
            </div>
          )}

          {featNames.length > 0 && (
            <div className="glass-card rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-base font-semibold text-gray-900">{t('featureImportanceTop')}</h3>
                <button
                  onClick={() => downloadChart(featChartRef, 'vibemine_feature_importance.png')}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-500 hover:text-indigo-600 hover:bg-indigo-50 transition-all"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M16 12l-4 4-4-4M12 16V4" />
                  </svg>
                  {t('downloadPng')}
                </button>
              </div>
              <ReactECharts ref={featChartRef} echarts={echarts} option={featOption} style={{ height: 360 }} notMerge={true} lazyUpdate={true} />
            </div>
          )}

          {taskType === 'classification' && misclassified_samples && misclassified_samples.length > 0 && (
            <div className="glass-card rounded-2xl p-6">
              <div className="flex items-center justify-between mb-1">
                <div>
                  <h3 className="text-base font-semibold text-gray-900">{t('misclassifiedSamples')}</h3>
                  <p className="text-xs text-gray-400 mt-0.5">{t('misclassifiedCount', { count: misclassified_samples.length })}</p>
                </div>
                <button
                  onClick={() => downloadCSV(misclassified_samples, misclassifiedCols, 'vibemine_misclassified.csv')}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-500 hover:text-indigo-600 hover:bg-indigo-50 transition-all"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M16 12l-4 4-4-4M12 16V4" />
                  </svg>
                  {t('downloadCsv')}
                </button>
              </div>
              <div className="overflow-x-auto max-h-64 overflow-y-auto rounded-xl border border-gray-100 mb-4">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-gray-50/80">
                    <tr>
                      {misclassifiedCols.map(col => (
                        <th key={col} className="px-3 py-2 text-left font-medium text-gray-500 whitespace-nowrap">
                          {col === targetColumn ? `${col} (${t('trueLabel')})` : col === 'prediction_label' ? `${col} (${t('predictLabel')})` : col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {misclassified_samples.map((row, i) => (
                      <tr key={i} className="border-t border-gray-50">
                        {misclassifiedCols.map(col => {
                          const isTrue = col === targetColumn;
                          const isPred = col === 'prediction_label';
                          return (
                            <td key={col} className={`px-3 py-2 whitespace-nowrap ${isTrue ? 'font-semibold text-red-500' : isPred ? 'font-semibold text-orange-500' : 'text-gray-600'}`}>
                              {fmt(row[col])}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {misclassifiedAnalysis ? (
                <div className="bg-gray-50/60 rounded-xl p-4 border border-gray-100">
                  <p className="text-sm text-gray-700 leading-relaxed" dangerouslySetInnerHTML={{ __html: parseMarkdown(misclassifiedAnalysis) }} />
                </div>
              ) : (
                <button
                  onClick={handleAiMisclassified}
                  disabled={aiLoading}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-xs font-medium shadow-md hover:shadow-lg hover:shadow-indigo-500/20 transition-all disabled:opacity-60"
                >
                  {aiLoading ? (
                    <>
                      <svg className="animate-spin w-3.5 h-3.5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                      </svg>
                      {t('aiAnalyzing')}
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      {t('aiMisclassified')}
                    </>
                  )}
                </button>
              )}
            </div>
          )}

          {taskType === 'classification' && (!misclassified_samples || misclassified_samples.length === 0) && (
            <div className="glass-card rounded-2xl p-6">
              <h3 className="text-base font-semibold text-gray-900 mb-2">{t('misclassifiedSamples')}</h3>
              <p className="text-sm text-gray-400 mb-4">{t('noMisclassified')}</p>
              {misclassifiedAnalysis ? (
                <div className="bg-gray-50/60 rounded-xl p-4 border border-gray-100">
                  <p className="text-sm text-gray-700 leading-relaxed" dangerouslySetInnerHTML={{ __html: parseMarkdown(misclassifiedAnalysis) }} />
                </div>
              ) : (
                <button
                  onClick={handleAiMisclassified}
                  disabled={aiLoading}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-xs font-medium shadow-md hover:shadow-lg hover:shadow-indigo-500/20 transition-all disabled:opacity-60"
                >
                  {aiLoading ? (
                    <>
                      <svg className="animate-spin w-3.5 h-3.5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                      </svg>
                      {t('aiAnalyzing')}
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      {t('aiMisclassified')}
                    </>
                  )}
                </button>
              )}
            </div>
          )}

          <div className="glass-card rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-semibold text-gray-900">{t('detailedMetrics')}</h3>
              <button
                onClick={() => downloadCSV(metrics_table, tableColumns, 'vibemine_metrics.csv')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-500 hover:text-indigo-600 hover:bg-indigo-50 transition-all"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M16 12l-4 4-4-4M12 16V4" />
                </svg>
                {t('downloadCsv')}
                </button>
              </div>
            <div className="overflow-x-auto rounded-xl border border-gray-100 mb-6">
              <table className="w-full text-xs">
                <thead className="bg-gray-50/80">
                  <tr>
                    {tableColumns.map(key => (
                      <th key={key} className="px-3 py-2 text-left font-medium text-gray-500 whitespace-nowrap">
                        {key === 'Model' ? t('model') : (t(METRIC_LABELS[key] as any) || METRIC_LABELS[key] || key)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics_table.map((row, i) => (
                    <tr key={i} className={`border-t border-gray-50 transition-colors ${i === bestIdx ? 'bg-indigo-50/50' : ''}`}>
                      {tableColumns.map(key => (
                        <td key={key} className={`px-3 py-2 whitespace-nowrap ${key === 'Model' ? 'font-semibold text-gray-800' : 'text-gray-600'}`}>
                          {key === 'Model' ? getFullModelName(String(row[key] ?? '')) : fmt(row[key])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {radarMetrics.length > 2 && (
              <>
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-medium text-gray-700">{t('radarChart')}</h4>
                  <button
                    onClick={() => downloadChart(radarChartRef, 'vibemine_radar_chart.png')}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-500 hover:text-indigo-600 hover:bg-indigo-50 transition-all"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M16 12l-4 4-4-4M12 16V4" />
                    </svg>
                    {t('downloadPng')}
                  </button>
                </div>
                <ReactECharts ref={radarChartRef} echarts={echarts} option={radarOption} style={{ height: 340 }} notMerge={true} lazyUpdate={true} />
              </>
            )}
          </div>
        </div>

        <div className="space-y-6">
          <div className="glass-card rounded-2xl p-6 sticky top-24">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-semibold text-gray-900">{t('aiInterpretation')}</h3>
              {!aiEvaluation && (
                <button
                  onClick={handleAiEvaluation}
                  disabled={aiLoading}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-xs font-medium shadow-md hover:shadow-lg hover:shadow-indigo-500/20 transition-all disabled:opacity-60"
                >
                  {aiLoading ? (
                    <>
                      <svg className="animate-spin w-3 h-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                      </svg>
                      {t('generating')}
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      {t('aiGenerateBtn')}
                    </>
                  )}
                </button>
              )}
            </div>
            {aiEvaluation ? (
              <div className="text-sm text-gray-600 leading-relaxed">
                <p className="whitespace-pre-wrap">{aiEvaluation}</p>
              </div>
            ) : (
              <p className="text-xs text-gray-400 italic">{t('aiHintText')}</p>
            )}
          </div>

          <div className="glass-card rounded-2xl p-6">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">{t('trainingSummary')}</h3>
            <div className="space-y-3">
              {[
                { label: t('taskType'), value: taskLabel },
                taskType !== 'clustering' && { label: t('targetColumn'), value: targetColumn },
                { label: t('modelCount'), value: t('countUnit', { count: metrics_table.length }) },
                { label: t('bestModel'), value: getFullModelName(bestModelName), highlight: true },
                { label: t('comparisonMetrics'), value: primaryMetrics.map(m => t(METRIC_LABELS[m] as any) || METRIC_LABELS[m] || m).join(', ') },
                  misclassified_samples && misclassified_samples.length > 0 && { label: t('misclassified'), value: t('countItems', { count: misclassified_samples.length }) },
              ].filter(Boolean).map((item: any) => (
                <div key={item.label} className="flex justify-between items-center py-1.5 border-b border-gray-100 last:border-0">
                  <span className="text-xs text-gray-500">{item.label}</span>
                  <span className={`text-xs font-medium text-right max-w-[60%] ${item.highlight ? 'text-indigo-600' : 'text-gray-700'}`}>{item.value}</span>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={() => setStep(4)}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-sm font-semibold shadow-lg shadow-indigo-500/20 hover:shadow-xl hover:shadow-indigo-500/30 transition-all duration-300"
          >
            {t('exportModel')}
          </button>
        </div>
      </div>
    </div>
  );
}
