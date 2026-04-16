import { useState } from 'react';
import { useAppStore } from '../store';
import { useI18n } from '../i18n/index';
import { downloadModel } from '../api';

export default function StepExport() {
  const { t } = useI18n();
  const trainResult = useAppStore((s) => s.trainResult);
  const taskType = useAppStore((s) => s.taskType);
  const targetColumn = useAppStore((s) => s.targetColumn);
  const reset = useAppStore((s) => s.reset);
  const filename = useAppStore((s) => s.filename);
  const selectedModels = useAppStore((s) => s.selectedModels);
  const ignoreColumns = useAppStore((s) => s.ignoreColumns);
  const useSmote = useAppStore((s) => s.useSmote);
  const dataInfo = useAppStore((s) => s.dataInfo);
  const dataDescribe = useAppStore((s) => s.dataDescribe);
  const totalRows = useAppStore((s) => s.totalRows);
  const shape = useAppStore((s) => s.shape);
  const aiInsight = useAppStore((s) => s.aiInsight);
  const aiEvaluation = useAppStore((s) => s.aiEvaluation);
  const misclassifiedAnalysis = useAppStore((s) => s.misclassifiedAnalysis);

  if (!trainResult) return null;

  const MODEL_NAMES_MAP: Record<string, string> = {
    lr: taskType === 'regression' ? 'Linear Regression' : 'Logistic Regression',
    ridge: 'Ridge Regression',
    lasso: 'Lasso Regression',
    en: 'ElasticNet',
    rf: 'Random Forest',
    et: 'Extra Trees',
    gbc: 'Gradient Boosting Classifier',
    gbr: 'Gradient Boosting Regressor',
    xgb: 'XGBoost',
    lightgbm: 'LightGBM',
    dt: 'Decision Tree',
    knn: 'K-Nearest Neighbors',
    ada: 'AdaBoost',
    lda: 'Linear Discriminant Analysis',
    nb: 'Naive Bayes',
    qda: 'Quadratic Discriminant Analysis',
    kmeans: 'K-Means',
    hclust: 'Hierarchical Clustering',
    dbscan: 'DBSCAN',
    meanshift: 'Mean Shift',
    affinity: 'Affinity Propagation',
  };

  const getFullModelName = (key: string): string => MODEL_NAMES_MAP[key] || key;

  const TASK_NAMES: Record<string, string> = {
    classification: t('classification'),
    regression: t('regression'),
    clustering: t('clustering'),
  };

  const generateReport = (): string => {
    const metrics = trainResult.metrics_table || [];
    const featImportance = trainResult.feature_importance || {};
    const modelScores = (trainResult as any).model_scores || {};
    const completedModels = (trainResult as any).completed_models || [];

    let report = `# VibeMine 训练报告\n\n`;
    report += `**生成时间**: ${new Date().toLocaleString('zh-CN')}\n`;
    report += `**数据集**: ${filename}\n`;
    report += `**数据维度**: ${shape[0]} 行 × ${shape[1]} 列\n\n`;

    report += `---\n\n## 一、数据概览\n\n`;
    report += `| 属性 | 值 |\n|------|----|\n`;
    report += `| 文件名 | ${filename} |\n`;
    report += `| 总行数 | ${totalRows} |\n`;
    report += `| 总列数 | ${shape[1]} |\n`;
    report += `| 目标列 | ${taskType === 'clustering' ? '无（聚类任务）' : targetColumn} |\n`;
    report += `| 任务类型 | ${TASK_NAMES[taskType] || taskType} |\n\n`;

    if (aiInsight) {
      report += `### 数据 AI 洞察\n\n${aiInsight}\n\n`;
    }

    report += `---\n\n## 二、数据配置\n\n`;
    report += `### 2.1 模型选择\n\n`;
    report += `用户选择参与训练的模型：\n\n`;
    selectedModels.forEach(m => { report += `- ${getFullModelName(m)} (${m})\n`; });
    report += `\n`;

    report += `### 2.2 忽略列\n\n`;
    if (ignoreColumns.length > 0) { ignoreColumns.forEach(c => { report += `- ${c}\n`; }); }
    else { report += `无\n`; }
    report += `\n`;

    report += `### 2.3 高级选项\n\n`;
    report += `| 选项 | 状态 |\n|------|------|\n`;
    report += `| 异常值处理 | ✅ 启用 |\n`;
    report += `| 缺失值填充 | ✅ 启用 |\n`;
    report += `| 分层交叉验证 | ✅ 启用 |\n`;
    report += `| 模型调优 | ✅ 启用 |\n`;
    report += `| 模型集成 | ✅ 启用 |\n`;
    report += `| SMOTE | ${useSmote ? '✅ 启用' : '❌ 未启用'} |\n\n`;

    report += `---\n\n## 三、数据预处理结果\n\n`;
    report += `### 3.1 列信息\n\n`;
    report += `| 列名 | 类型 | 非空值 | 缺失值 | 缺失率 | 唯一值 |\n`;
    report += `|------|------|--------|--------|--------|--------|\n`;
    dataInfo.forEach((info: any) => {
      report += `| ${info.column} | ${info.dtype} | ${info.non_null} | ${info.null} | ${(info.null / (info.non_null + info.null) * 100).toFixed(1)}% | ${info.unique} |\n`;
    });
    report += `\n`;

    report += `### 3.2 数据统计\n\n`;
    report += `| 统计项 | ${Object.keys(dataDescribe).slice(0, 5).join(' | ')} |\n`;
    report += `|--------|${Object.keys(dataDescribe).slice(0, 5).map(() => '--------|').join('')}\n`;
    ['count', 'mean', 'std', 'min', 'max'].forEach(key => {
      const vals = Object.keys(dataDescribe).slice(0, 5).map(k => { const v = (dataDescribe as any)[k]?.[key]; return v !== undefined ? v : '-'; });
      report += `| ${key} | ${vals.join(' | ')} |\n`;
    });
    report += `\n`;

    report += `---\n\n## 四、模型训练结果\n\n`;
    report += `### 4.1 训练概览\n\n`;
    report += `| 指标 | 值 |\n|------|----|\n`;
    report += `| 完成模型数 | ${completedModels.length} |\n`;
    const bestScore = (trainResult as any).best_score;
    report += `| 最佳模型得分 | ${bestScore !== null && bestScore !== undefined ? bestScore.toFixed(4) : '-'} |\n`;
    report += `| 最佳模型 | ${completedModels.length > 0 ? completedModels[0] : '-'} |\n\n`;

    if (metrics.length > 0) {
      report += `### 4.2 模型指标对比\n\n`;
      const metricCols = Object.keys(metrics[0]).filter(k => k !== 'Model');
      report += `| 模型 | ${metricCols.join(' | ')} |\n`;
      report += `|${'-'.repeat(10)}|${metricCols.map(() => '-'.repeat(10)).join('|')}|\n`;
      metrics.forEach(row => {
        const vals = metricCols.map(k => { const v = (row as any)[k]; return v !== null && v !== undefined ? v : '-'; });
        report += `| ${row.Model || '-'} | ${vals.join(' | ')} |\n`;
      });
      report += `\n`;
    }

    if (Object.keys(modelScores).length > 0) {
      report += `### 4.3 所有模型得分\n\n`;
      Object.entries(modelScores).forEach(([name, score]) => { report += `- **${name}**: ${score}\n`; });
      report += `\n`;
    }

    if (Object.keys(featImportance).length > 0) {
      report += `### 4.4 特征重要性 Top 10\n\n`;
      report += `| 特征 | 重要性 |\n|------|--------|\n`;
      Object.entries(featImportance).sort((a, b) => (b[1] as number) - (a[1] as number)).slice(0, 10).forEach(([name, value]) => {
        report += `| ${name} | ${value} |\n`;
      });
      report += `\n`;
    }

    if (aiEvaluation) { report += `---\n\n## 五、AI 模型评估\n\n${aiEvaluation}\n\n`; }
    if (misclassifiedAnalysis) { report += `## 六、错误样本分析\n\n${misclassifiedAnalysis}\n\n`; }

    report += `---\n\n*报告由 VibeMine 自动生成*`;
    return report;
  };

  const [showReport, setShowReport] = useState(false);

  const handleDownload = async () => {
    try {
      const blob = await downloadModel(trainResult.model_id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vibemine_model_${trainResult.model_id}.pkl`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch {
      alert(t('uploadError'));
    }
  };

  const handleDownloadReport = () => {
    const report = generateReport();
    const blob = new Blob([report], { type: 'text/markdown;charset=utf-8' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vibemine_report_${trainResult.model_id}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  const bestModel = (() => {
    if (!trainResult || trainResult.metrics_table.length === 0) return 'Unknown';
    const table = trainResult.metrics_table as Record<string, unknown>[];
    const metricKey = taskType === 'regression' ? 'R2' : taskType === 'clustering' ? 'Silhouette' : 'Accuracy';
    let bestIdx = 0;
    let bestVal = -Infinity;
    table.forEach((row, i) => {
      const val = parseFloat(String(row[metricKey] ?? '-Infinity'));
      if (!isNaN(val) && val > bestVal) {
        bestVal = val;
        bestIdx = i;
      }
    });
    return String(table[bestIdx]?.Model || 'Unknown');
  })();

  const pycaretModule = taskType === 'classification' ? 'classification' : taskType === 'regression' ? 'regression' : 'clustering';

  return (
    <div className="max-w-2xl mx-auto py-12">
      <div className="text-center mb-8">
        <div className="w-20 h-20 mx-auto mb-8 relative">
          <div className="absolute inset-0 rounded-full bg-gradient-to-br from-green-400 to-emerald-500 opacity-20 animate-ping" />
          <div className="relative w-20 h-20 rounded-full bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center shadow-xl shadow-green-500/20">
            <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          </div>
        </div>

        <h2 className="text-3xl font-semibold tracking-tight text-gray-900 mb-2">{t('train')}</h2>
        <p className="text-sm text-gray-500">{t('exportModel')}</p>
      </div>

      <div className="glass-card rounded-2xl p-6 mb-6">
        <div className="space-y-4">
          {[
            { label: t('bestModel'), value: bestModel, highlight: true },
            { label: t('taskType'), value: taskType === 'classification' ? t('classification') : taskType === 'regression' ? t('regression') : t('clustering') },
            ...(taskType !== 'clustering' ? [{ label: t('targetColumn'), value: targetColumn }] : []),
            { label: 'Model ID', value: trainResult.model_id, mono: true },
          ].map((item) => (
            <div key={item.label} className="flex justify-between items-center py-2 border-b border-gray-100 last:border-0">
              <span className="text-xs text-gray-500">{item.label}</span>
              {item.mono ? (
                <code className="text-xs bg-gray-100 px-2 py-0.5 rounded text-gray-600">{item.value}</code>
              ) : (
                <span className={`text-xs font-medium ${item.highlight ? 'text-indigo-600' : 'text-gray-700'}`}>{item.value}</span>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="flex gap-3 justify-center mb-6">
        <button
          onClick={handleDownload}
          className="py-3 px-6 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-sm font-semibold shadow-lg shadow-indigo-500/20 hover:shadow-xl hover:shadow-indigo-500/30 transition-all duration-300 flex items-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          导出模型
        </button>
        <button
          onClick={() => setShowReport(!showReport)}
          className="py-3 px-5 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-500 text-white text-sm font-semibold shadow-lg shadow-blue-500/20 hover:shadow-xl hover:shadow-blue-500/30 transition-all duration-300 flex items-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          {showReport ? '收起报告' : '查看报告'}
        </button>
        <button
          onClick={handleDownloadReport}
          className="py-3 px-5 rounded-xl bg-gradient-to-r from-emerald-500 to-green-500 text-white text-sm font-semibold shadow-lg shadow-emerald-500/20 hover:shadow-xl hover:shadow-emerald-500/30 transition-all duration-300 flex items-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          下载报告
        </button>
      </div>

      {showReport && (
        <div className="glass-card rounded-2xl p-6 mb-6 max-h-96 overflow-y-auto">
          <pre className="whitespace-pre-wrap font-sans text-sm text-gray-700 leading-relaxed">{generateReport()}</pre>
        </div>
      )}

      <div className="flex gap-3 justify-center">
        <button
          onClick={reset}
          className="py-3 px-6 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 text-white text-sm font-semibold shadow-lg shadow-amber-500/20 hover:shadow-xl hover:shadow-amber-500/30 transition-all duration-300 flex items-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          重新开始
        </button>
      </div>

      <div className="glass-card rounded-2xl p-5 mt-6 text-left">
        <p className="text-xs font-medium text-gray-500 mb-2">Python 部署代码</p>
        <p className="text-xs text-gray-600 leading-relaxed mb-3">
          下载模型后可使用以下代码加载：
        </p>
        <pre className="text-[11px] bg-gray-50 rounded-lg p-3 text-gray-700 leading-relaxed overflow-x-auto">
{`from pycaret.${pycaretModule} import load_model, predict_model
model = load_model('vibemine_model_${trainResult.model_id}')
predictions = predict_model(model, data=new_df)`}
        </pre>
      </div>
    </div>
  );
}
