import { useState } from 'react';
import { useAppStore } from '../store';
import { useI18n } from '../i18n';

export default function StepReport() {
  const { t } = useI18n();
  const trainResult = useAppStore((s) => s.trainResult);
  const taskType = useAppStore((s) => s.taskType);
  const targetColumn = useAppStore((s) => s.targetColumn);
  const selectedModels = useAppStore((s) => s.selectedModels);
  const ignoreColumns = useAppStore((s) => s.ignoreColumns);
  const useSmote = useAppStore((s) => s.useSmote);
  const filename = useAppStore((s) => s.filename);
  const dataInfo = useAppStore((s) => s.dataInfo);
  const dataDescribe = useAppStore((s) => s.dataDescribe);
  const totalRows = useAppStore((s) => s.totalRows);
  const shape = useAppStore((s) => s.shape);
  const aiInsight = useAppStore((s) => s.aiInsight);
  const aiEvaluation = useAppStore((s) => s.aiEvaluation);
  const misclassifiedAnalysis = useAppStore((s) => s.misclassifiedAnalysis);

  const [showReport, setShowReport] = useState(false);

  if (!trainResult) return null;

  const TASK_NAMES: Record<string, string> = {
    classification: t('classification'),
    regression: t('regression'),
    clustering: t('clustering'),
  };

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
    selectedModels.forEach(m => {
      report += `- ${getFullModelName(m)} (${m})\n`;
    });
    report += `\n`;

    report += `### 2.2 忽略列\n\n`;
    if (ignoreColumns.length > 0) {
      ignoreColumns.forEach(c => { report += `- ${c}\n`; });
    } else {
      report += `无\n`;
    }
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
    const statsKeys = ['count', 'mean', 'std', 'min', 'max'];
    statsKeys.forEach(key => {
      const vals = Object.keys(dataDescribe).slice(0, 5).map(k => {
        const v = (dataDescribe as any)[k]?.[key];
        return v !== undefined ? v : '-';
      });
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
        const vals = metricCols.map(k => {
          const v = (row as any)[k];
          return v !== null && v !== undefined ? v : '-';
        });
        report += `| ${row.Model || '-'} | ${vals.join(' | ')} |\n`;
      });
      report += `\n`;
    }

    if (Object.keys(modelScores).length > 0) {
      report += `### 4.3 所有模型得分\n\n`;
      Object.entries(modelScores).forEach(([name, score]) => {
        report += `- **${name}**: ${score}\n`;
      });
      report += `\n`;
    }

    if (Object.keys(featImportance).length > 0) {
      report += `### 4.4 特征重要性 Top 10\n\n`;
      report += `| 特征 | 重要性 |\n|------|--------|\n`;
      const top10 = Object.entries(featImportance).sort((a, b) => (b[1] as number) - (a[1] as number)).slice(0, 10);
      top10.forEach(([name, value]) => {
        report += `| ${name} | ${value} |\n`;
      });
      report += `\n`;
    }

    if (aiEvaluation) {
      report += `---\n\n## 五、AI 模型评估\n\n${aiEvaluation}\n\n`;
    }

    if (misclassifiedAnalysis) {
      report += `## 六、错误样本分析\n\n${misclassifiedAnalysis}\n\n`;
    }

    report += `---\n\n*报告由 VibeMine 自动生成*`;
    return report;
  };

  const handleDownload = () => {
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

  return (
    <>
      <div className="mt-6 flex items-center gap-3">
        <button
          onClick={() => setShowReport(!showReport)}
          className="px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-indigo-500 text-white text-sm font-medium shadow-md hover:shadow-lg hover:shadow-blue-500/20 transition-all flex items-center gap-2"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          {showReport ? '收起报告' : '查看完整报告'}
        </button>
        <button
          onClick={handleDownload}
          className="px-4 py-2 rounded-lg bg-gradient-to-r from-emerald-500 to-green-500 text-white text-sm font-medium shadow-md hover:shadow-lg hover:shadow-emerald-500/20 transition-all flex items-center gap-2"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          下载报告
        </button>
      </div>
      {showReport && (
        <div className="mt-4 glass-card rounded-2xl p-6">
          <div className="prose prose-sm max-w-none text-gray-700 whitespace-pre-wrap">
            <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">{generateReport()}</pre>
          </div>
        </div>
      )}
    </>
  );
}
