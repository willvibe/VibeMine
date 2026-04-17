import { useState, useEffect } from 'react';
import { useAppStore } from '../store';
import { useI18n } from '../i18n/index';
const TASK_MODELS: Record<string, string[]> = {
  classification: ['lr', 'rf', 'gbc', 'et', 'xgb', 'lightgbm', 'dt', 'knn', 'ada', 'lda', 'nb', 'qda'],
  regression: ['lr', 'ridge', 'lasso', 'en', 'rf', 'et', 'gbr', 'lightgbm', 'knn', 'dt'],
  clustering: ['kmeans', 'hclust', 'dbscan', 'meanshift', 'affinity'],
};

const MODEL_LABELS: Record<string, string> = {
  lr: 'Logistic Regression', rf: 'Random Forest', gbc: 'Gradient Boosting Classifier', et: 'Extra Trees Classifier',
  xgb: 'XGBoost', lightgbm: 'LightGBM', dt: 'Decision Tree', knn: 'K-Nearest Neighbors', ada: 'AdaBoost',
  lda: 'Linear Discriminant Analysis', nb: 'Naive Bayes', qda: 'Quadratic Discriminant Analysis',
  lasso: 'Lasso Regression', en: 'Elastic Net', ridge: 'Ridge Regression', gbr: 'Gradient Boosting Regressor',
  kmeans: 'K-Means', hclust: 'Hierarchical Clustering', dbscan: 'DBSCAN', meanshift: 'Mean Shift', affinity: 'Affinity Propagation',
};

const REGRESSION_MODEL_LABELS: Record<string, string> = {
  lr: 'Linear Regression', ridge: 'Ridge Regression', lasso: 'Lasso Regression',
  en: 'Elastic Net', rf: 'Random Forest Regressor', et: 'Extra Trees Regressor', gbr: 'Gradient Boosting Regressor',
  lightgbm: 'LightGBM Regressor', knn: 'K-Nearest Neighbors Regressor', dt: 'Decision Tree Regressor',
};

function getModelLabel(key: string, taskType: string): string {
  if (taskType === 'regression' && REGRESSION_MODEL_LABELS[key]) {
    return REGRESSION_MODEL_LABELS[key];
  }
  return MODEL_LABELS[key] || key;
}

export default function StepConfig() {
  const { t } = useI18n();
  const {
    filename, columns, taskType, targetColumn, selectedModels,
    ignoreColumns, useSmote, setTaskType, setTargetColumn, setSelectedModels,
    setIgnoreColumns, setUseSmote,
    setStep,
  } = useAppStore();

  const [classDistribution, setClassDistribution] = useState<{ value: string; count: number; pct: number }[]>([]);

  useEffect(() => {
    if (targetColumn && taskType === 'classification') {
      const classDist = useAppStore.getState().classDistributions?.[targetColumn];
      if (classDist) {
        const total = Object.values(classDist).reduce((a, b) => a + b, 0);
        const dist = Object.entries(classDist)
          .map(([value, count]) => ({ value, count, pct: total > 0 ? count / total : 0 }))
          .sort((a, b) => b.count - a.count);
        setClassDistribution(dist);
      }
    } else {
      setClassDistribution([]);
    }
  }, [targetColumn, taskType]);

  const suggestedModels = TASK_MODELS[taskType] || [];
  const isClustering = taskType === 'clustering';

  const handleModelToggle = (model: string) => {
    if (selectedModels.includes(model)) {
      setSelectedModels(selectedModels.filter(m => m !== model));
    } else {
      setSelectedModels([...selectedModels, model]);
    }
  };

  const handleSelectAll = () => setSelectedModels([...suggestedModels]);
  const handleClearAll = () => setSelectedModels([]);
  const handleDefaultModels = () => {
    const defaults = taskType === 'classification'
      ? ['lr', 'rf', 'gbc', 'et', 'xgb']
      : taskType === 'regression'
      ? ['lr', 'rf', 'et']
      : ['kmeans', 'hclust'];
    setSelectedModels(defaults);
  };

  const handleIgnoreToggle = (col: string) => {
    if (ignoreColumns.includes(col)) {
      setIgnoreColumns(ignoreColumns.filter(c => c !== col));
    } else {
      setIgnoreColumns([...ignoreColumns, col]);
    }
  };

  const handleStartTraining = () => {
    if (!filename) return;
    if (selectedModels.length === 0) {
      alert(t('models') + ': please select at least one model');
      return;
    }
    if (!isClustering && !targetColumn) {
      alert(t('targetCol') + ': please select a target column');
      return;
    }
    setStep(2);
  };

  const canUseSmote = taskType === 'classification' && targetColumn && !isClustering;

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div>
        <h2 className="text-3xl font-semibold tracking-tight text-gray-900">{t('stepConfig')}</h2>
        <p className="text-sm text-gray-500 mt-1">{filename}</p>
      </div>

      <div className="glass-card rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">{t('taskTypeLabel')}</h3>
        <div className="grid grid-cols-3 gap-3">
          {(['classification', 'regression', 'clustering'] as const).map(type => (
            <button
              key={type}
              onClick={() => {
                setTaskType(type);
                setTargetColumn('');
                setSelectedModels(TASK_MODELS[type].slice(0, type === 'clustering' ? 3 : 5));
              }}
              className={`px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                taskType === type
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200'
              }`}
            >
              {type === 'classification' ? t('classification') : type === 'regression' ? t('regression') : t('clustering')}
            </button>
          ))}
        </div>
        {taskType === 'clustering' && (
          <p className="text-xs text-gray-500 mt-2">{t('clusteringDesc')}</p>
        )}
      </div>

      {!isClustering && (
        <div className="glass-card rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">{t('targetCol')}</h3>
          <div className="flex flex-wrap gap-2">
            {columns.map(col => {
              const isSelected = targetColumn === col;
              return (
                <button
                  key={col}
                  onClick={() => setTargetColumn(isSelected ? '' : col)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    isSelected
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200'
                  }`}
                >
                  {col}
                </button>
              );
            })}
          </div>

          {targetColumn && taskType === 'classification' && classDistribution.length > 0 && (
            <div className="mt-4 p-4 bg-gray-50 rounded-xl">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{t('classDistribution')}</h4>
                <span className={`text-xs px-2 py-0.5 rounded-full ${classDistribution.length > 2 || (classDistribution[0]?.pct || 0) > 0.8 ? 'bg-amber-50 text-amber-700' : 'bg-green-50 text-green-700'}`}>
                  {classDistribution[0]?.pct > 0.8 ? t('classesImbalanced', { ratio: Math.round((classDistribution[0]?.count || 0) / (classDistribution[classDistribution.length - 1]?.count || 1)) }) : t('classesBalanced')}
                </span>
              </div>
              <div className="space-y-2">
                {classDistribution.map(({ value, count, pct }) => (
                  <div key={value} className="flex items-center gap-3">
                    <span className="text-xs text-gray-600 w-20 truncate">{value}</span>
                    <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div className="h-full bg-indigo-500 rounded-full transition-all" style={{ width: `${pct * 100}%` }} />
                    </div>
                    <span className="text-xs text-gray-500 w-24 text-right">{count} ({(pct * 100).toFixed(1)}%)</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="glass-card rounded-2xl p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-700">{t('ignoreColumns')}</h3>
          <span className="text-xs text-gray-400">{ignoreColumns.length} selected</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {columns.filter(c => c !== targetColumn).map(col => {
            const isIgnored = ignoreColumns.includes(col);
            return (
              <button
                key={col}
                onClick={() => handleIgnoreToggle(col)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  isIgnored
                    ? 'bg-red-100 text-red-600 border border-red-200 line-through opacity-60'
                    : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200'
                }`}
              >
                {col}
              </button>
            );
          })}
        </div>
      </div>

      <div className="glass-card rounded-2xl p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-700">{t('models')}</h3>
          <div className="flex gap-2">
            <button onClick={handleDefaultModels} className="text-xs text-indigo-500 hover:text-indigo-600 font-medium">{t('defaultModels')}</button>
            <span className="text-gray-300">|</span>
            <button onClick={handleSelectAll} className="text-xs text-indigo-500 hover:text-indigo-600 font-medium">{t('selectAll')}</button>
            <span className="text-gray-300">|</span>
            <button onClick={handleClearAll} className="text-xs text-gray-400 hover:text-gray-600">{t('clearAll')}</button>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {suggestedModels.map(m => {
            const isSelected = selectedModels.includes(m);
            return (
              <button
                key={m}
                onClick={() => handleModelToggle(m)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  isSelected
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200'
                }`}
              >
                {getModelLabel(m, taskType)}
              </button>
            );
          })}
        </div>
      </div>

      <div className="glass-card rounded-2xl p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-700">{t('advancedOptions')}</h3>
        </div>
        <div className="space-y-3">
          {taskType === 'classification' && (
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={useSmote}
                onChange={e => setUseSmote(e.target.checked)}
                disabled={!canUseSmote}
                className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 disabled:opacity-50"
              />
              <div>
                <span className="text-sm font-medium text-gray-700">{t('smote')}</span>
                {canUseSmote && (
                  <span className="text-xs text-gray-400 ml-2">{t('smoteEnabled')}</span>
                )}
              </div>
            </label>
          )}
        </div>
      </div>

      <div className="flex gap-3">
        <button
          onClick={() => setStep(0)}
          className="px-6 py-2.5 rounded-xl bg-white border border-gray-200 text-gray-700 text-sm font-medium hover:bg-gray-50 transition-colors"
        >
          {t('back')}
        </button>
        <button
          onClick={handleStartTraining}
          disabled={selectedModels.length === 0 || (!isClustering && !targetColumn)}
          className="flex-1 px-6 py-2.5 rounded-xl bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {t('startTraining')}
        </button>
      </div>
    </div>
  );
}
