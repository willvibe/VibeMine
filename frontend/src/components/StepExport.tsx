import { useAppStore } from '../store';
import { useI18n } from '../i18n/index';
import { downloadModel } from '../api';

export default function StepExport() {
  const { t } = useI18n();
  const trainResult = useAppStore((s) => s.trainResult);
  const taskType = useAppStore((s) => s.taskType);
  const targetColumn = useAppStore((s) => s.targetColumn);
  const reset = useAppStore((s) => s.reset);

  if (!trainResult) return null;

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
    <div className="max-w-lg mx-auto text-center py-12">
      <div className="w-20 h-20 mx-auto mb-8 relative">
        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-green-400 to-emerald-500 opacity-20 animate-ping" />
        <div className="relative w-20 h-20 rounded-full bg-gradient-to-br from-green-400 to-emerald-500 flex items-center justify-center shadow-xl shadow-green-500/20">
          <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
        </div>
      </div>

      <h2 className="text-3xl font-semibold tracking-tight text-gray-900 mb-2">{t('train')}</h2>
      <p className="text-sm text-gray-500 mb-10">{t('exportModel')}</p>

      <div className="glass-card rounded-2xl p-6 text-left mb-8">
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

      <div className="flex gap-3 justify-center mb-8">
        <button
          onClick={handleDownload}
          className="py-3 px-8 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-sm font-semibold shadow-lg shadow-indigo-500/20 hover:shadow-xl hover:shadow-indigo-500/30 transition-all duration-300 flex items-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          {t('exportModel')}
        </button>
        <button
          onClick={reset}
          className="py-3 px-6 rounded-xl border border-gray-200 text-gray-600 text-sm font-medium hover:bg-gray-50 transition-all duration-300"
        >
          {t('stepUpload')}
        </button>
      </div>

      <div className="glass-card rounded-2xl p-5 text-left">
        <p className="text-xs font-medium text-gray-500 mb-2">{t('exportModel')}</p>
        <p className="text-xs text-gray-600 leading-relaxed mb-3">
          {t('exportModel')}
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
