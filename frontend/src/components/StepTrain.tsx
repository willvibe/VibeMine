import { useAppStore } from '../store';
import { useI18n } from '../i18n/index';
import { startTraining, getTrainStatus, stopTraining } from '../api';
import { useEffect, useRef, useState, useCallback } from 'react';

export default function StepTrain() {
  const { t } = useI18n();
  const filename = useAppStore((s) => s.filename);
  const taskType = useAppStore((s) => s.taskType);
  const targetColumn = useAppStore((s) => s.targetColumn);
  const selectedModels = useAppStore((s) => s.selectedModels);
  const ignoreColumns = useAppStore((s) => s.ignoreColumns);
  const useSmote = useAppStore((s) => s.useSmote);
  const setTraining = useAppStore((s) => s.setTraining);
  const setTrainResult = useAppStore((s) => s.setTrainResult);
  const setStep = useAppStore((s) => s.setStep);
  const setError = useAppStore((s) => s.setError);
  const setAiEvaluation = useAppStore((s) => s.setAiEvaluation);
  const setMisclassifiedAnalysis = useAppStore((s) => s.setMisclassifiedAnalysis);

  const [progress, setProgress] = useState(0);
  const [trainingStage, setTrainingStage] = useState(t('initializing'));
  const [completedModels, setCompletedModels] = useState<string[]>([]);
  const sessionIdRef = useRef<string>('');
  const pollTimerRef = useRef<number | null>(null);
  const stoppedRef = useRef(false);

  const pollStatus = useCallback(async () => {
    if (!sessionIdRef.current || stoppedRef.current) return;
    try {
      const data = await getTrainStatus(sessionIdRef.current);

      if (data.status === 'completed' && data.result) {
        setProgress(100);
        setTrainingStage(t('trainingComplete'));
        const storedSessionId = sessionIdRef.current || data.session_id;
        setTrainResult(data.result, storedSessionId);
        setAiEvaluation('');
        setMisclassifiedAnalysis('');
        setStep(3);
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        return;
      }

      if (data.status === 'completed' && !data.result) {
        setError('训练结果为空，请检查数据格式后重试');
        setTraining(false);
        setStep(1);
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        return;
      }

      if (data.status === 'error') {
        setError(data.error || '训练失败，请稍后重试');
        setTraining(false);
        setStep(1);
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        return;
      }

      if (data.status === 'stopped') {
        setError(t('stopTraining'));
        setTraining(false);
        setStep(1);
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        return;
      }

      setProgress(data.progress || 0);
      setTrainingStage(data.current_model || t('training'));
      if (data.completed_models) setCompletedModels(data.completed_models);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : '网络错误，请检查连接后重试';
      setError(errorMsg);
      setTraining(false);
      setStep(1);
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    }
  }, [setTrainResult, setStep, setTraining, setError, t, setAiEvaluation, setMisclassifiedAnalysis]);

  useEffect(() => {
    stoppedRef.current = false;

    const start = async () => {
      try {
        const data = await startTraining({
          filename,
          task_type: taskType,
          target_column: taskType === 'clustering' ? undefined : targetColumn,
          selected_models: selectedModels,
          ignore_columns: ignoreColumns,
          use_smote: useSmote,
          use_outlier_removal: true,
          use_advanced_imputation: true,
          use_stratified_cv: true,
          use_tuning: true,
          use_ensembling: true,
        });
        sessionIdRef.current = data.session_id;
        setTrainingStage(t('initEngine'));
        pollTimerRef.current = window.setInterval(pollStatus, 1000);
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : '启动训练失败，请检查数据后重试';
        setError(errorMsg);
        setTraining(false);
        setStep(1);
      }
    };

    start();

    return () => {
      sessionIdRef.current = '';
      stoppedRef.current = true;
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  const handleStop = useCallback(async () => {
    if (sessionIdRef.current) {
      try {
        await stopTraining(sessionIdRef.current);
      } catch { /* ignore */ }
    }
    sessionIdRef.current = '';
    stoppedRef.current = true;
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    setTraining(false);
    setError(t('stopTraining'));
    setStep(1);
  }, [setTraining, setError, setStep, t]);

  const phases = [
    () => t('preprocessing'),
    () => t('modelTraining'),
    () => t('modelEvaluation'),
    () => t('done'),
  ];
  const phaseIndex = (() => {
    if (progress <= 8) return 0;
    if (progress < 92) return 1;
    if (progress < 100) return 2;
    return 3;
  })();

  const taskLabel = taskType === 'classification' ? t('classification') : taskType === 'regression' ? t('regression') : t('clustering');

  return (
    <div className="max-w-lg mx-auto text-center py-12">
      <div className="relative mb-10">
        <div className="w-24 h-24 mx-auto relative">
          <svg className="w-24 h-24 -rotate-90" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="45" fill="none" stroke="#f1f5f9" strokeWidth="6" />
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="url(#progress-gradient)"
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={`${progress * 2.83} 283`}
              className="transition-all duration-700 ease-out"
            />
            <defs>
              <linearGradient id="progress-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#6366f1" />
                <stop offset="100%" stopColor="#a855f7" />
              </linearGradient>
            </defs>
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-2xl font-bold text-gray-800">{progress}%</span>
          </div>
        </div>
      </div>

      <h2 className="text-2xl font-semibold tracking-tight text-gray-900 mb-1">{t('train')}</h2>
      <p className="text-sm text-gray-500 mb-8">{trainingStage}</p>

      <div className="flex gap-1.5 justify-center mb-8">
        {phases.map((phase, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <div
              className={`w-2 h-2 rounded-full transition-all duration-500 ${
                i < phaseIndex ? 'bg-indigo-500' : i === phaseIndex ? 'bg-indigo-400 animate-pulse' : 'bg-gray-200'
              }`}
            />
            <span className={`text-[10px] font-medium ${i <= phaseIndex ? 'text-gray-700' : 'text-gray-400'}`}>
              {phase()}
            </span>
          </div>
        ))}
      </div>

      {completedModels.length > 0 && (
        <div className="flex flex-wrap gap-2 justify-center mb-8">
          {completedModels.map((model) => (
            <span
              key={model}
              className="px-3 py-1 rounded-full bg-green-50 text-green-600 text-xs font-medium border border-green-100"
            >
              ✓ {model}
            </span>
          ))}
        </div>
      )}

      <div className="glass-card rounded-2xl p-5 text-left">
        <div className="space-y-3">
          {[
            { label: t('taskType'), value: taskLabel },
            taskType !== 'clustering' && { label: t('targetColumn'), value: targetColumn },
            { label: t('models'), value: `${selectedModels.length} ${t('modelCount')?.replace(/.*?(\d+).*/,'$1') || ''}` },
            useSmote && { label: t('smote'), value: t('smoteEnabled') },
            ignoreColumns.length > 0 && { label: t('ignoreColumns'), value: `${ignoreColumns.length}` },
          ]
            .filter(Boolean)
            .map((item: any, i) => (
              <div key={i} className="flex justify-between items-center py-1.5">
                <span className="text-xs text-gray-500">{item.label}</span>
                <span className={`text-xs font-medium ${item.label === t('smote') ? 'text-green-600' : 'text-gray-700'}`}>
                  {item.value}
                </span>
              </div>
            ))}
        </div>
      </div>

      <button
        onClick={handleStop}
        className="mt-8 px-6 py-2.5 rounded-xl bg-red-50 text-red-500 text-sm font-medium border border-red-100 hover:bg-red-100 transition-all duration-300"
      >
        {t('stopTraining')}
      </button>
    </div>
  );
}
