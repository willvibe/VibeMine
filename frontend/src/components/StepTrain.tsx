import { useAppStore } from '../store';
import { startTraining, getTrainStatus, stopTraining } from '../api';
import { useEffect, useRef, useState, useCallback } from 'react';

export default function StepTrain() {
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

  const [progress, setProgress] = useState(0);
  const [trainingStage, setTrainingStage] = useState('提交训练任务...');
  const [completedModels, setCompletedModels] = useState<string[]>([]);
  const [currentModel, setCurrentModel] = useState('');
  const [totalModels, setTotalModels] = useState(selectedModels.length);
  const sessionIdRef = useRef<string>('');
  const pollTimerRef = useRef<number | null>(null);
  const stoppedRef = useRef(false);

  const pollStatus = useCallback(async () => {
    if (!sessionIdRef.current || stoppedRef.current) return;
    try {
      const data = await getTrainStatus(sessionIdRef.current);

      if (data.status === 'completed' && data.result) {
        setProgress(100);
        setTrainingStage('训练完成!');
        setTrainResult(data.result);
        setStep(3);
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        return;
      }

      if (data.status === 'error') {
        setError(data.error || '训练失败');
        setTraining(false);
        setStep(1);
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        return;
      }

      if (data.status === 'stopped') {
        setError('训练已停止');
        setTraining(false);
        setStep(1);
        if (pollTimerRef.current) clearInterval(pollTimerRef.current);
        return;
      }

      setProgress(data.progress || 0);
      setTrainingStage(data.current_model || '训练中...');
      setCurrentModel(data.current_model || '');
      if (data.completed_models) setCompletedModels(data.completed_models);
    } catch {
      setError('获取训练状态失败');
      setTraining(false);
      setStep(1);
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    }
  }, [setTrainResult, setStep, setTraining, setError]);

  useEffect(() => {
    stoppedRef.current = false;

    const start = async () => {
      try {
        const data = await startTraining({
          filename,
          task_type: taskType,
          target_column: targetColumn,
          selected_models: selectedModels,
          ignore_columns: ignoreColumns,
          use_smote: useSmote,
        });
        sessionIdRef.current = data.session_id;
        setTrainingStage('训练已启动...');
        setTotalModels(selectedModels.length);

        pollTimerRef.current = window.setInterval(pollStatus, 1000);
      } catch {
        setError('提交训练任务失败');
        setTraining(false);
        setStep(1);
      }
    };

    start();

    const handleBeforeUnload = () => {
      if (sessionIdRef.current) {
        stopTraining(sessionIdRef.current);
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
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
    setError('用户取消了训练');
    setStep(1);
  }, [setTraining, setError, setStep]);

  const taskTypeLabel = {
    classification: '分类',
    regression: '回归',
    clustering: '聚类',
  }[taskType] || taskType;

  return (
    <div className="max-w-2xl mx-auto text-center py-16">
      <div className="relative inline-block mb-8">
        <div className="w-24 h-24 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin" />
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-3xl">🧠</span>
        </div>
      </div>

      <h2 className="text-2xl font-bold text-gray-800 mb-3">AutoML 引擎运行中</h2>
      <p className="text-gray-500 mb-2">
        正在使用 PyCaret 横向评估 {selectedModels.length} 种算法
      </p>
      <p className="text-indigo-600 font-medium mb-2">{trainingStage}</p>
      {currentModel && (
        <p className="text-sm text-purple-600 animate-pulse">
          正在训练: {currentModel}
        </p>
      )}

      <div className="w-full max-w-md mx-auto mb-6">
        <div className="flex justify-between text-sm text-gray-600 mb-1">
          <span>训练进度</span>
          <span>{progress}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
          <div
            className="bg-gradient-to-r from-indigo-500 to-purple-600 h-full rounded-full transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="mt-2 text-xs text-gray-500">
          已完成模型: {completedModels.length} / {totalModels}
        </div>
      </div>

      {completedModels.length > 0 && (
        <div className="w-full max-w-md mx-auto mb-6">
          <div className="flex flex-wrap gap-2 justify-center">
            {completedModels.map((model) => (
              <span
                key={model}
                className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm"
              >
                ✓ {model}
              </span>
            ))}
            {currentModel && (
              <span className="px-3 py-1 bg-yellow-100 text-yellow-700 rounded-full text-sm animate-pulse">
                ⚙ {currentModel}
              </span>
            )}
          </div>
        </div>
      )}

      <div className="bg-gray-50 rounded-xl p-6 text-left max-w-md mx-auto">
        <div className="space-y-3 text-sm">
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
            <span className="text-gray-500">选中算法</span>
            <span className="font-medium text-gray-700">{selectedModels.length} 个</span>
          </div>
          {ignoreColumns.length > 0 && (
            <div className="flex justify-between">
              <span className="text-gray-500">忽略列</span>
              <span className="font-medium text-gray-700">{ignoreColumns.length} 个</span>
            </div>
          )}
          {useSmote && (
            <div className="flex justify-between">
              <span className="text-gray-500">SMOTE</span>
              <span className="font-medium text-green-600">已开启</span>
            </div>
          )}
          <div className="flex justify-between">
            <span className="text-gray-500">当前阶段</span>
            <span className="font-medium text-indigo-600">{trainingStage}</span>
          </div>
        </div>
      </div>

      <div className="mt-8 flex gap-2 justify-center">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>

      <button
        onClick={handleStop}
        className="mt-8 px-6 py-3 bg-red-500 text-white font-medium rounded-xl hover:bg-red-600 transition-colors shadow-md"
      >
        停止训练 🛑
      </button>
    </div>
  );
}