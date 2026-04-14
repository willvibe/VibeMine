import { useAppStore } from '../store';
import { downloadModel } from '../api';

export default function StepExport() {
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
      alert('模型下载失败，请重试');
    }
  };

  const bestModel =
    trainResult.metrics_table.length > 0
      ? String(trainResult.metrics_table[0]['Model'] || trainResult.metrics_table[0]['index'] || '未知')
      : '未知';

  return (
    <div className="max-w-2xl mx-auto text-center py-12">
      <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg className="w-10 h-10 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
      </div>

      <h2 className="text-2xl font-bold text-gray-800 mb-3">挖掘完成！</h2>
      <p className="text-gray-500 mb-8">你的最佳模型已就绪，可以下载用于本地部署</p>

      <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mb-8 text-left">
        <div className="space-y-4">
          <div className="flex justify-between items-center py-3 border-b border-gray-100">
            <span className="text-gray-500">最佳模型</span>
            <span className="font-semibold text-indigo-600">{bestModel}</span>
          </div>
          <div className="flex justify-between items-center py-3 border-b border-gray-100">
            <span className="text-gray-500">任务类型</span>
            <span className="font-medium text-gray-700">
              {taskType === 'classification' ? '分类' : '回归'}
            </span>
          </div>
          <div className="flex justify-between items-center py-3 border-b border-gray-100">
            <span className="text-gray-500">目标列</span>
            <span className="font-medium text-gray-700">{targetColumn}</span>
          </div>
          <div className="flex justify-between items-center py-3">
            <span className="text-gray-500">模型 ID</span>
            <code className="text-sm bg-gray-100 px-3 py-1 rounded-lg text-gray-600">{trainResult.model_id}</code>
          </div>
        </div>
      </div>

      <div className="flex gap-4 justify-center">
        <button
          onClick={handleDownload}
          className="py-3 px-8 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700 shadow-md shadow-indigo-200 transition-all flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          下载最佳模型 (Pipeline)
        </button>
        <button
          onClick={reset}
          className="py-3 px-8 rounded-xl border-2 border-gray-200 text-gray-600 font-medium hover:bg-gray-50 transition-all"
        >
          开始新任务
        </button>
      </div>

      <div className="mt-8 p-4 bg-amber-50 rounded-xl border border-amber-200 text-sm text-amber-700 text-left">
        <strong>💡 使用提示：</strong> 下载的 .pkl 文件包含完整的 PyCaret Pipeline，你可以使用以下代码加载并预测：
        <pre className="mt-2 p-3 bg-amber-100 rounded-lg overflow-x-auto text-xs">
{`from pycaret.classification import load_model, predict_model
model = load_model('vibemine_model_${trainResult.model_id}')
predictions = predict_model(model, data=new_df)`}
        </pre>
      </div>
    </div>
  );
}
