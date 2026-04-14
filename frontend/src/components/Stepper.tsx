import { useAppStore } from '../store';

const steps = [
  { label: '数据上传', icon: '📤' },
  { label: '任务配置', icon: '⚙️' },
  { label: '模型训练', icon: '🚀' },
  { label: '评估解读', icon: '📊' },
  { label: '模型导出', icon: '💾' },
];

export default function Stepper() {
  const currentStep = useAppStore((s) => s.step);

  return (
    <div className="flex items-center justify-center py-6 px-4">
      {steps.map((s, i) => (
        <div key={i} className="flex items-center">
          <div className="flex flex-col items-center">
            <div
              className={`w-11 h-11 rounded-full flex items-center justify-center text-lg font-bold transition-all duration-500 ${
                i < currentStep
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-300'
                  : i === currentStep
                  ? 'bg-indigo-500 text-white ring-4 ring-indigo-200 shadow-lg shadow-indigo-300'
                  : 'bg-gray-200 text-gray-400'
              }`}
            >
              {i < currentStep ? '✓' : s.icon}
            </div>
            <span
              className={`mt-2 text-xs font-medium whitespace-nowrap ${
                i <= currentStep ? 'text-indigo-700' : 'text-gray-400'
              }`}
            >
              {s.label}
            </span>
          </div>
          {i < steps.length - 1 && (
            <div
              className={`w-12 sm:w-20 h-1 mx-1 rounded-full transition-all duration-500 ${
                i < currentStep ? 'bg-indigo-500' : 'bg-gray-200'
              }`}
            />
          )}
        </div>
      ))}
    </div>
  );
}
