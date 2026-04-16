import { useAppStore } from '../store';
import { useI18n } from '../i18n/index';

export default function Stepper() {
  const currentStep = useAppStore((s) => s.step);
  const { t } = useI18n();

  const steps = [
    { label: () => t('stepUpload'), icon: '↑' },
    { label: () => t('stepConfig'), icon: '⚙' },
    { label: () => t('stepTrain'), icon: '⚡' },
    { label: () => t('stepEvaluate'), icon: '◈' },
    { label: () => t('stepExport'), icon: '↓' },
  ];

  return (
    <div className="flex items-center justify-center py-8 px-4">
      {steps.map((s, i) => (
        <div key={i} className="flex items-center">
          <div className="flex flex-col items-center gap-1.5">
            <div
              className={`w-10 h-10 rounded-full flex items-center justify-center text-base font-semibold transition-all duration-500 ease-out ${
                i < currentStep
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/30'
                  : i === currentStep
                  ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/40 ring-4 ring-indigo-100'
                  : 'bg-gray-100 text-gray-400'
              }`}
            >
              {i < currentStep ? '✓' : s.icon}
            </div>
            <span
              className={`text-xs font-medium tracking-wide transition-colors duration-300 ${
                i <= currentStep ? 'text-indigo-600' : 'text-gray-400'
              }`}
            >
              {s.label()}
            </span>
          </div>
          {i < steps.length - 1 && (
            <div
              className={`h-0.5 mx-2 rounded-full transition-all duration-700 ease-out ${
                i < currentStep ? 'bg-indigo-400 w-8' : 'bg-gray-200 w-12'
              }`}
            />
          )}
        </div>
      ))}
    </div>
  );
}
