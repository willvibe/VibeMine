import { useAppStore } from './store';
import { I18nProvider } from './i18n/index';
import Header from './components/Header';
import Stepper from './components/Stepper';
import StepUpload from './components/StepUpload';
import StepConfig from './components/StepConfig';
import StepTrain from './components/StepTrain';
import StepEvaluate from './components/StepEvaluate';
import StepExport from './components/StepExport';

function App() {
  const step = useAppStore((s) => s.step);
  const error = useAppStore((s) => s.error);
  const setError = useAppStore((s) => s.setError);

  const renderStep = () => {
    switch (step) {
      case 0: return <StepUpload />;
      case 1: return <StepConfig />;
      case 2: return <StepTrain />;
      case 3: return <StepEvaluate />;
      case 4: return <StepExport />;
      default: return <StepUpload />;
    }
  };

  return (
    <I18nProvider>
      <div className="min-h-screen bg-[#f8f9fa]">
        <Header />
        <Stepper />
        <main className="px-4 pb-16">
          {error && (
            <div className="max-w-xl mx-auto mb-6 p-4 glass-card rounded-xl flex items-center justify-between gap-3 animate-in fade-in slide-in-from-top-2 duration-500">
              <div className="flex items-center gap-3">
                <span className="text-sm text-amber-600">⚠️</span>
                <p className="text-sm text-gray-700">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="text-gray-400 hover:text-gray-600 text-lg leading-none transition-colors"
              >
                ×
              </button>
            </div>
          )}
          <div className="animate-in fade-in duration-500">
            {renderStep()}
          </div>
        </main>
      </div>
    </I18nProvider>
  );
}

export default App;
