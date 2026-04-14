import { useAppStore } from './store';
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
      case 0:
        return <StepUpload />;
      case 1:
        return <StepConfig />;
      case 2:
        return <StepTrain />;
      case 3:
        return <StepEvaluate />;
      case 4:
        return <StepExport />;
      default:
        return <StepUpload />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <Stepper />
      <main className="px-4 pb-12">
        {error && (
          <div className="max-w-2xl mx-auto mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3">
            <span className="text-red-500 text-lg">⚠️</span>
            <div className="flex-1">
              <p className="text-red-700 text-sm font-medium">{error}</p>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-red-400 hover:text-red-600 text-lg leading-none"
            >
              ×
            </button>
          </div>
        )}
        {renderStep()}
      </main>
    </div>
  );
}

export default App;
