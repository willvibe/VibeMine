import { create } from 'zustand';

export interface ColumnDetail {
  name: string;
  dtype: string;
  missing_count: number;
  missing_ratio: number;
  unique_count: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  top_value?: string | null;
  is_id?: boolean;
  is_low_variance?: boolean;
}

export interface TrainResult {
  model_id: string;
  metrics_table: Record<string, unknown>[];
  feature_importance: Record<string, number>;
  ai_evaluation: string;
  shap_plot?: string;
  misclassified_samples?: Record<string, unknown>[];
}

interface AppState {
  step: number;
  filename: string;
  columns: string[];
  shape: number[];
  preview: Record<string, unknown>[];
  columnDetails: ColumnDetail[];
  aiInsight: string;
  taskType: string;
  targetColumn: string;
  selectedModels: string[];
  ignoreColumns: string[];
  useSmote: boolean;
  isTraining: boolean;
  trainResult: TrainResult | null;
  error: string | null;
  imbalanceDetected: boolean;
  imbalanceRatio: number;

  setStep: (step: number) => void;
  setUploadData: (data: {
    filename: string;
    columns: string[];
    shape: number[];
    preview: Record<string, unknown>[];
    columnDetails: ColumnDetail[];
    aiInsight: string;
  }) => void;
  setImbalanceInfo: (detected: boolean, ratio: number) => void;
  setTaskType: (type: string) => void;
  setTargetColumn: (col: string) => void;
  setSelectedModels: (models: string[]) => void;
  setIgnoreColumns: (cols: string[]) => void;
  setUseSmote: (val: boolean) => void;
  setTraining: (val: boolean) => void;
  setTrainResult: (result: TrainResult) => void;
  setError: (err: string | null) => void;
  reset: () => void;
}

const initialState = {
  step: 0,
  filename: '',
  columns: [],
  shape: [0, 0] as [number, number],
  preview: [] as Record<string, unknown>[],
  columnDetails: [] as ColumnDetail[],
  aiInsight: '',
  taskType: 'classification',
  targetColumn: '',
  selectedModels: ['lr', 'rf', 'gbc', 'et', 'xgb'],
  ignoreColumns: [] as string[],
  useSmote: false,
  isTraining: false,
  trainResult: null as TrainResult | null,
  error: null as string | null,
  imbalanceDetected: false,
  imbalanceRatio: 0,
};

export const useAppStore = create<AppState>((set) => ({
  ...initialState,
  setStep: (step) => set({ step }),
  setUploadData: (data) =>
    set({
      filename: data.filename,
      columns: data.columns,
      shape: data.shape,
      preview: data.preview,
      columnDetails: data.columnDetails,
      aiInsight: data.aiInsight,
    }),
  setImbalanceInfo: (detected, ratio) => set({ imbalanceDetected: detected, imbalanceRatio: ratio }),
  setTaskType: (taskType) => set({ taskType }),
  setTargetColumn: (targetColumn) => set({ targetColumn }),
  setSelectedModels: (selectedModels) => set({ selectedModels }),
  setIgnoreColumns: (ignoreColumns) => set({ ignoreColumns }),
  setUseSmote: (useSmote) => set({ useSmote }),
  setTraining: (isTraining) => set({ isTraining }),
  setTrainResult: (trainResult) => set({ trainResult }),
  setError: (error) => set({ error }),
  reset: () => set(initialState),
}));