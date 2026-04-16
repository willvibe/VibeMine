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
  shap_plot?: string;
  misclassified_samples?: Record<string, unknown>[];
}

export interface DataInfo {
  column: string;
  dtype: string;
  non_null: number;
  null: number;
  unique: number;
  sample?: string;
}

export interface DataDescribe {
  [key: string]: {
    count?: number | string;
    mean?: number | string;
    std?: number | string;
    min?: number | string;
    '25%'?: number | string;
    '50%'?: number | string;
    '75%'?: number | string;
    max?: number | string;
    top?: number | string;
    freq?: number | string;
    unique?: number | string;
  };
}

export interface ClassDistribution {
  [column: string]: {
    [value: string]: number;
  };
}

interface AppState {
  step: number;
  filename: string;
  columns: string[];
  shape: number[];
  preview: Record<string, unknown>[];
  columnDetails: ColumnDetail[];
  aiInsight: string;
  dataInfo: DataInfo[];
  dataDescribe: DataDescribe;
  totalRows: number;
  classDistributions: ClassDistribution;
  taskType: string;
  targetColumn: string;
  selectedModels: string[];
  ignoreColumns: string[];
  useSmote: boolean;
  isTraining: boolean;
  trainResult: TrainResult | null;
  sessionId: string | null;
  aiEvaluation: string;
  misclassifiedAnalysis: string;
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
    dataInfo?: DataInfo[];
    dataDescribe?: DataDescribe;
    totalRows?: number;
    classDistributions?: ClassDistribution;
  }) => void;
  setImbalanceInfo: (detected: boolean, ratio: number) => void;
  setTaskType: (type: string) => void;
  setTargetColumn: (col: string) => void;
  setSelectedModels: (models: string[]) => void;
  setIgnoreColumns: (cols: string[]) => void;
  setUseSmote: (val: boolean) => void;
  setTraining: (val: boolean) => void;
  setTrainResult: (result: TrainResult, sessionId?: string) => void;
  setAiEvaluation: (evaluation: string) => void;
  setMisclassifiedAnalysis: (analysis: string) => void;
  setAiInsight: (insight: string) => void;
  setError: (err: string | null) => void;
  reset: () => void;
}

const initialState = {
  step: 0,
  filename: '',
  columns: [] as string[],
  shape: [0, 0] as [number, number],
  preview: [] as Record<string, unknown>[],
  columnDetails: [] as ColumnDetail[],
  aiInsight: '',
  dataInfo: [] as DataInfo[],
  dataDescribe: {} as DataDescribe,
  totalRows: 0,
  classDistributions: {} as ClassDistribution,
  taskType: 'classification',
  targetColumn: '',
  selectedModels: ['lr', 'rf', 'gbc', 'et', 'xgb'],
  ignoreColumns: [] as string[],
  useSmote: false,
  isTraining: false,
  trainResult: null as TrainResult | null,
  sessionId: null as string | null,
  aiEvaluation: '',
  misclassifiedAnalysis: '',
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
      dataInfo: data.dataInfo || [],
      dataDescribe: data.dataDescribe || {},
      totalRows: data.totalRows || 0,
      classDistributions: data.classDistributions || {},
    }),
  setImbalanceInfo: (detected, ratio) => set({ imbalanceDetected: detected, imbalanceRatio: ratio }),
  setTaskType: (taskType) => set({ taskType }),
  setTargetColumn: (targetColumn) => set({ targetColumn }),
  setSelectedModels: (selectedModels) => set({ selectedModels }),
  setIgnoreColumns: (ignoreColumns) => set({ ignoreColumns }),
  setUseSmote: (useSmote) => set({ useSmote }),
  setTraining: (isTraining) => set({ isTraining }),
  setTrainResult: (result, sessionId) => set({ trainResult: result, sessionId: sessionId || null }),
  setAiEvaluation: (aiEvaluation) => set({ aiEvaluation }),
  setMisclassifiedAnalysis: (misclassifiedAnalysis) => set({ misclassifiedAnalysis }),
  setAiInsight: (aiInsight) => set({ aiInsight }),
  setError: (error) => set({ error }),
  reset: () => set(initialState),
}));
