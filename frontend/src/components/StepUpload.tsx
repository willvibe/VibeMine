import { useState, useCallback } from 'react';
import { useAppStore } from '../store';
import { uploadFile } from '../api';

export default function StepUpload() {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const setUploadData = useAppStore((s) => s.setUploadData);
  const setImbalanceInfo = useAppStore((s) => s.setImbalanceInfo);
  const setStep = useAppStore((s) => s.setStep);
  const setError = useAppStore((s) => s.setError);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.endsWith('.csv')) {
        setError('请上传 CSV 格式的文件');
        return;
      }
      setUploading(true);
      setError(null);
      try {
        const data = await uploadFile(file);
        setImbalanceInfo(data.imbalance_detected || false, data.imbalance_ratio || 0);
        setUploadData({
          filename: data.filename,
          columns: data.columns,
          shape: data.shape,
          preview: data.preview,
          columnDetails: data.column_details,
          aiInsight: data.ai_insight,
        });
        setStep(1);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : '上传失败，请重试';
        setError(msg);
      } finally {
        setUploading(false);
      }
    },
    [setUploadData, setStep, setError]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="max-w-2xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">上传你的数据集</h2>
        <p className="text-gray-500">支持 CSV 格式，AI 将自动分析数据特征并给出建议</p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`relative border-2 border-dashed rounded-2xl p-16 text-center transition-all duration-300 cursor-pointer ${
          dragging
            ? 'border-indigo-500 bg-indigo-50 scale-[1.02]'
            : 'border-gray-300 bg-gray-50 hover:border-indigo-400 hover:bg-indigo-50/50'
        }`}
        onClick={() => document.getElementById('file-input')?.click()}
      >
        <input
          id="file-input"
          type="file"
          accept=".csv"
          className="hidden"
          onChange={onFileChange}
        />
        {uploading ? (
          <div className="flex flex-col items-center gap-4">
            <div className="w-12 h-12 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin" />
            <p className="text-indigo-600 font-medium">正在上传并分析数据...</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 bg-indigo-100 rounded-2xl flex items-center justify-center">
              <svg className="w-8 h-8 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div>
              <p className="text-lg font-semibold text-gray-700">
                拖拽 CSV 文件到此处，或点击选择
              </p>
              <p className="text-sm text-gray-400 mt-1">支持 .csv 格式文件</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
