import { useState, useRef } from 'react';
import { useAppStore } from '../store';
import { useI18n } from '../i18n/index';
import { uploadFile, callGemini } from '../api';

type DataTab = 'data' | 'info' | 'stats';

export default function StepUpload() {
  const { t } = useI18n();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState('');
  const [aiLoading, setAiLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<DataTab>('data');

  const {
    filename,
    dataInfo,
    dataDescribe,
    totalRows,
    aiInsight,
    columns,
    preview,
    setUploadData,
    setImbalanceInfo,
    setAiInsight,
    setStep,
    reset,
  } = useAppStore();

  const fmt = (v: unknown): string => {
    if (v === null || v === undefined || v === 'N/A') return '-';
    if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(4);
    return String(v);
  };

  const handleUpload = async (file: File) => {
    if (!file.name.endsWith('.csv')) {
      setUploadError(t('uploadError') + ': Only CSV files supported');
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      setUploadError(t('fileTooBig'));
      return;
    }

    setUploading(true);
    setUploadError('');
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', file);

      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += 10;
        if (progress <= 90) setUploadProgress(progress);
      }, 100);

      const data = await uploadFile(formData);
      clearInterval(progressInterval);
      setUploadProgress(100);

      setUploadData({
        filename: data.filename,
        columns: data.columns,
        shape: data.shape,
        preview: data.preview,
        columnDetails: data.column_details,
        aiInsight: '',
        dataInfo: data.info || [],
        dataDescribe: data.describe || {},
        totalRows: data.total_rows || 0,
      });
      setImbalanceInfo(data.imbalance_detected || false, data.imbalance_ratio || 0);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) await handleUpload(file);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) await handleUpload(file);
  };

  const handleReUpload = () => {
    reset();
    fileInputRef.current?.click();
  };

  const handleAiInsight = async () => {
    if (!filename || dataInfo.length === 0) return;
    setAiLoading(true);
    try {
      const colSummaries = dataInfo.map((info: any) =>
        `- ${info.column} (${info.dtype}): 缺失${info.null}个(${((info.null / (info.non_null + info.null)) * 100).toFixed(1)}%), 唯一值${info.unique}个`
      ).join('\n');
      const prompt = `你是一个资深的数据挖掘专家。用户刚上传了一份数据集，基本信息如下：
数据规模：${totalRows}行 x ${columns.length}列
各列详情：
${colSummaries}

请简要分析该数据的潜在问题（如缺失值处理建议、数据分布），并根据特征判断这更适合做分类还是回归任务，推荐2种最合适的初阶算法。语言要专业且带有引导性，格式使用 Markdown，字数控制在 300 字以内。`;
      const result = await callGemini(prompt);
      setAiInsight(result);
    } catch (err) {
      console.error('AI insight error:', err);
    } finally {
      setAiLoading(false);
    }
  };

  if (filename && dataInfo.length > 0) {
    const describeKeys: string[] = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];

    return (
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="glass-card rounded-2xl p-0 overflow-hidden">
          <div className="flex items-center justify-between border-b border-gray-100 px-5 py-3">
            <div className="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span className="text-sm font-medium text-gray-600">{filename}</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleAiInsight}
                disabled={aiLoading}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-xs font-medium shadow-md hover:shadow-lg hover:shadow-indigo-500/20 transition-all disabled:opacity-60"
                title={t('dataAiInsight')}
              >
                {aiLoading ? (
                  <svg className="animate-spin w-3 h-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                  </svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                )}
                {!aiLoading && <span>AI</span>}
              </button>
              <button
                onClick={handleReUpload}
                className="px-3 py-1.5 rounded-lg bg-white border border-gray-200 text-gray-600 text-xs font-medium hover:bg-gray-50 transition-colors flex items-center gap-1.5"
                title={t('reuploadData')}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
              </button>
            </div>
          </div>

          {aiInsight && (
            <div className="p-5">
              <div className="glass-card rounded-xl p-4 bg-indigo-50/50 border border-indigo-100">
                <div className="flex items-center gap-2 mb-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <h3 className="text-sm font-semibold text-gray-900">{t('dataAiInsight')}</h3>
                </div>
                <div className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                  {aiInsight}
                </div>
              </div>
            </div>
          )}

          <div className="border-t border-gray-100">
            <div className="flex items-center gap-2 px-5 pt-4">
              <button
                onClick={() => setActiveTab('data')}
                className={`px-4 py-2 rounded-t-lg text-sm font-medium transition-all ${activeTab === 'data' ? 'bg-white text-indigo-600 border-b-2 border-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
              >
                Data
              </button>
              <button
                onClick={() => setActiveTab('info')}
                className={`px-4 py-2 rounded-t-lg text-sm font-medium transition-all ${activeTab === 'info' ? 'bg-white text-indigo-600 border-b-2 border-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
              >
                Info
              </button>
              <button
                onClick={() => setActiveTab('stats')}
                className={`px-4 py-2 rounded-t-lg text-sm font-medium transition-all ${activeTab === 'stats' ? 'bg-white text-indigo-600 border-b-2 border-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
              >
                Stats
              </button>
            </div>

          <div className="p-5">
            {activeTab === 'info' && (
              <>
                <div className="overflow-x-auto rounded-xl border border-gray-100">
                  <table className="w-full text-xs">
                    <thead className="bg-gray-50/80">
                      <tr>
                        <th className="px-3 py-2 text-left font-medium text-gray-500 whitespace-nowrap">{t('column')}</th>
                        <th className="px-3 py-2 text-left font-medium text-gray-500 whitespace-nowrap">{t('dtype')}</th>
                        <th className="px-3 py-2 text-left font-medium text-gray-500 whitespace-nowrap">{t('nonNull')}</th>
                        <th className="px-3 py-2 text-left font-medium text-gray-500 whitespace-nowrap">{t('nullCount')}</th>
                        <th className="px-3 py-2 text-left font-medium text-gray-500 whitespace-nowrap">{t('uniqueValues')}</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-50">
                      {dataInfo.map((row, i) => (
                        <tr key={i} className="hover:bg-gray-50/50">
                          <td className="px-3 py-2 font-medium text-gray-700 whitespace-nowrap">{row.column}</td>
                          <td className="px-3 py-2 text-gray-500 whitespace-nowrap">
                            <span className={`px-1.5 py-0.5 rounded text-xs font-mono ${row.dtype === 'object' ? 'bg-amber-50 text-amber-700' : 'bg-blue-50 text-blue-700'}`}>
                              {row.dtype}
                            </span>
                          </td>
                          <td className="px-3 py-2 text-gray-600 whitespace-nowrap">{fmt(row.non_null)}</td>
                          <td className="px-3 py-2 whitespace-nowrap">
                            {row.null > 0 ? (
                              <span className="px-1.5 py-0.5 rounded text-xs bg-red-50 text-red-600">{row.null} ({((row.null / totalRows) * 100).toFixed(1)}%)</span>
                            ) : (
                              <span className="text-gray-400">0</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-gray-600 whitespace-nowrap">{fmt(row.unique)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}

            {activeTab === 'stats' && (
              <div className="overflow-x-auto rounded-xl border border-gray-100">
                <table className="w-full text-xs">
                  <thead className="bg-gray-50/80">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium text-gray-500 whitespace-nowrap">{t('column')}</th>
                      {describeKeys.map(k => (
                        <th key={k} className="px-3 py-2 text-left font-medium text-gray-500 whitespace-nowrap">{t(k as any) || k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {Object.entries(dataDescribe).map(([col, stats]) => (
                      <tr key={col} className="hover:bg-gray-50/50">
                        <td className="px-3 py-2 font-medium text-gray-700 whitespace-nowrap">{col}</td>
                        {describeKeys.map(k => (
                          <td key={k} className="px-3 py-2 text-gray-500 whitespace-nowrap text-xs font-mono">{fmt((stats as any)?.[k])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {activeTab === 'data' && (
              <div className="rounded-xl border border-gray-100 overflow-hidden">
                <div className="max-h-80 overflow-y-auto">
                  <table className="w-full text-xs">
                    <thead className="bg-gray-50/80 sticky top-0">
                      <tr>
                        {columns.map(col => (
                          <th key={col} className="px-3 py-2 text-left font-medium text-gray-500 whitespace-nowrap">{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-50">
                      {(preview || []).slice(0, 20).map((row, i) => (
                        <tr key={i} className="hover:bg-gray-50/50">
                          {columns.map(col => (
                            <td key={col} className="px-3 py-2 text-gray-600 whitespace-nowrap">{String(row[col] ?? '')}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {(preview || []).length > 20 && (
                  <div className="px-3 py-2 text-xs text-gray-400 bg-gray-50/50 text-center">
                    显示前 20 / {(preview || []).length} 条数据
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

        <div className="flex justify-end">
          <button
            onClick={() => setStep(1)}
            className="px-6 py-2.5 rounded-xl bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 transition-colors"
          >
            {t('next')}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div
        className={`relative border-2 border-dashed rounded-2xl p-16 text-center transition-all mt-8 ${dragOver ? 'border-indigo-400 bg-indigo-50' : 'border-gray-200 hover:border-indigo-300'} ${uploading ? 'pointer-events-none opacity-70' : 'cursor-pointer'}`}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={handleFileChange}
        />

        {uploading ? (
          <div className="space-y-4">
            <svg className="animate-spin w-10 h-10 mx-auto text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            <p className="text-sm text-gray-500">{t('uploading')} {uploadProgress}%</p>
            <div className="w-48 mx-auto h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <div className="h-full bg-indigo-500 transition-all duration-300" style={{ width: `${uploadProgress}%` }} />
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto bg-gray-100 rounded-2xl flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="w-8 h-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m6.75 12l-3-3m0 0l-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-700">{t('selectFile')}</p>
              <p className="text-xs text-gray-400 mt-1">{t('dragDrop')}</p>
            </div>
          </div>
        )}
      </div>

      {uploadError && (
        <p className="mt-3 text-sm text-red-500 text-center">{uploadError}</p>
      )}

      <div className="mt-12 grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="glass-card rounded-xl p-4 text-center">
          <div className="w-10 h-10 mx-auto bg-indigo-50 rounded-xl flex items-center justify-center mb-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5" />
            </svg>
          </div>
          <h4 className="text-sm font-semibold text-gray-800">{t('anomalyDetect')}</h4>
          <p className="text-xs text-gray-400 mt-1">{t('anomalyDetectDesc')}</p>
        </div>
        <div className="glass-card rounded-xl p-4 text-center">
          <div className="w-10 h-10 mx-auto bg-indigo-50 rounded-xl flex items-center justify-center mb-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
            </svg>
          </div>
          <h4 className="text-sm font-semibold text-gray-800">{t('aiSuggest')}</h4>
          <p className="text-xs text-gray-400 mt-1">{t('aiSuggestDesc')}</p>
        </div>
        <div className="glass-card rounded-xl p-4 text-center">
          <div className="w-10 h-10 mx-auto bg-indigo-50 rounded-xl flex items-center justify-center mb-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
            </svg>
          </div>
          <h4 className="text-sm font-semibold text-gray-800">{t('modelEvaluation')}</h4>
          <p className="text-xs text-gray-400 mt-1">{t('performanceDesc')}</p>
        </div>
      </div>
    </div>
  );
}
