/** Progress bar component for OCR processing status. */

import React from 'react';
import type { DocumentStatusResponse } from '../types';

interface ProgressBarProps {
  status: DocumentStatusResponse;
  filename?: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ status, filename }) => {
  const pct = Math.min(100, Math.round(status.progress_percent));
  const isDone = status.status === 'completed';
  const isFailed = status.status === 'failed';

  return (
    <div className="glass-card p-6 animate-fade-in">
      <div className="flex items-center justify-between mb-3">
        <div>
          <p className="text-sm font-medium text-gray-300">
            {isDone ? '처리 완료' : isFailed ? '처리 실패' : 'OCR 처리 중…'}
          </p>
          {filename && <p className="text-xs text-gray-500 mt-0.5 truncate max-w-xs">{filename}</p>}
        </div>
        <span className={`text-2xl font-bold tabular-nums ${isDone ? 'text-emerald-400' : isFailed ? 'text-red-400' : 'text-brand-400'}`}>
          {pct}%
        </span>
      </div>

      {/* Progress track */}
      <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            isDone ? 'bg-emerald-500' : isFailed ? 'bg-red-500' : 'bg-brand-500'
          }`}
          style={{ width: `${pct}%` }}
        />
        {!isDone && !isFailed && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-[shimmer_1.5s_infinite]" />
        )}
      </div>

      <p className="text-xs text-gray-500 mt-2 text-right tabular-nums">
        {status.processed_pages} / {status.total_pages} 페이지
      </p>
    </div>
  );
};

export default ProgressBar;
