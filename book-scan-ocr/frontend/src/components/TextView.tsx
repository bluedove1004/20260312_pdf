/** Text view tab: displays the full_text of a page in readable form. */

import React from 'react';
import type { PageResult } from '../types';

interface TextViewProps {
  page: PageResult;
}

const TextView: React.FC<TextViewProps> = ({ page }) => {
  if (page.status === 'failed') {
    return (
      <div className="flex flex-col items-center justify-center h-40 text-red-400 gap-2">
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="text-sm">OCR 처리 실패: {page.error}</p>
      </div>
    );
  }

  if (page.status === 'empty' || !page.full_text) {
    return (
      <div className="flex flex-col items-center justify-center h-40 text-gray-500 gap-2">
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        <p className="text-sm">텍스트를 인식하지 못했습니다</p>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      {/* Stats row */}
      <div className="flex gap-4 mb-4 text-xs text-gray-500">
        <span><span className="text-gray-300 font-medium">{page.block_count}</span> 블록</span>
        <span>평균 신뢰도 <span className={`font-medium ${page.avg_confidence >= 0.9 ? 'text-emerald-400' : page.avg_confidence >= 0.7 ? 'text-yellow-400' : 'text-red-400'}`}>{(page.avg_confidence * 100).toFixed(1)}%</span></span>
        <span>크기 <span className="text-gray-300 font-medium">{page.width}×{page.height}</span></span>
      </div>

      {/* Full text */}
      <div className="bg-gray-950/50 rounded-xl border border-gray-800 p-5 leading-relaxed text-gray-200 text-[15px] whitespace-pre-wrap font-sans">
        {page.full_text}
      </div>
    </div>
  );
};

export default TextView;
