/** Page navigation component: prev/next buttons and direct page number input. */

import React, { useState } from 'react';

interface PageNavigationProps {
  currentPage: number;
  totalPages: number;
  canGoPrev: boolean;
  canGoNext: boolean;
  onGoPrev: () => void;
  onGoNext: () => void;
  onGoToPage: (page: number) => void;
}

const PageNavigation: React.FC<PageNavigationProps> = ({
  currentPage, totalPages, canGoPrev, canGoNext, onGoPrev, onGoNext, onGoToPage,
}) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const n = parseInt(inputValue, 10);
    if (!isNaN(n)) {
      onGoToPage(n);
      setInputValue('');
    }
  };

  return (
    <div className="flex items-center justify-between px-4 py-3 border-t border-gray-800 bg-gray-900/50">
      <button
        onClick={onGoPrev}
        disabled={!canGoPrev}
        className="btn-ghost disabled:opacity-30"
        aria-label="이전 페이지"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        이전
      </button>

      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <span className="text-sm text-gray-400">페이지</span>
        <input
          type="number"
          min={1}
          max={totalPages}
          placeholder={String(currentPage)}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          className="w-16 bg-gray-800 border border-gray-700 rounded-lg px-2 py-1 text-center text-sm text-gray-100
                     focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent"
          aria-label="페이지 번호 입력"
        />
        <span className="text-sm text-gray-400">/ {totalPages}</span>
        <button type="submit" className="btn-secondary text-xs px-3 py-1.5">이동</button>
      </form>

      <button
        onClick={onGoNext}
        disabled={!canGoNext}
        className="btn-ghost disabled:opacity-30"
        aria-label="다음 페이지"
      >
        다음
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </button>
    </div>
  );
};

export default PageNavigation;
