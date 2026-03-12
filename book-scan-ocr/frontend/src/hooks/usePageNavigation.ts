/**
 * Hook: manage page navigation state for the document viewer.
 */

import { useCallback, useState } from 'react';

interface UsePageNavigationOptions {
  totalPages: number;
  initialPage?: number;
}

interface UsePageNavigationResult {
  currentPage: number;
  canGoPrev: boolean;
  canGoNext: boolean;
  goToPage: (page: number) => void;
  goNext: () => void;
  goPrev: () => void;
}

export function usePageNavigation({
  totalPages,
  initialPage = 1,
}: UsePageNavigationOptions): UsePageNavigationResult {
  const [currentPage, setCurrentPage] = useState(initialPage);

  const goToPage = useCallback(
    (page: number) => {
      const clamped = Math.max(1, Math.min(page, totalPages));
      setCurrentPage(clamped);
    },
    [totalPages]
  );

  const goNext = useCallback(() => goToPage(currentPage + 1), [currentPage, goToPage]);
  const goPrev = useCallback(() => goToPage(currentPage - 1), [currentPage, goToPage]);

  return {
    currentPage,
    canGoPrev: currentPage > 1,
    canGoNext: currentPage < totalPages,
    goToPage,
    goNext,
    goPrev,
  };
}
