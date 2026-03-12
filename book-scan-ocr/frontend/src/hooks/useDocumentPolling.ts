/**
 * Hook: poll document status every 2 seconds while processing.
 * Stops automatically when status becomes 'completed' or 'failed'.
 */

import { useEffect, useRef, useState } from 'react';
import { getDocumentStatus } from '../services/api';
import type { DocumentStatusResponse } from '../types';

interface UseDocumentPollingOptions {
  documentId: string | null;
  enabled?: boolean;
  intervalMs?: number;
  onComplete?: (status: DocumentStatusResponse) => void;
}

interface UseDocumentPollingResult {
  status: DocumentStatusResponse | null;
  isPolling: boolean;
  error: string | null;
}

export function useDocumentPolling({
  documentId,
  enabled = true,
  intervalMs = 2000,
  onComplete,
}: UseDocumentPollingOptions): UseDocumentPollingResult {
  const [status, setStatus] = useState<DocumentStatusResponse | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    if (!documentId || !enabled) return;

    let cancelled = false;

    const poll = async () => {
      try {
        const s = await getDocumentStatus(documentId);
        if (cancelled) return;
        setStatus(s);
        setError(null);

        if (s.status === 'completed' || s.status === 'failed') {
          setIsPolling(false);
          onCompleteRef.current?.(s);
          return;
        }

        timerRef.current = setTimeout(poll, intervalMs);
      } catch (err: unknown) {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : 'Polling error';
        setError(msg);
        timerRef.current = setTimeout(poll, intervalMs * 2);
      }
    };

    setIsPolling(true);
    poll();

    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      setIsPolling(false);
    };
  }, [documentId, enabled, intervalMs]);

  return { status, isPolling, error };
}
