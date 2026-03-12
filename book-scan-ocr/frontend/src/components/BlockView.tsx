/** Block view tab: each text_block displayed as a card with confidence color coding. */

import React from 'react';
import type { PageResult, TextBlock } from '../types';

interface BlockViewProps {
  page: PageResult;
}

function confidenceClass(confidence: number): string {
  if (confidence >= 0.9) return 'border-emerald-800/50 bg-emerald-950/20';
  if (confidence >= 0.7) return 'border-yellow-800/50 bg-yellow-950/20';
  return 'border-red-800/50 bg-red-950/20';
}

function confidenceBadgeClass(confidence: number): string {
  if (confidence >= 0.9) return 'text-emerald-400 bg-emerald-900/40';
  if (confidence >= 0.7) return 'text-yellow-400 bg-yellow-900/40';
  return 'text-red-400 bg-red-900/40';
}

const BlockCard: React.FC<{ block: TextBlock; warn: boolean }> = ({ block, warn }) => (
  <div className={`rounded-xl border p-3 transition-colors ${confidenceClass(block.confidence)} ${warn ? 'ring-1 ring-red-700/40' : ''}`}>
    <div className="flex items-start justify-between gap-3 mb-1">
      <span className="text-[11px] text-gray-500 tabular-nums">#{block.block_id} L{block.line_number}</span>
      <span className={`text-[11px] font-mono px-2 py-0.5 rounded-md ${confidenceBadgeClass(block.confidence)}`}>
        {(block.confidence * 100).toFixed(1)}%
      </span>
    </div>
    <p className="text-sm text-gray-200 leading-relaxed">{block.text}</p>
    <p className="text-[10px] text-gray-600 mt-1 font-mono">
      [{block.bbox.top_left[0].toFixed(0)},{block.bbox.top_left[1].toFixed(0)}] →&nbsp;
      [{block.bbox.bottom_right[0].toFixed(0)},{block.bbox.bottom_right[1].toFixed(0)}]
    </p>
  </div>
);

const BlockView: React.FC<BlockViewProps> = ({ page }) => {
  const lowConfidenceCount = page.text_blocks.filter((b) => b.confidence < 0.7).length;

  return (
    <div className="animate-fade-in">
      {/* Legend */}
      <div className="flex flex-wrap gap-3 mb-4 text-xs">
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />신뢰도 ≥ 90%</span>
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-yellow-500 inline-block" />70% – 89%</span>
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-red-500 inline-block" />&lt; 70% ({lowConfidenceCount}개)</span>
      </div>

      {page.text_blocks.length === 0 ? (
        <p className="text-sm text-gray-500 text-center py-8">인식된 텍스트 블록이 없습니다</p>
      ) : (
        <div className="grid gap-2 overflow-auto max-h-[60vh] pr-1">
          {page.text_blocks.map((block) => (
            <BlockCard key={block.block_id} block={block} warn={block.confidence < 0.7} />
          ))}
        </div>
      )}
    </div>
  );
};

export default BlockView;
