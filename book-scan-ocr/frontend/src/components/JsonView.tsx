/** JSON view tab: syntax-highlighted, collapsible JSON tree via react-json-view. */

import React from 'react';
import ReactJson from 'react-json-view';
import type { PageResult } from '../types';

interface JsonViewProps {
  page: PageResult;
}

const JsonView: React.FC<JsonViewProps> = ({ page }) => {
  return (
    <div className="animate-fade-in">
      <div className="text-xs text-gray-500 mb-3">
        JSON 뷰 — 클릭하여 접기/펼치기
      </div>
      <div className="bg-gray-950/70 rounded-xl border border-gray-800 p-4 overflow-auto max-h-[60vh]">
        <ReactJson
          src={page as unknown as Record<string, unknown>}
          theme="tomorrow"
          collapsed={2}
          displayDataTypes={false}
          displayObjectSize
          enableClipboard
          style={{
            background: 'transparent',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 12,
          }}
        />
      </div>
    </div>
  );
};

export default JsonView;
