import { useState, useEffect, useRef } from "react";
import { getLogs, clearLogs } from "../api";

const LEVEL_COLORS: Record<string, string> = {
  ERROR: "text-red-400",
  WARNING: "text-amber-400",
  INFO: "text-emerald-400",
  DEBUG: "text-gray-500",
};

function colorForLine(line: string): string {
  for (const [level, cls] of Object.entries(LEVEL_COLORS)) {
    if (line.includes(` | ${level}`)) return cls;
  }
  return "text-gray-400";
}

export default function LogPanel() {
  const [logs, setLogs] = useState<string[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const data = await getLogs(200);
        setLogs(data);
      } catch {
        // ignore fetch errors
      }
    };
    fetchLogs();
    const interval = setInterval(fetchLogs, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, autoScroll]);

  return (
    <div className="flex h-full flex-col rounded-xl border border-gray-700 bg-gray-800">
      <div className="flex items-center justify-between border-b border-gray-700 px-4 py-2">
        <h2 className="text-sm font-semibold text-gray-300">API Logs</h2>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-xs text-gray-500">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="h-3 w-3 rounded border-gray-600 bg-gray-700"
            />
            Auto-scroll
          </label>
          <button
            onClick={async () => {
              await clearLogs();
              setLogs([]);
            }}
            className="rounded border border-gray-600 px-1.5 py-0.5 text-[10px] text-gray-400 transition hover:border-red-500 hover:text-red-400"
          >
            Clear
          </button>
          <span className="text-xs text-gray-600">{logs.length} lines</span>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-3 font-mono text-[11px] leading-relaxed">
        {logs.length === 0 && (
          <p className="text-gray-600">No logs yet...</p>
        )}
        {logs.map((line, i) => (
          <div key={i} className={`whitespace-pre-wrap break-all ${colorForLine(line)}`}>
            {line}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
