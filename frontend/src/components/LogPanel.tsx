import { useState, useEffect, useRef } from "react";
import { getLogs, clearLogs } from "../api";

const LEVEL_COLORS: Record<string, string> = {
  ERROR: "var(--accent-rose)",
  WARNING: "var(--accent-amber)",
  INFO: "var(--accent-teal)",
  DEBUG: "var(--text-muted)",
};

function colorForLine(line: string): string {
  for (const [level, color] of Object.entries(LEVEL_COLORS)) {
    if (line.includes(` | ${level}`)) return color;
  }
  return "var(--text-secondary)";
}

export default function LogPanel() {
  const [logs, setLogs] = useState<string[]>([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const data = await getLogs(200);
        setLogs(data);
      } catch { /* ignore */ }
    };
    fetchLogs();
    const interval = setInterval(fetchLogs, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  return (
    <div
      className="flex h-full flex-col rounded-2xl overflow-hidden"
      style={{ background: "var(--bg-card)", border: "1px solid var(--border-subtle)" }}
    >
      <div
        className="flex items-center justify-between px-4 py-2.5"
        style={{ borderBottom: "1px solid var(--border-subtle)" }}
      >
        <span className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>Logs</span>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-[10px]" style={{ color: "var(--text-muted)" }}>
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="h-3 w-3 rounded"
            />
            Auto
          </label>
          <button
            onClick={async () => { await clearLogs(); setLogs([]); }}
            className="rounded px-1.5 py-0.5 text-[10px] transition-colors hover:text-rose-400"
            style={{ color: "var(--text-muted)", border: "1px solid var(--border-medium)" }}
          >
            Clear
          </button>
        </div>
      </div>
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-3 font-num text-[10px] leading-relaxed"
      >
        {logs.length === 0 && (
          <p style={{ color: "var(--text-muted)" }}>No logs yet...</p>
        )}
        {logs.map((line, i) => (
          <div key={i} className="whitespace-pre-wrap break-all" style={{ color: colorForLine(line) }}>
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}
