import type { ClaudeInsight } from "../../api";

export default function ClaudeReviewLog({ insights }: { insights: ClaudeInsight[] }) {
  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-5">
      <h3 className="mb-3 text-sm font-semibold text-gray-300">
        Claude Review Log
      </h3>
      <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
        {insights.map((ins, i) => (
          <div
            key={i}
            className="rounded-lg border border-gray-700/50 bg-gray-900/50 p-3"
          >
            <div className="mb-1 flex items-center gap-2">
              <span
                className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                  ins.review_type === "market_brief"
                    ? "bg-blue-500/20 text-blue-400"
                    : "bg-purple-500/20 text-purple-400"
                }`}
              >
                {ins.review_type}
              </span>
              <span className="text-[10px] text-gray-600">
                {ins.timestamp
                  ? new Date(ins.timestamp).toLocaleString()
                  : "\u2014"}
              </span>
            </div>
            {ins.analysis && (
              <p className="text-xs text-gray-300">{ins.analysis}</p>
            )}
            {ins.market_brief && (
              <p className="mt-1 text-xs italic text-gray-400">
                {ins.market_brief}
              </p>
            )}
            {ins.recommendations.length > 0 && (
              <ul className="mt-1 space-y-0.5">
                {ins.recommendations.map((r, j) => (
                  <li key={j} className="text-[11px] text-gray-500">
                    &bull; {r}
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
        {insights.length === 0 && (
          <p className="text-xs text-gray-500">No insights yet</p>
        )}
      </div>
    </div>
  );
}
