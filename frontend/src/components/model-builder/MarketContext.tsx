import type { LearningStatus } from "../../api";

export default function MarketContext({ status }: { status: LearningStatus }) {
  const regimeBadge: Record<string, string> = {
    trending: "bg-green-500/20 text-green-400",
    ranging: "bg-yellow-500/20 text-yellow-400",
    volatile: "bg-red-500/20 text-red-400",
    unknown: "bg-gray-600/30 text-gray-400",
  };

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-5">
      <h3 className="mb-3 text-sm font-semibold text-gray-300">
        Market Context
      </h3>
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Timeframe:</span>
          <span className="rounded bg-gray-700 px-2 py-0.5 text-xs font-medium text-white">
            {status.timeframe}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Context:</span>
          <span className="text-xs text-gray-300">
            {status.context_timeframes.join(", ") || "\u2014"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Regime:</span>
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
              regimeBadge[status.regime] || regimeBadge.unknown
            }`}
          >
            {status.regime.toUpperCase()}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Session:</span>
          <span className="text-xs capitalize text-gray-300">
            {status.session}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Conf. Threshold:</span>
          <span className="text-xs font-medium text-white">
            {status.confidence_threshold.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}
