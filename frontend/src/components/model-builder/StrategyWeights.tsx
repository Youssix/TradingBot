import type { LearningStatus } from "../../api";

export default function StrategyWeights({ status }: { status: LearningStatus }) {
  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-5">
      <h3 className="mb-3 text-sm font-semibold text-gray-300">
        Strategy Weights
      </h3>
      <div className="space-y-2.5">
        {status.weights.map((w) => (
          <div key={w.name} className="flex items-center gap-3">
            <span className="w-28 truncate text-xs text-gray-400">
              {w.name}
            </span>
            <div className="relative flex-1 h-4 rounded-full bg-gray-700">
              <div
                className="h-4 rounded-full bg-blue-500 transition-all"
                style={{ width: `${Math.round(w.weight * 100)}%` }}
              />
              <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white">
                {Math.round(w.weight * 100)}%
              </span>
            </div>
            <span className="w-20 text-right text-xs text-gray-500">
              WR: {(w.win_rate * 100).toFixed(0)}% ({w.trades})
            </span>
          </div>
        ))}
        {status.weights.length === 0 && (
          <p className="text-xs text-gray-500">No strategies active</p>
        )}
      </div>
    </div>
  );
}
