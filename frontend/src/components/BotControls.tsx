import type { BotStatus } from "../api";

interface BotControlsProps {
  status: BotStatus | null;
}

function modeBadgeColor(mode: string): string {
  switch (mode.toLowerCase()) {
    case "live":
      return "bg-red-900/50 text-red-300 border-red-700";
    case "dry-run":
      return "bg-amber-900/50 text-amber-300 border-amber-700";
    case "backtest":
      return "bg-blue-900/50 text-blue-300 border-blue-700";
    default:
      return "bg-gray-700 text-gray-300 border-gray-600";
  }
}

export default function BotControls({ status }: BotControlsProps) {
  if (!status) {
    return (
      <div className="flex items-center gap-3">
        <span className="h-2.5 w-2.5 rounded-full bg-gray-600" />
        <span className="text-sm text-gray-500">Connecting...</span>
      </div>
    );
  }

  const isRunning = status.mode === "dry-run" || status.mode === "live";

  return (
    <div className="flex flex-wrap items-center gap-4">
      <div className="flex items-center gap-2">
        <span
          className={`h-2.5 w-2.5 rounded-full ${
            isRunning ? "bg-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.6)]" : "bg-red-400"
          }`}
        />
        <span className="text-sm font-medium text-gray-300">
          {isRunning ? "Running" : "Stopped"}
        </span>
      </div>

      <span
        className={`rounded-full border px-3 py-0.5 text-xs font-semibold uppercase ${modeBadgeColor(
          status.mode
        )}`}
      >
        {status.mode}
      </span>

      <span className="text-sm font-medium text-white">{status.symbol}</span>

      {status.strategies.length > 0 && (
        <div className="flex gap-1.5">
          {status.strategies.map((s) => (
            <span
              key={s}
              className="rounded-md bg-gray-700 px-2 py-0.5 text-xs text-gray-300"
            >
              {s}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
