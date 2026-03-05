import { useState, useEffect } from "react";
import type { BotStatus } from "../api";
import { setStrategyMode } from "../api";

interface BotControlsProps {
  status: BotStatus | null;
}

const ALL_STRATEGIES = ["ema_crossover", "asian_breakout", "bos", "candle_pattern"];

export default function BotControls({ status }: BotControlsProps) {
  const [strategyMode, setMode] = useState<"independent" | "ensemble">("independent");
  const [enabledStrategies, setEnabledStrategies] = useState<string[]>(ALL_STRATEGIES);

  useEffect(() => {
    if (status) {
      setMode((status.strategy_mode as "independent" | "ensemble") || "independent");
      if (status.enabled_strategies?.length) {
        setEnabledStrategies(status.enabled_strategies);
      }
    }
  }, [status]);

  if (!status) {
    return (
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full" style={{ background: "var(--text-muted)" }} />
        <span className="text-xs" style={{ color: "var(--text-muted)" }}>Connecting...</span>
      </div>
    );
  }

  const isRunning = status.mode === "dry-run" || status.mode === "live";

  const handleModeChange = async (newMode: "independent" | "ensemble") => {
    setMode(newMode);
    await setStrategyMode({ mode: newMode, enabled_strategies: enabledStrategies });
  };

  const modeBg = status.mode === "live"
    ? "rgba(244,63,94,0.15)"
    : status.mode === "dry-run"
      ? "rgba(245,158,11,0.15)"
      : "rgba(59,130,246,0.15)";

  const modeColor = status.mode === "live"
    ? "var(--accent-rose)"
    : status.mode === "dry-run"
      ? "var(--accent-amber)"
      : "var(--accent-blue)";

  return (
    <div className="flex items-center gap-3">
      {/* Status dot + label */}
      <div className="flex items-center gap-1.5">
        <span
          className={`h-2 w-2 rounded-full ${isRunning ? "animate-pulse-dot" : ""}`}
          style={{ background: isRunning ? "var(--accent-teal)" : "var(--accent-rose)" }}
        />
        <span className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
          {isRunning ? "Running" : "Stopped"}
        </span>
      </div>

      {/* Mode badge */}
      <span
        className="rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider"
        style={{ color: modeColor, background: modeBg }}
      >
        {status.mode}
      </span>

      {/* Symbol */}
      <span className="font-num text-xs font-semibold" style={{ color: "var(--text-primary)" }}>
        {status.symbol}
      </span>

      {/* Divider */}
      <span className="h-4 w-px" style={{ background: "var(--border-medium)" }} />

      {/* Strategy mode toggle */}
      <div className="flex rounded-md overflow-hidden" style={{ border: "1px solid var(--border-medium)" }}>
        <button
          onClick={() => handleModeChange("independent")}
          className="px-2.5 py-1 text-[10px] font-medium transition-colors"
          style={{
            background: strategyMode === "independent" ? "var(--bg-elevated)" : "transparent",
            color: strategyMode === "independent" ? "var(--text-primary)" : "var(--text-muted)",
          }}
        >
          Independent
        </button>
        <button
          onClick={() => handleModeChange("ensemble")}
          className="px-2.5 py-1 text-[10px] font-medium transition-colors"
          style={{
            background: strategyMode === "ensemble" ? "var(--bg-elevated)" : "transparent",
            color: strategyMode === "ensemble" ? "var(--text-primary)" : "var(--text-muted)",
          }}
        >
          Ensemble
        </button>
      </div>
    </div>
  );
}
