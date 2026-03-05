import type { EnsembleStatsResponse } from "../api";

interface EnsemblePanelProps {
  ensembleStats: EnsembleStatsResponse | null;
}

export default function EnsemblePanel({ ensembleStats }: EnsemblePanelProps) {
  if (!ensembleStats || ensembleStats.strategy === "none") return null;

  return (
    <div className="rounded-2xl p-5 card-glow" style={{ background: "var(--bg-card)", border: "1px solid var(--border-subtle)" }}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
          Ensemble
        </h3>
        <span
          className="rounded-md px-2 py-0.5 text-[10px] font-bold"
          style={{ color: "var(--accent-teal)", background: "rgba(20,184,166,0.1)" }}
        >
          {ensembleStats.strategy}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {ensembleStats.agents.map((agent) => (
          <div
            key={agent.name}
            className="rounded-xl p-3"
            style={{
              background: "var(--bg-elevated)",
              border: agent.is_active ? "1px solid rgba(20,184,166,0.4)" : "1px solid var(--border-subtle)",
            }}
          >
            <p
              className="text-xs font-bold mb-2"
              style={{ color: agent.is_active ? "var(--accent-teal)" : "var(--text-primary)" }}
            >
              {agent.name}
            </p>
            <div className="space-y-1.5 text-[11px]">
              <div className="flex justify-between">
                <span style={{ color: "var(--text-muted)" }}>Sharpe</span>
                <span className="font-num font-semibold" style={{
                  color: agent.sharpe > 1 ? "var(--accent-teal)" : agent.sharpe >= 0 ? "var(--accent-amber)" : "var(--accent-rose)"
                }}>
                  {agent.sharpe.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-muted)" }}>Win Rate</span>
                <span className="font-num font-semibold" style={{ color: "var(--text-primary)" }}>
                  {(agent.win_rate * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--text-muted)" }}>Weight</span>
                <span className="font-num font-semibold" style={{ color: "var(--text-primary)" }}>
                  {(agent.weight * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
