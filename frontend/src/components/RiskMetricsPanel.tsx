import type { RiskMetricsResponse } from "../api";

interface RiskMetricsPanelProps {
  riskMetrics: RiskMetricsResponse | null;
}

export default function RiskMetricsPanel({ riskMetrics }: RiskMetricsPanelProps) {
  if (!riskMetrics?.available) return null;

  return (
    <div className="rounded-2xl p-5 card-glow" style={{ background: "var(--bg-card)", border: "1px solid var(--border-subtle)" }}>
      <h3 className="text-xs font-semibold uppercase tracking-wider mb-4" style={{ color: "var(--text-muted)" }}>
        Risk Metrics
      </h3>
      <div className="space-y-2.5">
        <Row label="CVaR (5%)" value={riskMetrics.cvar_5} color="var(--accent-rose)" />
        <Row label="VaR (5%)" value={riskMetrics.var_5} color="var(--accent-amber)" />
        <Row label="Uncertainty" value={riskMetrics.q_std} color={riskMetrics.q_std > 0.5 ? "var(--accent-rose)" : "var(--accent-amber)"} />
        <Row label="Expected" value={riskMetrics.q_mean} color={riskMetrics.q_mean >= 0 ? "var(--accent-teal)" : "var(--accent-rose)"} />
        <Row label="Upside" value={riskMetrics.upside} color="var(--accent-teal)" />
      </div>
    </div>
  );
}

function Row({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>{label}</span>
      <span className="font-num text-[11px] font-semibold" style={{ color }}>{value.toFixed(4)}</span>
    </div>
  );
}
