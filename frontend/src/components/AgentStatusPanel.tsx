import { useState, useEffect, useCallback } from "react";
import {
  listRLModels,
  activateRLModel,
  type TrainingMetricsResponse,
  type RLModelInfo,
} from "../api";

interface AgentStatusPanelProps {
  trainingMetrics: TrainingMetricsResponse | null;
}

const READINESS: Record<string, { color: string; bg: string; label: string }> = {
  untrained: { color: "#f43f5e", bg: "rgba(244,63,94,0.1)", label: "Untrained" },
  learning: { color: "#f59e0b", bg: "rgba(245,158,11,0.1)", label: "Learning" },
  ready: { color: "#14b8a6", bg: "rgba(20,184,166,0.1)", label: "Ready" },
};

export default function AgentStatusPanel({ trainingMetrics }: AgentStatusPanelProps) {
  const [models, setModels] = useState<RLModelInfo[]>([]);
  const [showPicker, setShowPicker] = useState(false);
  const [activating, setActivating] = useState<number | null>(null);

  const fetchModels = useCallback(async () => {
    try {
      const resp = await listRLModels();
      setModels(resp.models);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    if (showPicker) fetchModels();
  }, [showPicker, fetchModels]);

  const handleActivate = async (id: number) => {
    setActivating(id);
    try {
      await activateRLModel(id);
      setShowPicker(false);
    } catch (e) {
      console.error("Activate failed:", e);
    } finally {
      setActivating(null);
    }
  };

  if (!trainingMetrics) {
    return (
      <div className="rounded-2xl p-5" style={{ background: "var(--bg-card)", border: "1px solid var(--border-subtle)" }}>
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>No agent data</p>
      </div>
    );
  }

  const m = trainingMetrics;
  const r = READINESS[m.readiness] ?? READINESS.untrained;
  const exploitPct = Math.round((1 - m.epsilon) * 100);

  return (
    <div className="rounded-2xl p-5 space-y-4 card-glow" style={{ background: "var(--bg-card)", border: "1px solid var(--border-subtle)" }}>
      {/* Header: Agent type + Readiness */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className="font-num rounded-md px-2 py-0.5 text-[11px] font-bold uppercase"
            style={{ color: r.color, background: r.bg }}
          >
            {r.label}
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full ${m.training ? "animate-pulse-dot" : ""}`}
              style={{ background: m.training ? "var(--accent-teal)" : "var(--text-muted)" }}
            />
            <span className="text-[11px] font-medium" style={{ color: "var(--text-secondary)" }}>
              {m.training ? "Training" : "Idle"}
            </span>
          </span>
        </div>
        <span
          className="font-num rounded-md px-2 py-0.5 text-[11px] font-bold uppercase"
          style={{ color: "var(--text-secondary)", background: "var(--bg-elevated)" }}
        >
          {m.agent_type}
        </span>
      </div>

      {/* Active Model */}
      <div className="rounded-xl p-3" style={{ background: "var(--bg-elevated)" }}>
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] font-medium uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
            Active Model
          </span>
          <button
            onClick={() => setShowPicker(!showPicker)}
            className="rounded px-2 py-0.5 text-[10px] font-medium transition-all hover:bg-white/5"
            style={{ color: "var(--accent-teal)", border: "1px solid var(--border-medium)" }}
          >
            {showPicker ? "Cancel" : "Switch"}
          </button>
        </div>
        {m.active_model_id != null ? (
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-num text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              #{m.active_model_id}
            </span>
            {m.active_model_name && (
              <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                {m.active_model_name}
              </span>
            )}
            {m.active_model_timeframe && (
              <span
                className="rounded px-1.5 py-0.5 text-[10px] font-medium"
                style={{ color: "var(--accent-blue)", background: "rgba(59,130,246,0.1)" }}
              >
                {m.active_model_timeframe}
              </span>
            )}
          </div>
        ) : (
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>No model loaded</p>
        )}
      </div>

      {/* Model Picker */}
      {showPicker && (
        <div
          className="max-h-44 space-y-1 overflow-y-auto rounded-xl p-2"
          style={{ background: "var(--bg-base)", border: "1px solid var(--border-medium)" }}
        >
          {models.length === 0 && (
            <p className="py-2 text-center text-xs" style={{ color: "var(--text-muted)" }}>No saved models</p>
          )}
          {models.map((model) => {
            const isActive = model.id === m.active_model_id;
            return (
              <button
                key={model.id}
                onClick={() => !isActive && handleActivate(model.id)}
                disabled={isActive || activating === model.id}
                className="w-full flex items-center justify-between rounded-lg px-3 py-2 text-left transition-all disabled:opacity-50"
                style={{
                  background: isActive ? "rgba(20,184,166,0.08)" : "transparent",
                  border: isActive ? "1px solid rgba(20,184,166,0.3)" : "1px solid transparent",
                }}
                onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = "var(--bg-elevated)"; }}
                onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = "transparent"; }}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="font-num text-[11px] font-semibold truncate" style={{ color: "var(--text-primary)" }}>
                    {model.model_name}
                  </span>
                  <span className="font-num text-[10px]" style={{ color: "var(--text-muted)" }}>
                    ep {model.episode.toLocaleString()}
                  </span>
                  <span className="font-num text-[10px]" style={{ color: "var(--text-muted)" }}>
                    WR {(model.win_rate * 100).toFixed(0)}%
                  </span>
                </div>
                <span className="font-num shrink-0 text-[10px] font-semibold" style={{ color: isActive ? "var(--accent-teal)" : "var(--accent-blue)" }}>
                  {isActive ? "Active" : activating === model.id ? "..." : "Use"}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-x-3 gap-y-2">
        <StatRow label="Episode" value={m.episode.toLocaleString()} />
        <StatRow label="Win Rate" value={`${(m.win_rate * 100).toFixed(1)}%`} />
        <StatRow label="Reward" value={m.total_reward.toFixed(1)} color={m.total_reward >= 0 ? "var(--accent-teal)" : "var(--accent-rose)"} />
        {m.alpha !== null && <StatRow label="Alpha" value={m.alpha.toFixed(3)} />}
      </div>

      {/* Exploitation bar */}
      <div>
        <div className="flex justify-between mb-1">
          <span className="text-[10px] font-medium" style={{ color: "var(--text-muted)" }}>Exploitation</span>
          <span className="font-num text-[10px] font-semibold" style={{ color: "var(--text-secondary)" }}>{exploitPct}%</span>
        </div>
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "var(--bg-elevated)" }}>
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${exploitPct}%`,
              background: exploitPct < 30 ? "var(--accent-rose)" : exploitPct < 70 ? "var(--accent-amber)" : "var(--accent-teal)",
            }}
          />
        </div>
      </div>

      {/* Badges */}
      {m.use_per && (
        <span
          className="inline-block rounded-md px-2 py-0.5 text-[10px] font-semibold"
          style={{ color: "var(--accent-blue)", background: "rgba(59,130,246,0.1)" }}
        >
          PER Active
        </span>
      )}
    </div>
  );
}

function StatRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>{label}</span>
      <span className="font-num text-[11px] font-semibold" style={{ color: color ?? "var(--text-primary)" }}>{value}</span>
    </div>
  );
}
