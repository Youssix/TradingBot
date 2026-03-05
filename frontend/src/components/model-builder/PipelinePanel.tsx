import { useState, useEffect, useRef, useCallback } from "react";
import {
  runPipelineStream,
  listRLModels,
  getPipelineStatus,
  cancelPipeline,
  type RLModelInfo,
} from "../../api";

interface PipelinePanelProps {
  agentType: string;
  loading: string;
  setLoading: (s: string) => void;
}

const PROFILE_OPTIONS = [
  { value: "max_profit", label: "Max Profit" },
  { value: "aggressive", label: "Aggressive" },
  { value: "medium", label: "Medium" },
  { value: "conservative", label: "Conservative" },
] as const;

const PIPELINE_PRESETS = [
  {
    value: "scalping",
    label: "Scalping",
    desc: "M1 only, 15 epochs, Gold, 3x aug — fast scalper",
    color: "text-red-400",
    border: "border-red-500/50",
    bg: "bg-red-500/10",
  },
  {
    value: "quick",
    label: "Quick",
    desc: "H1, 5 epochs, Gold only, 2x aug",
    color: "text-green-400",
    border: "border-green-500/50",
    bg: "bg-green-500/10",
  },
  {
    value: "standard",
    label: "Standard",
    desc: "D1 (Gold+Silver) -> H1 (Gold), augmented",
    color: "text-blue-400",
    border: "border-blue-500/50",
    bg: "bg-blue-500/10",
  },
  {
    value: "thorough",
    label: "Thorough",
    desc: "D1 -> H4 -> H1 -> M15, multi-symbol, max aug",
    color: "text-purple-400",
    border: "border-purple-500/50",
    bg: "bg-purple-500/10",
  },
] as const;

export default function PipelinePanel({
  agentType,
  loading,
  setLoading,
}: PipelinePanelProps) {
  const [expanded, setExpanded] = useState(false);
  const [preset, setPreset] = useState<string>("standard");
  const [profile, setProfile] = useState("max_profit");
  const [modelId, setModelId] = useState<number | null>(null);
  const [savedModels, setSavedModels] = useState<RLModelInfo[]>([]);
  const [pipelineLog, setPipelineLog] = useState<string[]>([]);
  const [pipelineProgress, setPipelineProgress] = useState<{
    step: number;
    totalSteps: number;
    stepTimeframe: string;
    pct: number;
  } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  // On mount: check if a pipeline is already running on the server
  useEffect(() => {
    listRLModels().then((r) => setSavedModels(r.models)).catch(() => {});

    getPipelineStatus().then((s) => {
      if (s.running) {
        setExpanded(true);
        setLoading("pipeline");
        setPipelineLog(s.logs);
        if (s.progress) {
          setPipelineProgress({
            step: s.progress.step,
            totalSteps: s.progress.total_steps,
            stepTimeframe: s.progress.step_timeframe,
            pct: s.progress.pct,
          });
        }
      }
    }).catch(() => {});
  }, []);

  // Poll pipeline status while running
  useEffect(() => {
    if (loading !== "pipeline") return;
    const iv = setInterval(() => {
      getPipelineStatus().then((s) => {
        if (s.logs.length > 0) setPipelineLog(s.logs);
        if (s.progress) {
          setPipelineProgress({
            step: s.progress.step,
            totalSteps: s.progress.total_steps,
            stepTimeframe: s.progress.step_timeframe,
            pct: s.progress.pct,
          });
        }
        if (!s.running) {
          setPipelineProgress(null);
          setLoading("");
          clearInterval(iv);
        }
      }).catch(() => {});
    }, 2000);
    return () => clearInterval(iv);
  }, [loading]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [pipelineLog]);

  const isPipelineRunning = loading === "pipeline";

  const handleRunPipeline = () => {
    setLoading("pipeline");
    setPipelineLog([]);
    setPipelineProgress(null);

    const controller = runPipelineStream(
      {
        preset,
        profile,
        train_every: 5,
        model_id: modelId,
      },
      {
        onStepStart: (d) => {
          setPipelineLog((prev) => [
            ...prev,
            `--- Step ${d.step + 1}/${d.total_steps}: ${d.timeframe} | ${d.symbols.join(", ")} | ${d.epochs} epochs | aug=${d.augmentation_factor}x ---`,
          ]);
          setPipelineProgress({
            step: d.step,
            totalSteps: d.total_steps,
            stepTimeframe: d.timeframe,
            pct: (d.step / d.total_steps) * 100,
          });
        },
        onProgress: (d) => {
          setPipelineProgress((prev) => ({
            step: d.step,
            totalSteps: prev?.totalSteps ?? 1,
            stepTimeframe: prev?.stepTimeframe ?? "",
            pct: ((d.step + d.pct / 100) / (prev?.totalSteps ?? 1)) * 100,
          }));
        },
        onEpoch: (d) => {
          const wr = (d.win_rate * 100).toFixed(0);
          setPipelineLog((prev) => [
            ...prev,
            `  Epoch ${d.epoch}: ${d.trades} trades | WR ${wr}% | reward ${d.total_reward.toFixed(2)} | eps ${d.epsilon.toFixed(4)}`,
          ]);
        },
        onStepDone: (d) => {
          setPipelineLog((prev) => [
            ...prev,
            `  Step done: ${d.timeframe} | ${d.trades} trades | WR ${(d.win_rate * 100).toFixed(1)}% | checkpoint saved`,
          ]);
        },
        onPipelineDone: (d) => {
          const msg = d.cancelled
            ? `--- Pipeline cancelled after ${d.steps_completed} steps (partial checkpoint saved) ---`
            : `--- Pipeline complete: ${d.steps_completed} steps | ${d.total_episodes} episodes | eps ${d.final_epsilon.toFixed(4)}${d.final_win_rate !== undefined ? ` | WR ${(d.final_win_rate * 100).toFixed(1)}%` : ""} ---`;
          setPipelineLog((prev) => [...prev, msg]);
          setPipelineProgress(null);
          setLoading("");
        },
        onError: (msg) => {
          setPipelineLog((prev) => [...prev, `ERROR: ${msg}`]);
          setPipelineProgress(null);
          setLoading("");
        },
      },
    );
    abortRef.current = controller;
  };

  const handleCancelPipeline = () => {
    // Abort local SSE stream if we started it
    abortRef.current?.abort();
    abortRef.current = null;
    // Tell server to cancel (it saves a checkpoint)
    cancelPipeline().catch(() => {});
    setPipelineLog((prev) => [...prev, "--- Pipeline cancelling (saving checkpoint)... ---"]);
  };

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between p-5"
      >
        <h3 className="text-sm font-semibold text-gray-300">
          Pipeline Training (Multi-Step)
        </h3>
        <span className="text-xs text-gray-500">{expanded ? "collapse" : "expand"}</span>
      </button>

      {expanded && (
        <div className="border-t border-gray-700 p-5 pt-4 space-y-4">
          {/* Preset selector */}
          <div>
            <label className="mb-1.5 block text-[10px] uppercase tracking-wider text-gray-500">
              Pipeline Preset
            </label>
            <div className="flex gap-2">
              {PIPELINE_PRESETS.map((p) => (
                <button
                  key={p.value}
                  onClick={() => !isPipelineRunning && setPreset(p.value)}
                  className={`flex-1 rounded-lg border px-3 py-2 text-left transition ${
                    preset === p.value
                      ? `${p.border} ${p.bg}`
                      : "border-gray-700 bg-gray-900/50 hover:border-gray-600"
                  } ${isPipelineRunning ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  <span className={`block text-xs font-semibold ${preset === p.value ? p.color : "text-gray-400"}`}>
                    {p.label}
                  </span>
                  <span className="block text-[10px] text-gray-500">{p.desc}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Profile + Model selector row */}
          <div className="flex gap-3">
            <div className="flex-1">
              <label className="mb-1 block text-[10px] uppercase tracking-wider text-gray-500">
                Profile
              </label>
              <select
                value={profile}
                onChange={(e) => setProfile(e.target.value)}
                disabled={isPipelineRunning}
                className="w-full rounded-lg border border-gray-600 bg-gray-700 px-3 py-1.5 text-xs text-white disabled:opacity-50"
              >
                {PROFILE_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>
            <div className="flex-1">
              <label className="mb-1 block text-[10px] uppercase tracking-wider text-gray-500">
                Continue from Model
              </label>
              <select
                value={modelId ?? ""}
                onChange={(e) => setModelId(e.target.value ? Number(e.target.value) : null)}
                disabled={isPipelineRunning}
                className="w-full rounded-lg border border-gray-600 bg-gray-700 px-3 py-1.5 text-xs text-white disabled:opacity-50"
              >
                <option value="">Start fresh</option>
                {savedModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    #{m.id} {m.profile} | ep {m.episode} | WR {(m.win_rate * 100).toFixed(1)}%
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Run / Cancel */}
          <div className="flex gap-3">
            {isPipelineRunning ? (
              <button
                onClick={handleCancelPipeline}
                className="rounded-lg border border-red-600 bg-red-700 px-4 py-1.5 text-xs font-medium text-white transition hover:bg-red-600"
              >
                Cancel Pipeline
              </button>
            ) : (
              <button
                onClick={handleRunPipeline}
                disabled={!!loading}
                className="rounded-lg border border-purple-600 bg-purple-700 px-4 py-1.5 text-xs font-medium text-white transition hover:bg-purple-600 disabled:opacity-50"
              >
                Run Pipeline
              </button>
            )}
          </div>

          {/* Pipeline progress */}
          {(isPipelineRunning || pipelineLog.length > 0) && (
            <div className="space-y-2">
              {pipelineProgress && (
                <div className="flex items-center gap-2">
                  <span className="w-32 text-[10px] text-gray-500">
                    Step {pipelineProgress.step + 1}/{pipelineProgress.totalSteps} ({pipelineProgress.stepTimeframe})
                  </span>
                  <div className="relative flex-1 h-3 rounded-full bg-gray-700 overflow-hidden">
                    <div
                      className="h-3 rounded-full bg-purple-500 transition-all duration-300"
                      style={{ width: `${Math.min(pipelineProgress.pct, 100)}%` }}
                    />
                    <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold text-white">
                      {Math.round(pipelineProgress.pct)}%
                    </span>
                  </div>
                </div>
              )}

              <div className="max-h-48 overflow-y-auto rounded-lg bg-gray-900/70 p-2.5 font-mono text-[11px] text-gray-400">
                {pipelineLog.map((line, i) => (
                  <div
                    key={i}
                    className={
                      line.startsWith("ERROR")
                        ? "text-red-400"
                        : line.startsWith("---")
                        ? "text-purple-400 font-semibold"
                        : ""
                    }
                  >
                    {line}
                  </div>
                ))}
                {isPipelineRunning && pipelineLog.length === 0 && (
                  <div className="text-gray-600 animate-pulse">Starting pipeline...</div>
                )}
                <div ref={logEndRef} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
