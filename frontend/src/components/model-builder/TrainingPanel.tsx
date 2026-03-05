import { useState, useEffect, useRef } from "react";
import {
  runRLBacktestStream,
  listRLModels,
  type RLBacktestResult,
  type RLBacktestEpochStats,
  type RLModelInfo,
} from "../../api";

interface TrainingPanelProps {
  agentType: string;
  onResult: (result: RLBacktestResult) => void;
  loading: string;
  setLoading: (s: string) => void;
}

const PROFILE_OPTIONS = [
  {
    value: "max_profit",
    label: "Max Profit",
    desc: "Profit-optimized: 1.2x SL, 3x TP, hold 20 bars, 3% risk",
    color: "text-blue-400",
    border: "border-blue-500/50",
    bg: "bg-blue-500/10",
  },
  {
    value: "aggressive",
    label: "Aggressive",
    desc: "High frequency, 1x ATR SL, 2x TP, 12-bar hold, 2% risk",
    color: "text-red-400",
    border: "border-red-500/50",
    bg: "bg-red-500/10",
  },
  {
    value: "medium",
    label: "Medium",
    desc: "Balanced, 1.5x ATR stops, 1.5% risk/trade",
    color: "text-yellow-400",
    border: "border-yellow-500/50",
    bg: "bg-yellow-500/10",
  },
  {
    value: "conservative",
    label: "Conservative",
    desc: "Selective, wide stops, 0.5% risk/trade",
    color: "text-green-400",
    border: "border-green-500/50",
    bg: "bg-green-500/10",
  },
] as const;

const SYMBOL_OPTIONS = [
  { value: "GC=F", label: "Gold", emoji: "Au" },
  { value: "SI=F", label: "Silver", emoji: "Ag" },
  { value: "CL=F", label: "Oil", emoji: "CL" },
  { value: "ES=F", label: "S&P500", emoji: "ES" },
] as const;

export default function TrainingPanel({
  agentType,
  onResult,
  loading,
  setLoading,
}: TrainingPanelProps) {
  const [timeframe, setTimeframe] = useState("H1");
  const [count, setCount] = useState(5000);
  const [epochs, setEpochs] = useState(10);
  const [profile, setProfile] = useState("max_profit");
  const [modelId, setModelId] = useState<number | null>(null);
  const [savedModels, setSavedModels] = useState<RLModelInfo[]>([]);
  const [symbols, setSymbols] = useState<string[]>(["GC=F"]);
  const [augFactor, setAugFactor] = useState(0);
  const [trainingLog, setTrainingLog] = useState<string[]>([]);
  const [progress, setProgress] = useState<{
    pct: number;
    epoch: number;
    totalEpochs: number;
    barPct: number;
  } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listRLModels().then((r) => setSavedModels(r.models)).catch(() => {});
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [trainingLog]);

  const toggleSymbol = (sym: string) => {
    setSymbols((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
    );
  };

  const estimatedBars = count * symbols.length * (1 + augFactor);
  const selectedModel = savedModels.find((m) => m.id === modelId);

  const handleTrain = () => {
    if (symbols.length === 0) return;
    setLoading("backtest");
    setTrainingLog([]);
    setProgress({ pct: 0, epoch: 0, totalEpochs: epochs, barPct: 0 });

    const controller = runRLBacktestStream(
      {
        timeframe,
        count,
        epochs,
        train_every: 5,
        profile,
        model_id: modelId,
        symbols,
        augmentation_factor: augFactor,
        agent_type: agentType,
      },
      {
        onProgress: (d) => {
          setProgress((prev) => ({
            pct: prev ? (prev.epoch / epochs) * 100 + (d.pct / epochs) : d.pct,
            epoch: d.epoch,
            totalEpochs: epochs,
            barPct: d.pct,
          }));
        },
        onEpoch: (d) => {
          const wr = (d.win_rate * 100).toFixed(0);
          const pnl = (d as RLBacktestEpochStats & { cumulative_pnl: number }).cumulative_pnl;
          const line = `Epoch ${d.epoch}: ${d.trades} trades | WR ${wr}% | PnL ${pnl >= 0 ? "+" : ""}${pnl.toFixed(2)} | eps ${d.epsilon.toFixed(4)}`;
          setTrainingLog((prev) => [...prev, line]);
          setProgress({
            pct: ((d.epoch + 1) / epochs) * 100,
            epoch: d.epoch + 1,
            totalEpochs: epochs,
            barPct: 100,
          });
        },
        onDone: (data) => {
          setTrainingLog((prev) => [...prev, `--- Training complete: ${data.total_trades} trades, WR ${(data.win_rate * 100).toFixed(1)}% ---`]);
          setProgress(null);
          setLoading("");
          onResult(data);
        },
        onError: (msg) => {
          setTrainingLog((prev) => [...prev, `ERROR: ${msg}`]);
          setProgress(null);
          setLoading("");
        },
      },
    );
    abortRef.current = controller;
  };

  const handleCancel = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setTrainingLog((prev) => [...prev, "--- Training cancelled ---"]);
    setProgress(null);
    setLoading("");
  };

  const isTraining = loading === "backtest";

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-5">
      <h3 className="mb-3 text-sm font-semibold text-gray-300">
        Train on Historical Data
      </h3>

      {/* Profile selector */}
      <div className="mb-4">
        <label className="mb-1.5 block text-[10px] uppercase tracking-wider text-gray-500">
          Strategy Profile
        </label>
        <div className="flex gap-2">
          {PROFILE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => !isTraining && setProfile(opt.value)}
              className={`flex-1 rounded-lg border px-3 py-2 text-left transition ${
                profile === opt.value
                  ? `${opt.border} ${opt.bg}`
                  : "border-gray-700 bg-gray-900/50 hover:border-gray-600"
              } ${isTraining ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              <span
                className={`block text-xs font-semibold ${
                  profile === opt.value ? opt.color : "text-gray-400"
                }`}
              >
                {opt.label}
              </span>
              <span className="block text-[10px] text-gray-500">{opt.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Continue from Model */}
      <div className="mb-4">
        <label className="mb-1.5 block text-[10px] uppercase tracking-wider text-gray-500">
          Continue from Model
        </label>
        <select
          value={modelId ?? ""}
          onChange={(e) => setModelId(e.target.value ? Number(e.target.value) : null)}
          disabled={isTraining}
          className="w-full rounded-lg border border-gray-600 bg-gray-700 px-3 py-1.5 text-xs text-white disabled:opacity-50"
        >
          <option value="">Start fresh (new model)</option>
          {savedModels.map((m) => (
            <option key={m.id} value={m.id}>
              #{m.id} {m.profile} | ep {m.episode} | WR {(m.win_rate * 100).toFixed(1)}% | eps {m.epsilon.toFixed(4)} | {m.timeframe}
            </option>
          ))}
        </select>
        {selectedModel && (
          <div className="mt-1 flex gap-3 text-[10px] text-gray-500">
            <span>Episode: {selectedModel.episode}</span>
            <span>Epsilon: {selectedModel.epsilon.toFixed(4)}</span>
            <span>Reward: {selectedModel.total_reward.toFixed(1)}</span>
          </div>
        )}
      </div>

      {/* Symbols */}
      <div className="mb-4">
        <label className="mb-1.5 block text-[10px] uppercase tracking-wider text-gray-500">
          Symbols
        </label>
        <div className="flex gap-2">
          {SYMBOL_OPTIONS.map((s) => (
            <button
              key={s.value}
              onClick={() => !isTraining && toggleSymbol(s.value)}
              disabled={isTraining}
              className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
                symbols.includes(s.value)
                  ? "border-blue-500/50 bg-blue-500/10 text-blue-400"
                  : "border-gray-700 bg-gray-900/50 text-gray-500 hover:border-gray-600"
              } disabled:opacity-50`}
            >
              <span className="mr-1 font-bold">{s.emoji}</span>
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* Data Augmentation slider */}
      <div className="mb-4">
        <div className="mb-1.5 flex items-center justify-between">
          <label className="text-[10px] uppercase tracking-wider text-gray-500">
            Data Augmentation
          </label>
          <span className="text-[10px] text-gray-400">
            {augFactor === 0 ? "Off" : `${augFactor}x`}
            {" "}
            <span className="text-gray-600">
              (~{estimatedBars.toLocaleString()} total bars)
            </span>
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={5}
          value={augFactor}
          onChange={(e) => setAugFactor(Number(e.target.value))}
          disabled={isTraining}
          className="w-full accent-blue-500 disabled:opacity-50"
        />
        <div className="flex justify-between text-[9px] text-gray-600">
          <span>Off</span>
          <span>1x</span>
          <span>2x</span>
          <span>3x</span>
          <span>4x</span>
          <span>5x</span>
        </div>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="mb-1 block text-[10px] uppercase tracking-wider text-gray-500">
            Timeframe
          </label>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            disabled={isTraining}
            className="rounded-lg border border-gray-600 bg-gray-700 px-3 py-1.5 text-xs text-white disabled:opacity-50"
          >
            <option value="M1">M1 - Scalping (5 days)</option>
            <option value="M5">M5 (60 days)</option>
            <option value="M15">M15 (60 days)</option>
            <option value="H1">H1 (2 years)</option>
            <option value="H4">H4 (2 years)</option>
            <option value="D1">D1 (all)</option>
          </select>
        </div>
        <div>
          <label className="mb-1 block text-[10px] uppercase tracking-wider text-gray-500">
            Candles
          </label>
          <input
            type="number"
            value={count}
            onChange={(e) => setCount(Number(e.target.value))}
            min={100}
            max={10000}
            disabled={isTraining}
            className="w-24 rounded-lg border border-gray-600 bg-gray-700 px-3 py-1.5 text-xs text-white disabled:opacity-50"
          />
        </div>
        <div>
          <label className="mb-1 block text-[10px] uppercase tracking-wider text-gray-500">
            Epochs
          </label>
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
            min={1}
            max={20}
            disabled={isTraining}
            className="w-16 rounded-lg border border-gray-600 bg-gray-700 px-3 py-1.5 text-xs text-white disabled:opacity-50"
          />
        </div>
        {isTraining ? (
          <button
            onClick={handleCancel}
            className="rounded-lg border border-red-600 bg-red-700 px-4 py-1.5 text-xs font-medium text-white transition hover:bg-red-600"
          >
            Cancel Training
          </button>
        ) : (
          <button
            onClick={handleTrain}
            disabled={!!loading || symbols.length === 0}
            className="rounded-lg border border-blue-600 bg-blue-700 px-4 py-1.5 text-xs font-medium text-white transition hover:bg-blue-600 disabled:opacity-50"
          >
            Train on Historical Data
          </button>
        )}
      </div>

      {/* Live training progress */}
      {(isTraining || trainingLog.length > 0) && (
        <div className="mt-4 space-y-2">
          {progress && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="w-28 text-[10px] text-gray-500">
                  Overall: Epoch {progress.epoch}/{progress.totalEpochs}
                </span>
                <div className="relative flex-1 h-3 rounded-full bg-gray-700 overflow-hidden">
                  <div
                    className="h-3 rounded-full bg-blue-500 transition-all duration-300"
                    style={{ width: `${Math.min(progress.pct, 100)}%` }}
                  />
                  <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold text-white">
                    {Math.round(progress.pct)}%
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-28 text-[10px] text-gray-500">
                  Epoch bars
                </span>
                <div className="relative flex-1 h-2 rounded-full bg-gray-700 overflow-hidden">
                  <div
                    className="h-2 rounded-full bg-cyan-600 transition-all duration-300"
                    style={{ width: `${Math.min(progress.barPct, 100)}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          <div className="max-h-40 overflow-y-auto rounded-lg bg-gray-900/70 p-2.5 font-mono text-[11px] text-gray-400">
            {trainingLog.map((line, i) => (
              <div key={i} className={line.startsWith("ERROR") ? "text-red-400" : line.startsWith("---") ? "text-blue-400 font-semibold" : ""}>
                {line}
              </div>
            ))}
            {isTraining && trainingLog.length === 0 && (
              <div className="text-gray-600 animate-pulse">Fetching data and starting training...</div>
            )}
            <div ref={logEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}
