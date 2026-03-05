import { useState, useEffect, useCallback } from "react";
import {
  getLearningStatus,
  getClaudeInsights,
  toggleLearning,
  triggerReview,
  triggerOptimization,
  HYPERPARAMS,
} from "../api";
import type { LearningStatus, ClaudeInsight, RLBacktestResult } from "../api";
import AgentSelector from "./model-builder/AgentSelector";
import HyperparameterPanel from "./model-builder/HyperparameterPanel";
import TrainingPanel from "./model-builder/TrainingPanel";
import BacktestResults from "./model-builder/BacktestResults";
import PipelinePanel from "./model-builder/PipelinePanel";
import ModelManager from "./model-builder/ModelManager";
import RLStatusCard from "./model-builder/RLStatusCard";
import MarketContext from "./model-builder/MarketContext";
import StrategyWeights from "./model-builder/StrategyWeights";
import ClaudeReviewLog from "./model-builder/ClaudeReviewLog";

function getDefaults(): Record<string, any> {
  const d: Record<string, any> = {};
  for (const p of HYPERPARAMS) d[p.key] = p.default;
  return d;
}

export default function StrategyBuilder() {
  const [status, setStatus] = useState<LearningStatus | null>(null);
  const [insights, setInsights] = useState<ClaudeInsight[]>([]);
  const [loading, setLoading] = useState("");
  const [backtestResult, setBacktestResult] = useState<RLBacktestResult | null>(null);
  const [modelRefreshKey, setModelRefreshKey] = useState(0);

  // New state for agent selection & hyperparams
  const [selectedAgent, setSelectedAgent] = useState("sac");
  const [ensembleAgents, setEnsembleAgents] = useState<string[]>(["sac", "ppo"]);
  const [hyperparams, setHyperparams] = useState<Record<string, any>>(getDefaults);

  const fetchData = useCallback(async () => {
    const results = await Promise.allSettled([
      getLearningStatus(),
      getClaudeInsights(),
    ]);
    if (results[0].status === "fulfilled") setStatus(results[0].value);
    if (results[1].status === "fulfilled") setInsights(results[1].value);
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10_000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleReview = async () => {
    setLoading("review");
    try {
      await triggerReview();
      await fetchData();
    } finally {
      setLoading("");
    }
  };

  const handleOptimize = async () => {
    setLoading("optimize");
    try {
      await triggerOptimization();
      await fetchData();
    } finally {
      setLoading("");
    }
  };

  const handleToggle = async () => {
    if (!status) return;
    setLoading("toggle");
    try {
      await toggleLearning(!status.enabled);
      await fetchData();
    } finally {
      setLoading("");
    }
  };

  if (!status) {
    return (
      <div className="flex items-center justify-center py-20">
        <p className="text-sm text-gray-500">Loading learning system...</p>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* Enabled banner */}
      {!status.enabled && (
        <div className="rounded-lg border border-yellow-700/50 bg-yellow-900/20 px-4 py-2 text-xs text-yellow-400">
          Learning system is disabled. Click "Enable Learning" below to enable.
        </div>
      )}

      {/* RL Status */}
      <RLStatusCard status={status} />

      {/* Agent Selector */}
      <AgentSelector
        selectedAgent={selectedAgent}
        onSelect={setSelectedAgent}
        ensembleAgents={ensembleAgents}
        onEnsembleAgentsChange={setEnsembleAgents}
      />

      {/* Hyperparameter Panel */}
      {selectedAgent !== "dqn" && (
        <HyperparameterPanel
          agentType={selectedAgent}
          values={hyperparams}
          onChange={(key, value) =>
            setHyperparams((prev) => ({ ...prev, [key]: value }))
          }
          onReset={() => setHyperparams(getDefaults())}
        />
      )}

      {/* Train on History */}
      <TrainingPanel
        agentType={selectedAgent}
        onResult={(r) => {
          setBacktestResult(r);
          setModelRefreshKey((k) => k + 1);
        }}
        loading={loading}
        setLoading={setLoading}
      />

      {/* Pipeline Training */}
      <PipelinePanel
        agentType={selectedAgent}
        loading={loading}
        setLoading={setLoading}
      />

      {/* Backtest Results */}
      <BacktestResults result={backtestResult} />

      {/* Saved Models Manager */}
      <ModelManager refreshKey={modelRefreshKey} />

      {/* Weights + Context row */}
      <div className="grid gap-5 lg:grid-cols-2">
        <StrategyWeights status={status} />
        <MarketContext status={status} />
      </div>

      {/* Claude Review Log */}
      <ClaudeReviewLog insights={insights} />

      {/* Controls */}
      <div className="flex flex-wrap gap-3">
        <button
          onClick={handleReview}
          disabled={loading === "review"}
          className="rounded-lg border border-gray-600 bg-gray-700 px-4 py-2 text-xs font-medium text-gray-200 transition hover:bg-gray-600 disabled:opacity-50"
        >
          {loading === "review" ? "Running..." : "Run Review Now"}
        </button>
        <button
          onClick={handleOptimize}
          disabled={loading === "optimize"}
          className="rounded-lg border border-gray-600 bg-gray-700 px-4 py-2 text-xs font-medium text-gray-200 transition hover:bg-gray-600 disabled:opacity-50"
        >
          {loading === "optimize" ? "Optimizing..." : "Trigger Optimization"}
        </button>
        <button
          onClick={handleToggle}
          disabled={loading === "toggle"}
          className={`rounded-lg border px-4 py-2 text-xs font-medium transition disabled:opacity-50 ${
            status.enabled
              ? "border-red-700/50 bg-red-900/30 text-red-400 hover:bg-red-900/50"
              : "border-green-700/50 bg-green-900/30 text-green-400 hover:bg-green-900/50"
          }`}
        >
          {loading === "toggle"
            ? "..."
            : status.enabled
            ? "Disable Learning"
            : "Enable Learning"}
        </button>
      </div>
    </div>
  );
}
