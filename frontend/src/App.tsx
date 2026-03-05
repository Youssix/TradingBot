import { useState, useEffect, useCallback } from "react";
import {
  getAccount,
  getCandles,
  getClosedTrades,
  getOpenTrades,
  getBotStatus,
  getTrainingMetrics,
  getEnsembleStats,
  getRiskMetrics,
  getLearningStatus,
  type AccountInfo,
  type Candle,
  type Trade,
  type BotStatus,
  type TrainingMetricsResponse,
  type EnsembleStatsResponse,
  type RiskMetricsResponse,
  type LearningStatus,
} from "./api";
import KPICards from "./components/KPICards";
import CandlestickChart from "./components/CandlestickChart";
import EquityCurve from "./components/EquityCurve";
import TradesTable from "./components/TradesTable";
import BacktestPanel from "./components/BacktestPanel";
import BotControls from "./components/BotControls";
import TabNav from "./components/TabNav";
import StrategyBuilder from "./components/StrategyBuilder";
import LogPanel from "./components/LogPanel";
import AgentStatusPanel from "./components/AgentStatusPanel";
import EnsemblePanel from "./components/EnsemblePanel";
import RiskMetricsPanel from "./components/RiskMetricsPanel";

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [account, setAccount] = useState<AccountInfo | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [closedTrades, setClosedTrades] = useState<Trade[]>([]);
  const [openTrades, setOpenTrades] = useState<Trade[]>([]);
  const [botStatus, setBotStatus] = useState<BotStatus | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetricsResponse | null>(null);
  const [ensembleStats, setEnsembleStats] = useState<EnsembleStatsResponse | null>(null);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetricsResponse | null>(null);
  const [learningStatus, setLearningStatus] = useState<LearningStatus | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchAll = useCallback(async () => {
    const results = await Promise.allSettled([
      getAccount(),
      getCandles(),
      getClosedTrades(),
      getOpenTrades(),
      getBotStatus(),
      getTrainingMetrics(),
      getEnsembleStats(),
      getRiskMetrics(),
      getLearningStatus(),
    ]);

    if (results[0].status === "fulfilled") setAccount(results[0].value);
    if (results[1].status === "fulfilled") setCandles(results[1].value);
    if (results[2].status === "fulfilled") setClosedTrades(results[2].value);
    if (results[3].status === "fulfilled") setOpenTrades(results[3].value);
    if (results[4].status === "fulfilled") setBotStatus(results[4].value);
    if (results[5].status === "fulfilled") setTrainingMetrics(results[5].value);
    if (results[6].status === "fulfilled") setEnsembleStats(results[6].value);
    if (results[7].status === "fulfilled") setRiskMetrics(results[7].value);
    if (results[8].status === "fulfilled") setLearningStatus(results[8].value);
    setLastUpdate(new Date());
  }, []);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 10000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  const showEnsemble = ensembleStats && ensembleStats.strategy !== "none";
  const showRisk = riskMetrics?.available;

  return (
    <div className="min-h-screen" style={{ background: "var(--bg-base)" }}>
      {/* Header */}
      <header className="sticky top-0 z-30 border-b backdrop-blur-xl" style={{ borderColor: "var(--border-subtle)", background: "rgba(11,15,25,0.85)" }}>
        <div className="mx-auto flex max-w-[1800px] items-center justify-between px-5 py-3">
          <div className="flex items-center gap-5">
            <h1 className="text-lg font-semibold tracking-tight" style={{ fontFamily: "var(--font-sans)" }}>
              <span className="text-gradient">TradingBot</span>
            </h1>
            <BotControls status={botStatus} />
          </div>
          <div className="flex items-center gap-3">
            <span className="font-num text-[11px]" style={{ color: "var(--text-muted)" }}>
              {lastUpdate.toLocaleTimeString()}
            </span>
            <button
              onClick={fetchAll}
              className="rounded-lg px-3 py-1.5 text-xs font-medium transition-all hover:bg-white/5"
              style={{ color: "var(--text-secondary)", border: "1px solid var(--border-medium)" }}
            >
              Refresh
            </button>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <TabNav activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Main Content */}
      <div className="mx-auto flex max-w-[1800px] gap-5 px-5 py-5">
        {/* Left: Log Panel */}
        <aside className="hidden w-72 shrink-0 xl:block" style={{ height: "calc(100vh - 130px)" }}>
          <LogPanel />
        </aside>

        {/* Main area */}
        <main className="min-w-0 flex-1 space-y-5">
          {activeTab === "dashboard" && (
            <>
              {/* Hero KPIs + Agent Status */}
              <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
                <KPICards
                  account={account}
                  closedTrades={closedTrades}
                  trainingMetrics={trainingMetrics}
                  learningStatus={learningStatus}
                />
                <AgentStatusPanel trainingMetrics={trainingMetrics} />
              </div>

              {/* Charts */}
              <div className="grid gap-5 lg:grid-cols-2">
                <CandlestickChart candles={candles} />
                <EquityCurve trades={closedTrades} />
              </div>

              {/* Optional panels: only show when active */}
              {(showEnsemble || showRisk) && (
                <div className={`grid gap-5 ${showEnsemble && showRisk ? "lg:grid-cols-2" : ""}`}>
                  {showEnsemble && <EnsemblePanel ensembleStats={ensembleStats} />}
                  {showRisk && <RiskMetricsPanel riskMetrics={riskMetrics} />}
                </div>
              )}

              {/* Trades */}
              <TradesTable closedTrades={closedTrades} openTrades={openTrades} />

              <BacktestPanel />
            </>
          )}

          {activeTab === "strategy-builder" && <StrategyBuilder />}

          {activeTab === "logs" && <LogPanel />}
        </main>
      </div>
    </div>
  );
}
