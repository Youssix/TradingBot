import { useState, useEffect, useCallback } from "react";
import {
  getAccount,
  getCandles,
  getClosedTrades,
  getOpenTrades,
  getBotStatus,
  type AccountInfo,
  type Candle,
  type Trade,
  type BotStatus,
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

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [account, setAccount] = useState<AccountInfo | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [closedTrades, setClosedTrades] = useState<Trade[]>([]);
  const [openTrades, setOpenTrades] = useState<Trade[]>([]);
  const [botStatus, setBotStatus] = useState<BotStatus | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchAll = useCallback(async () => {
    const results = await Promise.allSettled([
      getAccount(),
      getCandles(),
      getClosedTrades(),
      getOpenTrades(),
      getBotStatus(),
    ]);

    if (results[0].status === "fulfilled") setAccount(results[0].value);
    if (results[1].status === "fulfilled") setCandles(results[1].value);
    if (results[2].status === "fulfilled") setClosedTrades(results[2].value);
    if (results[3].status === "fulfilled") setOpenTrades(results[3].value);
    if (results[4].status === "fulfilled") setBotStatus(results[4].value);
    setLastUpdate(new Date());
  }, []);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 10000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold tracking-tight text-white">
              Trading Bot
            </h1>
            <BotControls status={botStatus} />
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500">
              Updated {lastUpdate.toLocaleTimeString()}
            </span>
            <button
              onClick={fetchAll}
              className="rounded-lg border border-gray-700 bg-gray-800 px-3 py-1.5 text-xs font-medium text-gray-300 transition hover:border-gray-600 hover:text-white"
            >
              Refresh
            </button>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <TabNav activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Main Content */}
      <div className="mx-auto flex max-w-[1920px] gap-4 px-4 py-6">
        {/* Left: Log Panel */}
        <aside className="hidden w-80 shrink-0 xl:block" style={{ height: "calc(100vh - 140px)" }}>
          <LogPanel />
        </aside>

        {/* Right: Dashboard */}
        <main className="min-w-0 flex-1 space-y-6">
          {activeTab === "dashboard" && (
            <>
              <KPICards account={account} closedTrades={closedTrades} />

              <div className="grid gap-6 lg:grid-cols-2">
                <CandlestickChart candles={candles} />
                <EquityCurve trades={closedTrades} />
              </div>

              <TradesTable closedTrades={closedTrades} openTrades={openTrades} />

              <BacktestPanel />
            </>
          )}

          {activeTab === "strategy-builder" && <StrategyBuilder />}
        </main>
      </div>
    </div>
  );
}
