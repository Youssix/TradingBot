import { useState, useEffect, useRef } from "react";
import { createChart, type IChartApi, ColorType } from "lightweight-charts";
import { runBacktest, type BacktestResult } from "../api";

function MiniEquityCurve({ data }: { data: { time: string; value: number }[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#111827" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      rightPriceScale: { borderColor: "#374151" },
      timeScale: { borderColor: "#374151", timeVisible: true },
      height: 200,
    });

    chartRef.current = chart;

    const series = chart.addAreaSeries({
      topColor: "rgba(16,185,129,0.4)",
      bottomColor: "rgba(16,185,129,0.0)",
      lineColor: "#10b981",
      lineWidth: 2,
    });

    const chartData = data.map((p) => ({
      time: (new Date(p.time).getTime() / 1000) as number,
      value: p.value,
    }));

    series.setData(chartData as any);
    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [data]);

  return <div ref={containerRef} className="h-[200px] w-full" />;
}

export default function BacktestPanel() {
  const [open, setOpen] = useState(false);
  const [strategy, setStrategy] = useState<"ema" | "breakout">("ema");
  const [count, setCount] = useState(500);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setLoading(true);
    setError(null);
    try {
      const data = await runBacktest(strategy, count);
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Backtest failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between p-5 text-left"
      >
        <h3 className="text-sm font-semibold tracking-wider text-gray-400 uppercase">
          Backtesting
        </h3>
        <span className="text-gray-500 transition" style={{ transform: open ? "rotate(180deg)" : "" }}>
          &#9660;
        </span>
      </button>

      {open && (
        <div className="border-t border-gray-700 p-5">
          <div className="mb-5 flex flex-wrap items-end gap-4">
            <div>
              <label className="mb-1 block text-xs text-gray-400">Strategy</label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as "ema" | "breakout")}
                className="rounded-lg border border-gray-600 bg-gray-900 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
              >
                <option value="ema">EMA Crossover</option>
                <option value="breakout">Asian Breakout</option>
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs text-gray-400">Bar Count</label>
              <input
                type="number"
                value={count}
                onChange={(e) => setCount(Number(e.target.value))}
                min={50}
                max={5000}
                className="w-28 rounded-lg border border-gray-600 bg-gray-900 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
              />
            </div>
            <button
              onClick={handleRun}
              disabled={loading}
              className="rounded-lg bg-indigo-600 px-5 py-2 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Running...
                </span>
              ) : (
                "Run Backtest"
              )}
            </button>
          </div>

          {error && (
            <div className="mb-4 rounded-lg border border-red-800 bg-red-900/30 px-4 py-3 text-sm text-red-300">
              {error}
            </div>
          )}

          {result && (
            <div>
              <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                {[
                  { label: "Total Trades", value: result.metrics.total_trades.toString() },
                  {
                    label: "Win Rate",
                    value: `${result.metrics.win_rate.toFixed(1)}%`,
                    color:
                      result.metrics.win_rate >= 50
                        ? "text-emerald-400"
                        : "text-red-400",
                  },
                  {
                    label: "Profit Factor",
                    value: result.metrics.profit_factor.toFixed(2),
                    color:
                      result.metrics.profit_factor >= 1
                        ? "text-emerald-400"
                        : "text-red-400",
                  },
                  {
                    label: "Total P&L",
                    value: `$${result.metrics.total_pnl.toFixed(2)}`,
                    color:
                      result.metrics.total_pnl >= 0
                        ? "text-emerald-400"
                        : "text-red-400",
                  },
                  {
                    label: "Avg Win",
                    value: `$${result.metrics.avg_win.toFixed(2)}`,
                    color: "text-emerald-400",
                  },
                  {
                    label: "Avg Loss",
                    value: `$${result.metrics.avg_loss.toFixed(2)}`,
                    color: "text-red-400",
                  },
                  {
                    label: "Max Drawdown",
                    value: `$${result.metrics.max_drawdown.toFixed(2)}`,
                    color: "text-amber-400",
                  },
                  {
                    label: "Sharpe Ratio",
                    value: result.metrics.sharpe_ratio.toFixed(2),
                    color:
                      result.metrics.sharpe_ratio >= 1
                        ? "text-emerald-400"
                        : "text-amber-400",
                  },
                ].map((m) => (
                  <div
                    key={m.label}
                    className="rounded-lg border border-gray-700 bg-gray-900 p-3"
                  >
                    <p className="mb-0.5 text-[10px] text-gray-500 uppercase">
                      {m.label}
                    </p>
                    <p className={`text-lg font-bold ${m.color || "text-white"}`}>
                      {m.value}
                    </p>
                  </div>
                ))}
              </div>

              {result.equity_curve.length > 0 && (
                <div className="rounded-lg border border-gray-700 bg-gray-900 p-3">
                  <p className="mb-2 text-xs text-gray-500 uppercase">
                    Backtest Equity Curve
                  </p>
                  <MiniEquityCurve data={result.equity_curve} />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
