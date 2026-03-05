import { useEffect, useRef } from "react";
import { createChart, type IChartApi, ColorType } from "lightweight-charts";
import type { Trade } from "../api";

interface EquityCurveProps {
  trades: Trade[];
  height?: number;
  showDrawdown?: boolean;
  showMarkers?: boolean;
}

export default function EquityCurve({
  trades,
  height = 400,
  showDrawdown = true,
  showMarkers = true,
}: EquityCurveProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const sharpeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    let chart: IChartApi;
    try {
      chart = createChart(containerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: "#111827" },
          textColor: "#64748b",
        },
        grid: {
          vertLines: { color: "rgba(148,163,184,0.06)" },
          horzLines: { color: "rgba(148,163,184,0.06)" },
        },
        rightPriceScale: {
          borderColor: "rgba(148,163,184,0.1)",
        },
        timeScale: {
          borderColor: "rgba(148,163,184,0.1)",
          timeVisible: true,
          secondsVisible: false,
        },
        height,
      });
    } catch {
      return;
    }

    chartRef.current = chart;

    try {
      const lineSeries = chart.addAreaSeries({
        topColor: "rgba(99,102,241,0.4)",
        bottomColor: "rgba(99,102,241,0.0)",
        lineColor: "#6366f1",
        lineWidth: 2,
      });

      if (trades.length > 0) {
        const sorted = [...trades]
          .filter((t) => t.closed_at)
          .sort(
            (a, b) =>
              new Date(a.closed_at!).getTime() - new Date(b.closed_at!).getTime()
          );

        let cumulative = 0;
        let lastTime = 0;
        const data = sorted.map((t) => {
          cumulative += t.pnl ?? 0;
          let time = Math.floor(new Date(t.closed_at!).getTime() / 1000);
          if (time <= lastTime) {
            time = lastTime + 1;
          }
          lastTime = time;
          return { time: time as number, value: cumulative };
        });

        if (data.length > 0) {
          lineSeries.setData(data as any);

          // Drawdown shading
          if (showDrawdown) {
            const drawdownSeries = chart.addAreaSeries({
              topColor: "rgba(239,68,68,0.3)",
              bottomColor: "rgba(239,68,68,0.0)",
              lineColor: "rgba(239,68,68,0.5)",
              lineWidth: 1,
            });

            let peak = -Infinity;
            const ddData = data.map((d) => {
              if (d.value > peak) peak = d.value;
              // Show peak line; the gap between peak and equity line = drawdown area
              return { time: d.time as number, value: peak > d.value ? peak : d.value };
            });
            // Only set data if there's actual drawdown
            const hasDrawdown = data.some((d, i) => ddData[i].value > d.value);
            if (hasDrawdown) {
              drawdownSeries.setData(ddData as any);
            }
          }

          // Trade markers
          if (showMarkers && sorted.length > 0) {
            const markers = sorted.map((t, i) => {
              const pnl = t.pnl ?? 0;
              return {
                time: data[i].time,
                position: pnl > 0 ? "aboveBar" as const : "belowBar" as const,
                color: pnl > 0 ? "#10b981" : "#ef4444",
                shape: pnl > 0 ? "arrowUp" as const : "arrowDown" as const,
                text: pnl > 0 ? `+${pnl.toFixed(2)}` : pnl.toFixed(2),
              };
            });
            lineSeries.setMarkers(markers as any);
          }

          // Compute Sharpe ratio
          const pnls = sorted.map((t) => t.pnl ?? 0);
          if (pnls.length > 1) {
            const mean = pnls.reduce((s, v) => s + v, 0) / pnls.length;
            const variance =
              pnls.reduce((s, v) => s + (v - mean) ** 2, 0) / (pnls.length - 1);
            const std = Math.sqrt(variance);
            const sharpe = std > 0 ? mean / std : 0;
            if (sharpeRef.current) {
              sharpeRef.current.textContent = `Sharpe: ${sharpe.toFixed(2)}`;
              sharpeRef.current.style.color =
                sharpe >= 1 ? "#10b981" : sharpe >= 0 ? "#eab308" : "#ef4444";
            }
          } else if (sharpeRef.current) {
            sharpeRef.current.textContent = "Sharpe: --";
            sharpeRef.current.style.color = "#9ca3af";
          }

          chart.timeScale().fitContent();
        }
      }
    } catch (e) {
      console.warn("EquityCurve chart data error:", e);
    }

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
  }, [trades, height, showDrawdown, showMarkers]);

  return (
    <div className="rounded-2xl p-4 card-glow" style={{ background: "var(--bg-card)", border: "1px solid var(--border-subtle)" }}>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
          Equity Curve
        </h3>
        <div
          ref={sharpeRef}
          className="font-num text-xs font-semibold"
          style={{ color: "var(--text-secondary)" }}
        >
          Sharpe: --
        </div>
      </div>
      <div ref={containerRef} className="w-full" style={{ height }} />
    </div>
  );
}
