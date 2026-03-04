import { useEffect, useRef } from "react";
import { createChart, type IChartApi, ColorType } from "lightweight-charts";
import type { Trade } from "../api";

interface EquityCurveProps {
  trades: Trade[];
  height?: number;
}

export default function EquityCurve({ trades, height = 400 }: EquityCurveProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#1f2937" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "#374151" },
        horzLines: { color: "#374151" },
      },
      rightPriceScale: {
        borderColor: "#4b5563",
      },
      timeScale: {
        borderColor: "#4b5563",
        timeVisible: true,
        secondsVisible: false,
      },
      height,
    });

    chartRef.current = chart;

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
      const data = sorted.map((t) => {
        cumulative += t.pnl ?? 0;
        return {
          time: (new Date(t.closed_at!).getTime() / 1000) as number,
          value: cumulative,
        };
      });

      if (data.length > 0) {
        lineSeries.setData(data as any);
        chart.timeScale().fitContent();
      }
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
  }, [trades, height]);

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-4">
      <h3 className="mb-3 text-sm font-semibold tracking-wider text-gray-400 uppercase">
        Equity Curve
      </h3>
      <div ref={containerRef} className="w-full" style={{ height }} />
    </div>
  );
}
