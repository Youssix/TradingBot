import { useEffect, useRef } from "react";
import { createChart, type IChartApi, ColorType } from "lightweight-charts";
import type { Candle } from "../api";

interface CandlestickChartProps {
  candles: Candle[];
}

export default function CandlestickChart({ candles }: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#111827" },
        textColor: "#64748b",
      },
      grid: {
        vertLines: { color: "rgba(148,163,184,0.06)" },
        horzLines: { color: "rgba(148,163,184,0.06)" },
      },
      crosshair: {
        mode: 0,
      },
      rightPriceScale: {
        borderColor: "rgba(148,163,184,0.1)",
      },
      timeScale: {
        borderColor: "rgba(148,163,184,0.1)",
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#14b8a6",
      downColor: "#f43f5e",
      borderDownColor: "#f43f5e",
      borderUpColor: "#14b8a6",
      wickDownColor: "#f43f5e",
      wickUpColor: "#14b8a6",
    });

    const volumeSeries = chart.addHistogramSeries({
      color: "#3b82f6",
      priceFormat: { type: "volume" },
      priceScaleId: "",
    });

    chart.priceScale("").applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    if (candles.length > 0) {
      const candleData = candles.map((c) => ({
        time: (new Date(c.datetime).getTime() / 1000) as number,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }));

      const volumeData = candles.map((c) => ({
        time: (new Date(c.datetime).getTime() / 1000) as number,
        value: c.volume,
        color: c.close >= c.open ? "rgba(20,184,166,0.2)" : "rgba(244,63,94,0.2)",
      }));

      candleSeries.setData(candleData as any);
      volumeSeries.setData(volumeData as any);
      chart.timeScale().fitContent();
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
  }, [candles]);

  return (
    <div className="rounded-2xl p-4 card-glow" style={{ background: "var(--bg-card)", border: "1px solid var(--border-subtle)" }}>
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
        Price Chart
      </h3>
      <div ref={containerRef} className="h-[400px] w-full" />
    </div>
  );
}
