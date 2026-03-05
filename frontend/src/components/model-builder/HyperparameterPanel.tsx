import { useState } from "react";
import { HYPERPARAMS, type HyperparamMeta } from "../../api";

interface HyperparameterPanelProps {
  agentType: string;
  values: Record<string, any>;
  onChange: (key: string, value: any) => void;
  onReset: () => void;
}

const CATEGORY_LABELS: Record<string, string> = {
  learning: "Learning",
  architecture: "Architecture",
  exploration: "Exploration",
  risk: "Risk",
};

const CATEGORY_ORDER = ["architecture", "learning", "exploration", "risk"];

function TooltipIcon({ text }: { text: string }) {
  const [show, setShow] = useState(false);
  return (
    <span
      className="relative ml-1 inline-flex cursor-help"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <span className="flex h-4 w-4 items-center justify-center rounded-full bg-gray-700 text-[10px] text-gray-400">
        ?
      </span>
      {show && (
        <span className="absolute bottom-full left-1/2 z-20 mb-2 w-56 -translate-x-1/2 rounded-lg border border-gray-600 bg-gray-900 p-2 text-xs text-gray-300 shadow-xl">
          {text}
        </span>
      )}
    </span>
  );
}

function SliderControl({
  param,
  value,
  onChange,
}: {
  param: HyperparamMeta;
  value: number;
  onChange: (v: number) => void;
}) {
  const displayVal =
    param.step && param.step < 0.001
      ? value.toExponential(1)
      : param.step && param.step < 1
        ? value.toFixed(
            Math.max(0, -Math.floor(Math.log10(param.step)))
          )
        : String(value);

  return (
    <div className="flex items-center gap-3">
      <span className="w-10 text-right text-xs text-gray-500">{param.min}</span>
      <input
        type="range"
        min={param.min}
        max={param.max}
        step={param.step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-gray-700 accent-indigo-500"
      />
      <span className="w-10 text-xs text-gray-500">{param.max}</span>
      <span className="w-16 rounded bg-gray-900 px-2 py-0.5 text-center text-xs font-mono text-white">
        {displayVal}
      </span>
    </div>
  );
}

function ToggleControl({
  value,
  onChange,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition ${
        value ? "bg-indigo-600" : "bg-gray-700"
      }`}
    >
      <span
        className={`inline-block h-4 w-4 rounded-full bg-white transition-transform ${
          value ? "translate-x-6" : "translate-x-1"
        }`}
      />
    </button>
  );
}

function SelectControl({
  param,
  value,
  onChange,
}: {
  param: HyperparamMeta;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-md border border-gray-600 bg-gray-900 px-3 py-1.5 text-sm text-gray-300 outline-none focus:border-indigo-500"
    >
      {param.options?.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

export default function HyperparameterPanel({
  agentType,
  values,
  onChange,
  onReset,
}: HyperparameterPanelProps) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const filtered = HYPERPARAMS.filter((p) => p.agents.includes(agentType));

  const grouped: Record<string, HyperparamMeta[]> = {};
  for (const p of filtered) {
    if (!grouped[p.category]) grouped[p.category] = [];
    grouped[p.category].push(p);
  }

  const categories = CATEGORY_ORDER.filter((c) => grouped[c]);

  if (categories.length === 0) {
    return (
      <div className="rounded-xl border border-gray-700 bg-gray-800 p-6 text-center text-sm text-gray-500">
        No configurable hyperparameters for this agent type.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {categories.map((cat) => {
        const isCollapsed = collapsed[cat] ?? false;
        const params = grouped[cat];
        return (
          <div
            key={cat}
            className="rounded-xl border border-gray-700 bg-gray-800 overflow-hidden"
          >
            <button
              onClick={() =>
                setCollapsed((prev) => ({ ...prev, [cat]: !isCollapsed }))
              }
              className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-gray-700/30 transition"
            >
              <span className="text-xs font-semibold uppercase tracking-wider text-gray-400">
                {CATEGORY_LABELS[cat]}
              </span>
              <span className="text-gray-500 text-xs">
                {isCollapsed ? "\u25B6" : "\u25BC"}
              </span>
            </button>

            {!isCollapsed && (
              <div className="space-y-4 px-4 pb-4">
                {params.map((p) => {
                  const val = values[p.key] ?? p.default;
                  return (
                    <div key={p.key}>
                      <div className="mb-1.5 flex items-center">
                        <label className="text-sm font-medium text-gray-300">
                          {p.label}
                        </label>
                        <TooltipIcon text={p.tooltip} />
                      </div>
                      {p.type === "slider" && (
                        <SliderControl
                          param={p}
                          value={val as number}
                          onChange={(v) => onChange(p.key, v)}
                        />
                      )}
                      {p.type === "toggle" && (
                        <ToggleControl
                          value={val as boolean}
                          onChange={(v) => onChange(p.key, v)}
                        />
                      )}
                      {p.type === "select" && (
                        <SelectControl
                          param={p}
                          value={val as string}
                          onChange={(v) => onChange(p.key, v)}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}

      <button
        onClick={onReset}
        className="w-full rounded-lg border border-gray-600 bg-gray-800 px-4 py-2 text-sm text-gray-400 transition hover:border-gray-500 hover:text-white"
      >
        Reset to Defaults
      </button>
    </div>
  );
}
