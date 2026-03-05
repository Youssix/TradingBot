import { useState } from "react";

interface AgentSelectorProps {
  selectedAgent: string;
  onSelect: (agent: string) => void;
  ensembleAgents?: string[];
  onEnsembleAgentsChange?: (agents: string[]) => void;
}

const AGENTS = [
  {
    id: "sac",
    icon: "\uD83E\uDDE0",
    name: "SAC",
    subtitle: "Smart Adaptive Critic",
    description:
      "The most versatile agent. Automatically balances exploration and exploitation with entropy tuning.",
    color: "purple",
    ring: "ring-purple-500",
    border: "border-purple-500",
    accent: "text-purple-400",
  },
  {
    id: "ppo",
    icon: "\uD83D\uDCCA",
    name: "PPO",
    subtitle: "Proximal Policy Optimizer",
    description:
      "Stable and reliable. Makes careful, measured improvements each training step.",
    color: "blue",
    ring: "ring-blue-500",
    border: "border-blue-500",
    accent: "text-blue-400",
  },
  {
    id: "ddpg",
    icon: "\uD83C\uDFAF",
    name: "DDPG",
    subtitle: "Deep Deterministic Policy Gradient",
    description:
      "Precise and decisive. Best for clear-cut trading signals.",
    color: "amber",
    ring: "ring-amber-500",
    border: "border-amber-500",
    accent: "text-amber-400",
  },
  {
    id: "dqn",
    icon: "\uD83D\uDCA1",
    name: "DQN",
    subtitle: "Deep Q-Network",
    description:
      "The classic. Simple but effective for basic trading patterns.",
    color: "gray",
    ring: "ring-gray-400",
    border: "border-gray-400",
    accent: "text-gray-300",
  },
  {
    id: "ensemble",
    icon: "\uD83E\uDD1D",
    name: "Ensemble",
    subtitle: "Multi-Agent Team",
    description:
      "Combines multiple agents. The team votes on each trade for better decisions.",
    color: "emerald",
    ring: "ring-emerald-500",
    border: "border-emerald-500",
    accent: "text-emerald-400",
  },
] as const;

const ENSEMBLE_OPTIONS = ["sac", "ppo", "ddpg"];

export default function AgentSelector({
  selectedAgent,
  onSelect,
  ensembleAgents = ["sac", "ppo"],
  onEnsembleAgentsChange,
}: AgentSelectorProps) {
  const [hoveredTooltip, setHoveredTooltip] = useState<string | null>(null);

  function handleEnsembleToggle(agent: string) {
    const current = new Set(ensembleAgents);
    if (current.has(agent)) {
      if (current.size <= 2) return; // minimum 2
      current.delete(agent);
    } else {
      current.add(agent);
    }
    onEnsembleAgentsChange?.(Array.from(current));
  }

  return (
    <div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {AGENTS.map((a) => {
          const isSelected = selectedAgent === a.id;
          return (
            <button
              key={a.id}
              onClick={() => onSelect(a.id)}
              onMouseEnter={() => setHoveredTooltip(a.id)}
              onMouseLeave={() => setHoveredTooltip(null)}
              className={`relative rounded-xl border bg-gray-800 p-4 text-left transition cursor-pointer hover:border-gray-500 ${
                isSelected
                  ? `${a.border} ring-2 ${a.ring} shadow-lg shadow-${a.color}-500/10`
                  : "border-gray-700"
              }`}
            >
              <div className="mb-2 text-2xl">{a.icon}</div>
              <div className={`text-sm font-bold ${isSelected ? a.accent : "text-white"}`}>
                {a.name}
              </div>
              <div className="mt-0.5 text-xs text-gray-500">{a.subtitle}</div>
              {hoveredTooltip === a.id && (
                <div className="absolute left-0 right-0 top-full z-10 mt-2 rounded-lg border border-gray-600 bg-gray-900 p-3 text-xs text-gray-300 shadow-xl">
                  {a.description}
                </div>
              )}
            </button>
          );
        })}
      </div>

      {selectedAgent === "ensemble" && (
        <div className="mt-4 rounded-lg border border-gray-700 bg-gray-800/50 p-4">
          <p className="mb-3 text-xs font-medium text-gray-400">
            Select sub-agents (minimum 2):
          </p>
          <div className="flex gap-4">
            {ENSEMBLE_OPTIONS.map((agent) => {
              const info = AGENTS.find((a) => a.id === agent)!;
              const checked = ensembleAgents.includes(agent);
              const disabled = checked && ensembleAgents.length <= 2;
              return (
                <label
                  key={agent}
                  className={`flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-2 text-sm transition ${
                    checked
                      ? `${info.border} ${info.accent}`
                      : "border-gray-700 text-gray-500"
                  } ${disabled ? "opacity-60" : ""}`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => handleEnsembleToggle(agent)}
                    disabled={disabled}
                    className="accent-indigo-500"
                  />
                  <span>{info.icon}</span>
                  <span>{info.name}</span>
                </label>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
