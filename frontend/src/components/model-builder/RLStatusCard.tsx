import type { LearningStatus } from "../../api";
import Stat from "./Stat";

export default function RLStatusCard({ status }: { status: LearningStatus }) {
  const rl = status.rl_stats;
  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">RL Agent Status</h3>
        <span
          className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
            rl.training
              ? "bg-green-500/20 text-green-400"
              : "bg-gray-600/30 text-gray-400"
          }`}
        >
          {rl.training ? "Training" : "Idle"}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Episode" value={rl.episode.toLocaleString()} />
        <Stat label="Epsilon" value={rl.epsilon.toFixed(4)} />
        <Stat label="Total Reward" value={rl.total_reward.toFixed(1)} highlight />
        <Stat
          label="Win Rate"
          value={`${(rl.win_rate * 100).toFixed(1)}%`}
          highlight
        />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        <Stat label="Buffer Size" value={rl.buffer_size.toLocaleString()} />
        <Stat
          label="PyTorch"
          value={rl.torch_available ? "Available" : "Missing"}
        />
      </div>
    </div>
  );
}
