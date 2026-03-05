import { useState, useEffect, useCallback } from "react";
import {
  listRLModels,
  deleteRLModel,
  activateRLModel,
  getTrainingMetrics,
  type RLModelInfo,
} from "../../api";

const MODEL_PROFILE_TABS = ["all", "max_profit", "aggressive", "medium", "conservative"] as const;

export default function ModelManager({ refreshKey }: { refreshKey: number }) {
  const [models, setModels] = useState<RLModelInfo[]>([]);
  const [tab, setTab] = useState<string>("all");
  const [busy, setBusy] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  const [activeModelId, setActiveModelId] = useState<number | null>(null);

  const fetchModels = useCallback(async () => {
    try {
      const resp = await listRLModels(tab === "all" ? "" : tab);
      setModels(resp.models);
    } catch {
      /* ignore */
    }
  }, [tab]);

  // Fetch active model id on mount and when refreshKey changes
  useEffect(() => {
    (async () => {
      try {
        const metrics = await getTrainingMetrics();
        setActiveModelId(metrics.active_model_id);
      } catch {
        /* ignore */
      }
    })();
  }, [refreshKey]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels, refreshKey]);

  const handleActivate = async (id: number) => {
    setBusy(`activate-${id}`);
    try {
      await activateRLModel(id);
      setActiveModelId(id);
      await fetchModels();
    } catch (e) {
      console.error("Activate failed:", e);
    } finally {
      setBusy("");
    }
  };

  const handleDelete = async (id: number) => {
    setBusy(`delete-${id}`);
    try {
      await deleteRLModel(id);
      setConfirmDelete(null);
      if (activeModelId === id) setActiveModelId(null);
      await fetchModels();
    } catch (e) {
      console.error("Delete failed:", e);
    } finally {
      setBusy("");
    }
  };

  const profileBadge: Record<string, string> = {
    max_profit: "bg-blue-500/20 text-blue-400",
    aggressive: "bg-red-500/20 text-red-400",
    medium: "bg-yellow-500/20 text-yellow-400",
    conservative: "bg-green-500/20 text-green-400",
  };

  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800 p-5">
      <h3 className="mb-3 text-sm font-semibold text-gray-300">
        Saved RL Models
      </h3>

      {/* Profile tabs */}
      <div className="mb-3 flex gap-1.5">
        {MODEL_PROFILE_TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`rounded-lg px-3 py-1 text-[11px] font-medium capitalize transition ${
              tab === t
                ? "bg-gray-600 text-white"
                : "bg-gray-900/50 text-gray-500 hover:text-gray-300"
            }`}
          >
            {t === "all" ? "All" : t.replace("_", " ")}
          </button>
        ))}
      </div>

      {/* Model list */}
      <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
        {models.map((m) => {
          const isActive = m.id === activeModelId;
          return (
            <div
              key={m.id}
              className={`flex items-center justify-between rounded-lg px-3 py-2 ${
                isActive
                  ? "border-2 border-emerald-500/60 bg-emerald-900/10"
                  : "border border-gray-700/50 bg-gray-900/50"
              }`}
            >
              <div className="flex items-center gap-3 min-w-0">
                <span
                  className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider ${
                    profileBadge[m.profile || m.model_name] || "bg-gray-600/30 text-gray-400"
                  }`}
                >
                  {m.profile || m.model_name}
                </span>
                <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-gray-400">
                  <span>ep {m.episode}</span>
                  <span>WR {(m.win_rate * 100).toFixed(1)}%</span>
                  <span>reward {m.total_reward.toFixed(1)}</span>
                  {m.timeframe && <span>{m.timeframe}</span>}
                  <span className="text-gray-600">
                    {m.created_at
                      ? new Date(m.created_at).toLocaleDateString()
                      : "\u2014"}
                  </span>
                </div>
              </div>
              <div className="ml-3 flex shrink-0 gap-1.5">
                {isActive ? (
                  <span className="rounded border border-emerald-600/50 bg-emerald-900/30 px-2.5 py-1 text-[10px] font-medium text-emerald-400">
                    Active
                  </span>
                ) : (
                  <button
                    onClick={() => handleActivate(m.id)}
                    disabled={busy === `activate-${m.id}`}
                    className="rounded border border-green-700/50 bg-green-900/30 px-2.5 py-1 text-[10px] font-medium text-green-400 transition hover:bg-green-900/50 disabled:opacity-50"
                  >
                    {busy === `activate-${m.id}` ? "..." : "Activate"}
                  </button>
                )}
                {confirmDelete === m.id ? (
                  <div className="flex gap-1">
                    <button
                      onClick={() => handleDelete(m.id)}
                      disabled={busy === `delete-${m.id}`}
                      className="rounded border border-red-600 bg-red-700 px-2 py-1 text-[10px] font-medium text-white disabled:opacity-50"
                    >
                      Confirm
                    </button>
                    <button
                      onClick={() => setConfirmDelete(null)}
                      className="rounded border border-gray-600 px-2 py-1 text-[10px] text-gray-400"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setConfirmDelete(m.id)}
                    className="rounded border border-red-700/50 bg-red-900/30 px-2.5 py-1 text-[10px] font-medium text-red-400 transition hover:bg-red-900/50"
                  >
                    Delete
                  </button>
                )}
              </div>
            </div>
          );
        })}
        {models.length === 0 && (
          <p className="py-4 text-center text-xs text-gray-500">
            No saved models{tab !== "all" ? ` for ${tab.replace("_", " ")}` : ""}
          </p>
        )}
      </div>
    </div>
  );
}
