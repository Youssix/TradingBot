interface TabNavProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const TABS = [
  { id: "dashboard", label: "Dashboard" },
  { id: "strategy-builder", label: "Strategy Builder" },
  { id: "logs", label: "Logs", mobileOnly: true },
];

export default function TabNav({ activeTab, onTabChange }: TabNavProps) {
  return (
    <nav
      className="flex gap-0 overflow-x-auto px-5"
      style={{ borderBottom: "1px solid var(--border-subtle)", background: "rgba(11,15,25,0.6)" }}
    >
      {TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`relative shrink-0 px-5 py-2.5 text-[13px] font-medium transition-colors ${
            activeTab === tab.id
              ? "text-white"
              : "text-slate-500 hover:text-slate-300"
          } ${"mobileOnly" in tab && tab.mobileOnly ? "xl:hidden" : ""}`}
        >
          {tab.label}
          {activeTab === tab.id && (
            <span
              className="absolute bottom-0 left-2 right-2 h-[2px] rounded-full"
              style={{ background: "var(--accent-teal)" }}
            />
          )}
        </button>
      ))}
    </nav>
  );
}
