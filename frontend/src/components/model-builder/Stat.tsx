export default function Stat({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string | number;
  highlight?: boolean;
}) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider text-gray-500">
        {label}
      </p>
      <p
        className={`text-sm font-semibold ${
          highlight ? "text-white" : "text-gray-300"
        }`}
      >
        {value}
      </p>
    </div>
  );
}
