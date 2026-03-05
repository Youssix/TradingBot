import Tooltip from "./Tooltip";

interface InfoIconProps {
  tooltip: string;
}

export default function InfoIcon({ tooltip }: InfoIconProps) {
  return (
    <Tooltip text={tooltip}>
      <span className="inline-flex h-4 w-4 cursor-help items-center justify-center rounded-full border border-gray-500 text-[9px] font-bold text-gray-400 hover:border-gray-400 hover:text-gray-300">
        ?
      </span>
    </Tooltip>
  );
}
