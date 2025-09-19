export type TooltipData = { x: number; y: number; title: string; text: string };

export function Tooltip({ tip }: { tip: TooltipData }) {
  return (
    <div
      className="fixed z-[1000] w-80 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 p-3 text-xs text-neutral-700 dark:text-neutral-300 shadow-2xl whitespace-pre-wrap"
      style={{ left: tip.x + 12, top: tip.y }}
    >
      <div className="font-medium mb-1">{tip.title}</div>
      {tip.text}
    </div>
  );
}