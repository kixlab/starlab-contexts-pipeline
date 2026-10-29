export function ProgressBar({ 
  value, 
  variant = "default", 
  color 
}: { 
  value: number; 
  variant?: "default" | "rainbow";
  color?: string;
}) {
  return (
    <div className="w-full h-3 rounded-full bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
      <div 
        className={`h-full transition-all duration-500 ease-in-out ${
          variant === "rainbow" ? "" : color ? "" : "bg-blue-600"
        }`} 
        style={{ 
          width: `${Math.min(100, Math.max(0, value))}%`,
          backgroundColor: variant === "rainbow" ? undefined : color || undefined,
          backgroundImage:
            variant === "rainbow"
              ? "linear-gradient(to right, #648FFF, #EEA400, #785EF0, #DC267F)"
              : undefined
        }} 
      />
    </div>
  );
}