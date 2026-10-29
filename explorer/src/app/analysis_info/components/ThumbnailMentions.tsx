import { getInfoTypeColor } from "@/app/constants/colors";

interface ThumbnailMentionsProps {
  mentions: number;
  videoCount: number;
  unitId: string;
  className?: string;
  contentType?: string;
}

export function ThumbnailMentions({ 
  mentions, 
  videoCount, 
  unitId, 
  className = "",
  contentType
}: ThumbnailMentionsProps) {
  const maxThumbs = Math.min(3, videoCount);
  const offsetStep = 8; // px offset between overlapped thumbnails
  const color = contentType ? getInfoTypeColor(contentType) : "#3B82F6"; // fallback to blue
  return (
    <div className={`relative w-[64px] h-[40px] ${className}`}>
      {/* Overlapped thumbnails in background (diagonal from top-right to bottom-left) */}
      <div className="absolute inset-0">
        {Array.from({ length: maxThumbs }, (_, index) => {
          // First (index 0) sits bottom-left; subsequent ones shift up-right
          const left = index * offsetStep;
          const top = (maxThumbs - 1 - index) * offsetStep;
          return (
            <div
              key={`${unitId}-thumb-${index}`}
              className="absolute w-12 h-8 rounded-md bg-neutral-200 dark:bg-neutral-800 flex items-center justify-center text-[8px] text-neutral-500 border border-neutral-300 dark:border-neutral-700"
              style={{ left, top, zIndex: maxThumbs - index }}
              aria-hidden
            >
              {index === 0 ? '📹' : ''}
            </div>
          );
        })}
      </div>
      
      {/* Mentions count prominently on top */}
      {(() => {
        const firstTop = (maxThumbs - 1) * offsetStep; // top position of first (bottom-left) thumbnail
        return (
          <div
            className="absolute min-w-[20px] h-5 text-white text-[10px] font-bold rounded-full flex items-center justify-center shadow-lg border border-white dark:border-neutral-900"
            style={{ 
              left: 24, 
              top: firstTop + 16, 
              transform: 'translate(-50%, -50%)', 
              zIndex: maxThumbs + 2,
              backgroundColor: color
            }}
          >
            {mentions}
          </div>
        );
      })()}
    </div>
  );
}