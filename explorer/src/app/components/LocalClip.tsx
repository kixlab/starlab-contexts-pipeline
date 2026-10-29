"use client";
import { useEffect, useRef } from "react";

/** HTML5 playback for local or self-hosted media. No external player service. */
export function LocalClip({src, startTime = 0, title = "Tutorial clip"}: {
  src: string; startTime?: number; title?: string;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (ref.current && ref.current.readyState >= 1) ref.current.currentTime = startTime;
  }, [src, startTime]);
  return <video key={src} ref={ref} src={src} aria-label={title} controls preload="metadata"
    className="w-full rounded-lg bg-black" onLoadedMetadata={() => {
      if (ref.current) ref.current.currentTime = startTime;
    }} />;
}
