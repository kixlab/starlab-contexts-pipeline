"use client";

import { useEffect, useRef, useState } from "react";

interface VideoPlayerProps {
  videoUrl: string;
  startTime: number;
  endTime?: number;
  className?: string;
  maxWidth?: number;
}

export function VideoPlayer({ videoUrl, startTime, endTime, className = "", maxWidth = 400 }: VideoPlayerProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>(null);
  const [calculatedHeight, setCalculatedHeight] = useState<number>(112); // Default 28 * 4 (h-28 = 7rem = 112px)
  const [embedUrl, setEmbedUrl] = useState<string>("");

  // Extract YouTube video ID from URL
  const getYouTubeVideoId = (url: string): string | null => {
    const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
    const match = url.match(regex);
    return match ? match[1] : null;
  };

  const videoId = getYouTubeVideoId(videoUrl);

  // Fetch video dimensions from YouTube API
  const fetchVideoDimensions = async (videoId: string) => {
    try {
      // Using a simple approach - most YouTube videos are 16:9 aspect ratio
      // For more accuracy, you could use YouTube Data API with an API key
      const response = await fetch(`https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`);
      if (response.ok) {
        const data = await response.json();
        // YouTube oEmbed returns width and height
        setVideoDimensions({ width: data.width, height: data.height });
      } else {
        // Fallback to standard 16:9 aspect ratio
        setVideoDimensions({ width: 1280, height: 720 });
      }
    } catch (error) {
      // Fallback to standard 16:9 aspect ratio
      setVideoDimensions({ width: 1280, height: 720 });
    }
  };

  // Calculate height based on container width and video aspect ratio
  useEffect(() => {
    if (videoDimensions && containerRef.current) {
      const containerWidth = containerRef.current.offsetWidth;
      const aspectRatio = videoDimensions.height / videoDimensions.width;
      const newHeight = Math.min(containerWidth * aspectRatio, maxWidth * aspectRatio);
      setCalculatedHeight(Math.max(newHeight, 80)); // Minimum height of 80px
    }
  }, [videoDimensions, maxWidth]);

  useEffect(() => {
    if (!videoId) {
      setError("Invalid YouTube URL");
      setIsLoading(false);
      return;
    }

    // Fetch video dimensions first
    fetchVideoDimensions(videoId);

    // Create YouTube embed URL with start time
    const newEmbedUrl = `https://www.youtube.com/embed/${videoId}?start=${Math.floor(startTime)}&autoplay=0&controls=1&modestbranding=1&rel=0&showinfo=0`;
    setEmbedUrl(newEmbedUrl);
  }, [videoId, startTime]);

  const handleLoad = () => {
    setIsLoading(false);
    setError(null);
  };

  const handleError = () => {
    setError("Failed to load video");
    setIsLoading(false);
  };

  if (!videoId) {
    return (
      <div className={`w-full h-28 rounded-lg bg-red-100 dark:bg-red-900/30 flex items-center justify-center text-xs text-red-600 dark:text-red-400 ${className}`}>
        Invalid YouTube URL
      </div>
    );
  }

  if (error) {
    return (
      <div className={`w-full h-28 rounded-lg bg-red-100 dark:bg-red-900/30 flex items-center justify-center text-xs text-red-600 dark:text-red-400 ${className}`}>
        {error}
      </div>
    );
  }

  return (
    <div 
      ref={containerRef}
      className={`relative w-full rounded-lg overflow-hidden bg-neutral-200 dark:bg-neutral-800 ${className}`}
      style={{ height: `${calculatedHeight}px` }}
    >
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center text-xs text-neutral-500">
          Loading video...
        </div>
      )}
      {embedUrl && (
        <iframe
          ref={iframeRef}
          className="w-full h-full"
          src={embedUrl}
          title="Video Player"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          onLoad={handleLoad}
          onError={handleError}
        />
      )}
      {endTime && (
        <div className="absolute bottom-1 right-1 bg-black/70 text-white text-[10px] px-1 py-0.5 rounded">
          {Math.floor(startTime)}s - {Math.floor(endTime)}s
        </div>
      )}
    </div>
  );
}