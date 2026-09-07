export interface VideoActivityRow {
  video_id?: unknown;
  last_processed_at?: unknown;
  last_indexed_at?: unknown;
  last_session_at?: unknown;
  processing?: unknown;
  pending_count?: unknown;
  visual_count?: unknown;
  [key: string]: unknown;
}

interface QueueActivityRow {
  videoId?: unknown;
  body?: { video_id?: unknown };
  createdAt?: unknown;
}

interface ProcessedActivityRow {
  videoId?: unknown;
  processedAt?: unknown;
}

export function applyVideoActivity(
  videos: VideoActivityRow[],
  queue: QueueActivityRow[],
  processed: ProcessedActivityRow[],
): VideoActivityRow[] {
  const activity = new Map<string, number>();
  for (const item of processed) {
    const videoId = String(item.videoId || "");
    if (videoId) activity.set(videoId, Math.max(activity.get(videoId) || 0, Number(item.processedAt || 0)));
  }
  for (const item of queue) {
    const videoId = String(item.videoId || item.body?.video_id || "");
    if (videoId) activity.set(videoId, Math.max(activity.get(videoId) || 0, Number(item.createdAt || 0)));
  }
  return videos.map((video) => ({
    ...video,
    last_processed_at: Math.max(
      activity.get(String(video.video_id || "")) || 0,
      Number(video.last_processed_at || 0),
      Number(video.last_indexed_at || 0),
      Number(video.last_session_at || 0),
    ),
  })).sort((left, right) =>
    Number(right.processing) - Number(left.processing)
    || Number(right.last_processed_at || 0) - Number(left.last_processed_at || 0)
    || Number(right.pending_count || 0) - Number(left.pending_count || 0)
    || Number(right.visual_count || 0) - Number(left.visual_count || 0)
  );
}
