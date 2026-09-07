import { repairWebmBase64Duration } from "@indexed/clients/webm-duration";

export interface CapturedVideoIngestBody {
  video_filename?: unknown;
  video_base64?: unknown;
  video_frames?: unknown;
  captured_seconds?: unknown;
  [key: string]: unknown;
}

export async function prepareCapturedVideoForIngest(
  body: CapturedVideoIngestBody,
): Promise<CapturedVideoIngestBody> {
  if (Array.isArray(body.video_frames) && body.video_frames.length) return body;
  const filename = String(body.video_filename || "").toLowerCase();
  const capturedMilliseconds = Number(body.captured_seconds || 0) * 1000;
  if (filename.endsWith(".mp4") || filename.endsWith(".m4v") || capturedMilliseconds <= 0) return body;
  const repaired = await repairWebmBase64Duration(
    String(body.video_base64 || ""),
    capturedMilliseconds,
  );
  return { ...body, video_base64: repaired.videoBase64 };
}
