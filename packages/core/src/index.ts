import type {
  SearchRequest,
  SearchResponse,
  VideoIdentity,
  VideoMemory,
} from "@indexed/contracts";

export interface VideoMemoryRepository {
  search(request: SearchRequest): Promise<SearchResponse>;
  list(limit: number): Promise<VideoMemory[]>;
  update(identity: VideoIdentity, patch: Partial<Pick<VideoMemory, "title" | "note" | "tags">>): Promise<VideoMemory>;
  delete(identity: VideoIdentity): Promise<{ deletedVisual: number; deletedTranscript: number }>;
}

export class IndexedApplication {
  constructor(private readonly memories: VideoMemoryRepository) {}

  search(request: SearchRequest): Promise<SearchResponse> {
    return this.memories.search(request);
  }

  listVideos(limit = 100): Promise<VideoMemory[]> {
    return this.memories.list(limit);
  }

  updateVideo(
    identity: VideoIdentity,
    patch: Partial<Pick<VideoMemory, "title" | "note" | "tags">>,
  ): Promise<VideoMemory> {
    return this.memories.update(identity, patch);
  }

  deleteVideo(identity: VideoIdentity, confirmed: boolean): Promise<{ deletedVisual: number; deletedTranscript: number }> {
    if (!confirmed) throw new Error("Deleting cloud vectors requires explicit confirmation.");
    return this.memories.delete(identity);
  }
}

export { deleteVideo, prepareVideoDeletion, listVideos, search, searchImage, status, updateVideo } from "./library.js";
export { ingest, ingestTranscript, ingestVideo, ingestVisual, prepareVideoEmbeddingInput } from "./ingest.js";
export { listLocalFiles, scanLocalLibrary, searchLocalFiles } from "./local-library.js";
export {
  addLibrary,
  ASSET_EXTENSIONS,
  AssetAccessError,
  assetModelInfo,
  browseAssetDirectories,
  cancelScan,
  hasActiveAssetScans,
  libraryCoverage,
  EDITABLE_ASSET_EXTENSIONS,
  librariesStatus,
  listAssets,
  pruneMissingAssets,
  queueScan,
  readAssetDocument,
  removeLibrary,
  resolveAssetForEditing,
  resolveAssetForServing,
  resolveAssetPreview,
  scanJob,
  scanLibrary,
  searchAssets,
  startAutoScan,
  stopAutoScan,
  writeAssetDocument,
} from "./asset-library.js";
export { searchAllAssets } from "./search-assets.js";
export {
  chunkDocumentText,
  documentFormatCapabilities,
  extractDocumentText,
  DOCUMENT_CHUNK_VERSION,
} from "./document-assets.js";
export type { DocumentChunk, DocumentChunks, DocumentFormatCapability } from "./document-assets.js";
export type {
  AssetDocument,
  AssetDocumentWrite,
  AssetHit,
  AssetJob,
  AssetKind,
  AssetLibrariesStatus,
  AssetLibraryEntry,
  CoverageResult,
  DirectoryBrowseResult,
  DirectoryEntry,
  LibraryCoverage,
  ListOptions,
  PruneOptions,
  PruneResult,
  ScanOptions,
  ScanResult,
  SearchAssetsResult,
  SearchOptions,
  ServedAsset,
} from "./asset-library.js";
export type { SearchAllAssetsResult } from "./search-assets.js";
export {
  estimateIngestPerformance,
  readIngestPerformanceHistory,
  recordIngestPerformance,
} from "./ingest-performance.js";
export type {
  IngestPerformanceEstimate,
  IngestPerformanceObservation,
  IngestPerformanceSample,
  KindCounters,
  KindPerformanceRate,
  PerformanceAssetKind,
} from "./ingest-performance.js";
