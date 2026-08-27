// Surgical ES-module adaptation of fix-webm-duration by Yury Sitnikov.
// Original project: https://github.com/yusitnikov/fix-webm-duration (MIT)
// Only the Info header is rebuilt; encoded clusters remain byte-for-byte intact.
// @ts-nocheck -- binary behavior is covered by compatibility build tests.

const SEGMENT_ID = 0x08538067;
const INFO_ID = 0x0549a966;
const TIMECODE_SCALE_ID = 0x0ad7b1;
const DURATION_ID = 0x0489;

function readVint(source, offset) {
  const first = source[offset];
  if (first === undefined || first === 0) throw new Error("Invalid WebM variable integer");
  let length = 1;
  let mask = 0x80;
  while (length <= 8 && !(first & mask)) {
    length += 1;
    mask >>= 1;
  }
  if (length > 8 || offset + length > source.length) throw new Error("Invalid WebM variable integer length");
  let value = first & (mask - 1);
  let unknown = value === mask - 1;
  for (let index = 1; index < length; index += 1) {
    value = value * 256 + source[offset + index];
    unknown = unknown && source[offset + index] === 0xff;
  }
  return { value: unknown ? -1 : value, length };
}

function encodeVint(value) {
  let length = 1;
  while (length < 8 && value >= 2 ** (7 * length) - 1) length += 1;
  const output = new Uint8Array(length);
  let remaining = value;
  for (let index = length - 1; index >= 0; index -= 1) {
    output[index] = remaining & 0xff;
    remaining = Math.floor(remaining / 256);
  }
  output[0] |= 1 << (8 - length);
  return output;
}

function concatBytes(parts) {
  const output = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}

function parseElement(source, offset, containerEnd) {
  const id = readVint(source, offset);
  const sizeOffset = offset + id.length;
  const size = readVint(source, sizeOffset);
  const dataOffset = sizeOffset + size.length;
  const dataEnd = size.value < 0 ? containerEnd : Math.min(containerEnd, dataOffset + size.value);
  if (dataEnd < dataOffset) throw new Error("Invalid WebM element size");
  return { id: id.value, start: offset, sizeOffset, dataOffset, dataEnd, unknownSize: size.value < 0 };
}

function findChild(source, start, end, targetId) {
  let offset = start;
  while (offset < end) {
    const element = parseElement(source, offset, end);
    if (element.id === targetId) return element;
    if (element.dataEnd <= offset) break;
    offset = element.dataEnd;
  }
  return null;
}

function uintValue(source, start, end) {
  let value = 0;
  for (let offset = start; offset < end; offset += 1) value = value * 256 + source[offset];
  return value;
}

function durationElement(durationUnits) {
  const data = new Uint8Array(8);
  new DataView(data.buffer).setFloat64(0, durationUnits, false);
  return concatBytes([encodeVint(DURATION_ID), encodeVint(data.length), data]);
}

export async function fixWebmDuration(blob, durationMilliseconds) {
  const duration = Number(durationMilliseconds);
  if (!(blob instanceof Blob) || !Number.isFinite(duration) || duration <= 0) return blob;
  try {
    const source = new Uint8Array(await blob.arrayBuffer());
    const segment = findChild(source, 0, source.length, SEGMENT_ID);
    if (!segment) return blob;
    const info = findChild(source, segment.dataOffset, segment.dataEnd, INFO_ID);
    if (!info) return blob;
    const timecodeScale = findChild(source, info.dataOffset, info.dataEnd, TIMECODE_SCALE_ID);
    if (!timecodeScale) return blob;
    const scaleNanoseconds = uintValue(source, timecodeScale.dataOffset, timecodeScale.dataEnd);
    if (!scaleNanoseconds) return blob;

    const replacement = durationElement((duration * 1_000_000) / scaleNanoseconds);
    const existing = findChild(source, info.dataOffset, info.dataEnd, DURATION_ID);
    const infoPayload = existing
      ? concatBytes([
          source.slice(info.dataOffset, existing.start),
          replacement,
          source.slice(existing.dataEnd, info.dataEnd),
        ])
      : concatBytes([source.slice(info.dataOffset, info.dataEnd), replacement]);
    const rebuiltInfo = concatBytes([
      source.slice(info.start, info.sizeOffset),
      encodeVint(infoPayload.length),
      infoPayload,
    ]);
    const segmentPayload = concatBytes([
      source.slice(segment.dataOffset, info.start),
      rebuiltInfo,
      source.slice(info.dataEnd, segment.dataEnd),
    ]);
    const rebuiltSegment = concatBytes([
      source.slice(segment.start, segment.sizeOffset),
      segment.unknownSize
        ? source.slice(segment.sizeOffset, segment.dataOffset)
        : encodeVint(segmentPayload.length),
      segmentPayload,
    ]);
    const repaired = concatBytes([
      source.slice(0, segment.start),
      rebuiltSegment,
      source.slice(segment.dataEnd),
    ]);
    return new Blob([repaired], { type: blob.type || "video/webm" });
  } catch {
    return blob;
  }
}
