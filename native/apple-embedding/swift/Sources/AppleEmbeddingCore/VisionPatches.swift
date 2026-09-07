import Foundation

/// CPU-only vision staging; no MLX arrays or streams cross the stage boundary.
/// The 256-value normalization table is materialized once with the original MLX
/// expression during model load, preserving its exact FP32 rounding.
struct VisionPatches: Sendable {
    let values: [Float]
    let rows: Int
    static let columns = 3 * 2 * 16 * 16

    static func normalizedCHW(_ rgba: [UInt8], side: Int, table: [Float]) -> [Float] {
        precondition(table.count == 256 && rgba.count == side * side * 4)
        let pixels = side * side
        var result = [Float](repeating: 0, count: pixels * 3)
        for channel in 0..<3 {
            let channelBase = channel * pixels
            for pixel in 0..<pixels { result[channelBase + pixel] = table[Int(rgba[pixel * 4 + channel])] }
        }
        return result
    }

    static func pair(_ first: [Float], _ second: [Float], side: Int) -> VisionPatches {
        precondition(side > 0 && side.isMultiple(of: 32))
        precondition(first.count == side * side * 3 && second.count == first.count)
        let grid = side / 16
        var result = [Float](repeating: 0, count: grid * grid * columns)
        // Match reshape [1,2,3,gh/2,2,16,gw/2,2,16], transpose
        // [0,3,6,4,7,2,1,5,8], then flatten [gh*gw,1536].
        for blockY in 0..<(grid / 2) {
            for blockX in 0..<(grid / 2) {
                for mergeY in 0..<2 {
                    for mergeX in 0..<2 {
                        let patch = ((blockY * (grid / 2) + blockX) * 2 + mergeY) * 2 + mergeX
                        for channel in 0..<3 {
                            for temporal in 0..<2 {
                                let frame = temporal == 0 ? first : second
                                let destination = patch * columns + (channel * 2 + temporal) * 256
                                let origin = channel * side * side + (blockY * 32 + mergeY * 16) * side + blockX * 32 + mergeX * 16
                                for y in 0..<16 {
                                    for x in 0..<16 { result[destination + y * 16 + x] = frame[origin + y * side + x] }
                                }
                            }
                        }
                    }
                }
            }
        }
        return VisionPatches(values: result, rows: grid * grid)
    }
}

struct VisionFeatures: Sendable {
    let values: [Float]
    let shape: [Int]
}
