import Foundation

/// Byte-for-byte compatible implementation of Pillow's 8-bit BICUBIC resize.
///
/// WeMM's published Python path performs a forced Pillow resize to 448×448
/// before Qwen patchification.  Core Image's similarly named filter uses a
/// different downsampling kernel, which is enough to move image embeddings in
/// the unified text/image space.  This is a small Swift port of Pillow's
/// `libImaging/Resample.c` coefficient and two-pass integer code.
enum PillowBicubic {
    private static let precisionBits = 22
    private static let coefficientScale = 1 << precisionBits
    private static let rounding = 1 << (precisionBits - 1)

    private struct Coefficients {
        let starts: [Int]
        let counts: [Int]
        let kernelSize: Int
        let values: [Int32]
    }

    static func resizeRGBA(
        _ source: [UInt8],
        width sourceWidth: Int,
        height sourceHeight: Int,
        to targetWidth: Int,
        _ targetHeight: Int
    ) -> [UInt8] {
        precondition(source.count == sourceWidth * sourceHeight * 4)
        let horizontal = coefficients(input: sourceWidth, output: targetWidth)
        let vertical = coefficients(input: sourceHeight, output: targetHeight)

        var intermediate = [UInt8](
            repeating: 255,
            count: targetWidth * sourceHeight * 4
        )
        for y in 0 ..< sourceHeight {
            for x in 0 ..< targetWidth {
                let start = horizontal.starts[x]
                let count = horizontal.counts[x]
                let kernelOffset = x * horizontal.kernelSize
                let destination = (y * targetWidth + x) * 4
                for channel in 0 ..< 3 {
                    var sum = rounding
                    for tap in 0 ..< count {
                        let value = Int(source[(y * sourceWidth + start + tap) * 4 + channel])
                        sum += value * Int(horizontal.values[kernelOffset + tap])
                    }
                    intermediate[destination + channel] = clipped(sum >> precisionBits)
                }
            }
        }

        var output = [UInt8](repeating: 255, count: targetWidth * targetHeight * 4)
        for y in 0 ..< targetHeight {
            let start = vertical.starts[y]
            let count = vertical.counts[y]
            let kernelOffset = y * vertical.kernelSize
            for x in 0 ..< targetWidth {
                let destination = (y * targetWidth + x) * 4
                for channel in 0 ..< 3 {
                    var sum = rounding
                    for tap in 0 ..< count {
                        let value = Int(intermediate[((start + tap) * targetWidth + x) * 4 + channel])
                        sum += value * Int(vertical.values[kernelOffset + tap])
                    }
                    output[destination + channel] = clipped(sum >> precisionBits)
                }
            }
        }
        return output
    }

    private static func coefficients(input: Int, output: Int) -> Coefficients {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1.0)
        let support = 2.0 * filterScale
        let kernelSize = Int(ceil(support)) * 2 + 1
        let inverseFilterScale = 1.0 / filterScale
        var starts = [Int](repeating: 0, count: output)
        var counts = [Int](repeating: 0, count: output)
        var values = [Int32](repeating: 0, count: output * kernelSize)

        for destination in 0 ..< output {
            let center = (Double(destination) + 0.5) * scale
            let minimum = max(0, Int(center - support + 0.5))
            let maximum = min(input, Int(center + support + 0.5))
            let count = maximum - minimum
            var weights = [Double](repeating: 0, count: count)
            var total = 0.0
            for tap in 0 ..< count {
                let distance = (Double(tap + minimum) - center + 0.5) * inverseFilterScale
                let weight = bicubic(distance)
                weights[tap] = weight
                total += weight
            }
            starts[destination] = minimum
            counts[destination] = count
            if total != 0 {
                for tap in 0 ..< count {
                    let normalized = weights[tap] / total
                    let scaled = normalized * Double(coefficientScale)
                    let rounded = normalized < 0 ? Int(scaled - 0.5) : Int(scaled + 0.5)
                    values[destination * kernelSize + tap] = Int32(rounded)
                }
            }
        }
        return Coefficients(
            starts: starts,
            counts: counts,
            kernelSize: kernelSize,
            values: values
        )
    }

    private static func bicubic(_ input: Double) -> Double {
        let x = abs(input)
        let a = -0.5
        if x < 1.0 {
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        }
        if x < 2.0 {
            return (((x - 5.0) * x + 8.0) * x - 4.0) * a
        }
        return 0.0
    }

    private static func clipped(_ value: Int) -> UInt8 {
        UInt8(clamping: value)
    }
}
