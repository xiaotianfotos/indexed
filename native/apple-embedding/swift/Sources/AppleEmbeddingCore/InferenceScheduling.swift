import Foundation

/// Bounded FIFO permits with an explicit priority lane and cooperative cancel.
/// Used for request admission, single-owner stages, and D's shared ANE lane.
final class InferenceGate: @unchecked Sendable {
    private struct Waiter { let id: UUID; let priority: Bool }
    private let condition = NSCondition()
    let capacity: Int
    private var active = 0
    private var peak = 0
    private var waiting: [Waiter] = []

    init(capacity: Int = 1) {
        precondition(capacity > 0 && capacity <= 2)
        self.capacity = capacity
    }

    var snapshot: (active: Int, peak: Int, waiting: Int, priorityWaiting: Int) {
        condition.lock(); defer { condition.unlock() }
        return (active, peak, waiting.count, waiting.filter(\.priority).count)
    }

    func withPermit<T>(priority: Bool = false, cancellation: CancellationToken, _ body: () throws -> T) throws -> T {
        let id = UUID()
        condition.lock()
        waiting.append(Waiter(id: id, priority: priority))
        while true {
            do { try cancellation.check() }
            catch {
                waiting.removeAll { $0.id == id }
                condition.broadcast(); condition.unlock()
                throw error
            }
            let first = waiting.first(where: \.priority) ?? waiting.first
            if active < capacity, first?.id == id {
                waiting.removeAll { $0.id == id }
                active += 1; peak = max(peak, active)
                condition.broadcast(); condition.unlock()
                break
            }
            _ = condition.wait(until: Date(timeIntervalSinceNow: 0.02))
        }
        defer {
            condition.lock(); active -= 1
            condition.broadcast(); condition.unlock()
        }
        try cancellation.check()
        return try body()
    }
}

/// Runtime evidence of bounded admission and actual stage overlap. Keep metrics
/// independent of model/ANE locks so health remains readable during inference.
final class InferencePipeline: @unchecked Sendable {
    enum Stage { case vision, language }
    let admission: InferenceGate
    let vision = InferenceGate()
    let language = InferenceGate()
    let ane = InferenceGate()
    private let lock = NSLock()
    private var visionActive = 0
    private var languageActive = 0
    private var overlapStarted: Double?
    private var overlapMS: Double = 0
    private var overlapEvents = 0

    init(depth: Int) { admission = InferenceGate(capacity: depth) }

    func withStage<T>(_ stage: Stage, cancellation: CancellationToken, _ body: () throws -> T) throws -> T {
        try (stage == .vision ? vision : language).withPermit(cancellation: cancellation) {
            transition(stage, delta: 1)
            defer { transition(stage, delta: -1) }
            return try body()
        }
    }

    private func transition(_ stage: Stage, delta: Int) {
        lock.withLock {
            if stage == .vision { visionActive += delta } else { languageActive += delta }
            let now = ProcessInfo.processInfo.systemUptime
            if visionActive > 0 && languageActive > 0 {
                if overlapStarted == nil { overlapStarted = now; overlapEvents += 1 }
            } else if let start = overlapStarted {
                overlapMS += (now - start) * 1000; overlapStarted = nil
            }
        }
    }

    var diagnostics: JSONValue {
        let admission = admission.snapshot, vision = vision.snapshot, language = language.snapshot, ane = ane.snapshot
        return lock.withLock { .object([
            "pipeline_depth": .number(Double(self.admission.capacity)),
            "in_flight": .number(Double(admission.active)), "peak_in_flight": .number(Double(admission.peak)),
            "vision_active": .number(Double(vision.active)), "language_active": .number(Double(language.active)),
            "peak_vision_active": .number(Double(vision.peak)), "peak_language_active": .number(Double(language.peak)),
            "vision_language_overlap_events": .number(Double(overlapEvents)),
            "vision_language_overlap_ms": .number(overlapMS + (overlapStarted.map { (ProcessInfo.processInfo.systemUptime - $0) * 1000 } ?? 0)),
            "ane_priority_waiters": .number(Double(ane.priorityWaiting)),
            "peak_ane_lane_active": .number(Double(ane.peak)),
        ]) }
    }
}
