import Darwin
import Foundation

public final class LoopbackHTTPServer: @unchecked Sendable {
    private let router: ServiceRouter
    private let configuration: ServiceConfiguration
    private let acceptQueue = DispatchQueue(label: "indexed.apple-embedding.accept")
    private let clientQueue = DispatchQueue(
        label: "indexed.apple-embedding.clients",
        qos: .userInitiated,
        attributes: .concurrent
    )
    private let stateLock = NSLock()
    private var listener: Int32 = -1
    private var acceptSource: DispatchSourceRead?
    private var clients: Set<Int32> = []

    public private(set) var boundPort: UInt16 = 0

    public init(router: ServiceRouter, configuration: ServiceConfiguration) {
        self.router = router
        self.configuration = configuration
    }

    deinit { stop() }

    public func start() throws {
        guard configuration.host == "127.0.0.1" || configuration.host == "localhost" else {
            throw ServiceError.invalidRequest("原生服务只允许监听 127.0.0.1")
        }
        let descriptor = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)
        guard descriptor >= 0 else { throw posixError("socket") }
        var reuse: Int32 = 1
        guard setsockopt(
            descriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout.size(ofValue: reuse))
        ) == 0 else {
            Darwin.close(descriptor)
            throw posixError("setsockopt")
        }

        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = configuration.port.bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let bindResult = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            Darwin.close(descriptor)
            throw posixError("bind")
        }
        guard listen(descriptor, 32) == 0 else {
            Darwin.close(descriptor)
            throw posixError("listen")
        }
        let currentFlags = fcntl(descriptor, F_GETFL)
        guard currentFlags >= 0, fcntl(descriptor, F_SETFL, currentFlags | O_NONBLOCK) == 0 else {
            Darwin.close(descriptor)
            throw posixError("fcntl(O_NONBLOCK)")
        }

        var actual = sockaddr_in()
        var actualLength = socklen_t(MemoryLayout<sockaddr_in>.size)
        let nameResult = withUnsafeMutablePointer(to: &actual) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(descriptor, $0, &actualLength)
            }
        }
        guard nameResult == 0 else {
            Darwin.close(descriptor)
            throw posixError("getsockname")
        }

        listener = descriptor
        boundPort = UInt16(bigEndian: actual.sin_port)
        let source = DispatchSource.makeReadSource(fileDescriptor: descriptor, queue: acceptQueue)
        source.setEventHandler { [weak self] in self?.acceptReadyClients() }
        source.setCancelHandler { Darwin.close(descriptor) }
        acceptSource = source
        source.resume()
    }

    public func stop() {
        let source = stateLock.withLock { () -> DispatchSourceRead? in
            let value = acceptSource
            acceptSource = nil
            listener = -1
            return value
        }
        source?.cancel()
        let active = stateLock.withLock { () -> [Int32] in
            let value = Array(clients)
            clients.removeAll()
            return value
        }
        for descriptor in active {
            Darwin.shutdown(descriptor, SHUT_RDWR)
        }
    }

    private func acceptReadyClients() {
        while true {
            let descriptor = Darwin.accept(listener, nil, nil)
            if descriptor < 0 {
                if errno == EAGAIN || errno == EWOULDBLOCK { return }
                return
            }
            let clientFlags = fcntl(descriptor, F_GETFL)
            guard clientFlags >= 0,
                  fcntl(descriptor, F_SETFL, clientFlags & ~O_NONBLOCK) == 0 else {
                Darwin.close(descriptor)
                continue
            }
            var timeout = timeval(tv_sec: 300, tv_usec: 0)
            _ = setsockopt(
                descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                socklen_t(MemoryLayout.size(ofValue: timeout))
            )
            var noSignal: Int32 = 1
            _ = setsockopt(
                descriptor, SOL_SOCKET, SO_NOSIGPIPE, &noSignal,
                socklen_t(MemoryLayout.size(ofValue: noSignal))
            )
            _ = stateLock.withLock { clients.insert(descriptor) }
            clientQueue.async { [weak self] in self?.serveClient(descriptor) }
        }
    }

    private func serveClient(_ descriptor: Int32) {
        let cancellation = CancellationToken(timeoutMS: configuration.maxRequestDurationMS)
        var watcher: DispatchSourceTimer?
        defer {
            watcher?.cancel()
            _ = stateLock.withLock { clients.remove(descriptor) }
            Darwin.close(descriptor)
        }
        do {
            let request = try readRequest(descriptor, cancellation: cancellation)
            if request.method == "POST", request.path == "/v1/embeddings" {
                watcher = try watchDisconnect(descriptor, cancellation: cancellation)
            }
            let response = router.route(request, cancellation: cancellation)
            watcher?.cancel()
            try writeResponse(response, to: descriptor)
        } catch let caught as ServiceError {
            let error: ServiceError = cancellation.hasExpired ? .deadlineExceeded : caught
            let payload = try? JSONEncoder().encode(JSONValue.object(["error": .object([
                "message": .string(error.localizedDescription),
                "type": .string(error.responseType),
                "code": .string(error.responseCode),
            ])]))
            try? writeResponse(
                HTTPResponse(
                    status: error.status,
                    body: payload ?? Data(),
                    headers: [
                        "Content-Type": "application/json; charset=utf-8",
                        "Cache-Control": "no-store",
                        "Access-Control-Allow-Origin": "*",
                    ]
                ),
                to: descriptor
            )
        } catch {
            fputs("[apple-embedding] HTTP connection failed: \(error)\n", stderr)
        }
    }

    /// The helper handles one request per connection. EOF before its response
    /// means cancellation; callers keep the read/write connection open until
    /// completion. A duplicated descriptor prevents a close/reuse race.
    private func watchDisconnect(_ descriptor: Int32, cancellation: CancellationToken) throws -> DispatchSourceTimer {
        let observed = dup(descriptor)
        guard observed >= 0 else { throw posixError("dup") }
        let source = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        source.schedule(deadline: .now(), repeating: .milliseconds(25), leeway: .milliseconds(5))
        source.setEventHandler {
            guard !cancellation.isCancelled else { return }
            var byte: UInt8 = 0
            let received = recv(observed, &byte, 1, MSG_PEEK | MSG_DONTWAIT)
            if received == 0 || (received < 0 && ![EAGAIN, EWOULDBLOCK, EINTR].contains(errno)) {
                cancellation.cancel()
            }
        }
        source.setCancelHandler { Darwin.close(observed) }
        source.resume()
        return source
    }

    private func readRequest(_ descriptor: Int32, cancellation: CancellationToken) throws -> HTTPRequest {
        let terminator = Data("\r\n\r\n".utf8)
        var data = Data()
        var headerEnd: Data.Index?
        var contentLength = 0
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)

        while true {
            try cancellation.check()
            let remaining = cancellation.remainingSeconds
            var timeout = timeval(tv_sec: Int(remaining), tv_usec: Int32((remaining.truncatingRemainder(dividingBy: 1) * 1_000_000).rounded(.up)))
            if timeout.tv_usec >= 1_000_000 { timeout.tv_sec += 1; timeout.tv_usec = 0 }
            _ = setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout.size(ofValue: timeout)))
            let received = recv(descriptor, &buffer, buffer.count, 0)
            if received == 0 { throw ServiceError.invalidRequest("连接在请求完成前关闭") }
            if received < 0 {
                if errno == EINTR { continue }
                try cancellation.check()
                throw posixError("recv")
            }
            data.append(buffer, count: received)
            if headerEnd == nil, let range = data.range(of: terminator) {
                guard range.lowerBound <= 64 * 1024 else {
                    throw ServiceError.invalidRequest("HTTP headers 过大", status: 431)
                }
                headerEnd = range.upperBound
                let head = try parseHead(data[..<range.lowerBound])
                try router.configureDeadline(HTTPRequest(method: head.method, path: head.path, headers: head.headers, body: Data()), cancellation: cancellation)
                if let encoding = head.headers["transfer-encoding"], !encoding.isEmpty {
                    throw ServiceError.invalidRequest("不支持 Transfer-Encoding", status: 411)
                }
                if let raw = head.headers["content-length"] {
                    guard let length = Int(raw), length >= 0 else {
                        throw ServiceError.invalidRequest("Content-Length 无效")
                    }
                    contentLength = length
                }
                guard contentLength <= configuration.maxRequestBytes else {
                    throw ServiceError.invalidRequest("请求体超过 Apple helper 限制", status: 413)
                }
            }
            if let headerEnd, data.count >= headerEnd + contentLength {
                let head = try parseHead(data[..<(headerEnd - terminator.count)])
                let path = head.path.split(separator: "?", maxSplits: 1).first.map(String.init)
                    ?? head.path
                return HTTPRequest(
                    method: head.method,
                    path: path,
                    headers: head.headers,
                    body: data.subdata(in: headerEnd ..< headerEnd + contentLength)
                )
            }
            if headerEnd == nil, data.count > 64 * 1024 {
                throw ServiceError.invalidRequest("HTTP headers 过大", status: 431)
            }
        }
    }

    private func parseHead(_ data: Data.SubSequence) throws
        -> (method: String, path: String, headers: [String: String])
    {
        guard let text = String(data: data, encoding: .utf8) else {
            throw ServiceError.invalidRequest("HTTP headers 不是 UTF-8")
        }
        var lines = text.components(separatedBy: "\r\n")
        guard !lines.isEmpty else { throw ServiceError.invalidRequest("缺少 HTTP request line") }
        let requestLine = lines.removeFirst().split(separator: " ", omittingEmptySubsequences: true)
        guard requestLine.count == 3, requestLine[2].hasPrefix("HTTP/1.") else {
            throw ServiceError.invalidRequest("HTTP request line 无效")
        }
        var headers: [String: String] = [:]
        for line in lines where !line.isEmpty {
            guard let separator = line.firstIndex(of: ":") else {
                throw ServiceError.invalidRequest("HTTP header 无效")
            }
            let name = line[..<separator].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: separator)...]
                .trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }
        return (String(requestLine[0]).uppercased(), String(requestLine[1]), headers)
    }

    private func writeResponse(_ response: HTTPResponse, to descriptor: Int32) throws {
        var headers = response.headers
        headers["Content-Length"] = String(response.body.count)
        headers["Connection"] = "close"
        var text = "HTTP/1.1 \(response.status) \(reason(response.status))\r\n"
        for key in headers.keys.sorted() { text += "\(key): \(headers[key]!)\r\n" }
        text += "\r\n"
        try sendAll(Data(text.utf8), to: descriptor)
        try sendAll(response.body, to: descriptor)
    }

    private func sendAll(_ data: Data, to descriptor: Int32) throws {
        try data.withUnsafeBytes { rawBuffer in
            guard let base = rawBuffer.baseAddress else { return }
            var offset = 0
            while offset < rawBuffer.count {
                let sent = Darwin.send(descriptor, base.advanced(by: offset), rawBuffer.count - offset, 0)
                if sent < 0 {
                    if errno == EINTR { continue }
                    throw posixError("send")
                }
                offset += sent
            }
        }
    }

    private func reason(_ status: Int) -> String {
        switch status {
        case 200: "OK"
        case 202: "Accepted"
        case 204: "No Content"
        case 400: "Bad Request"
        case 401: "Unauthorized"
        case 404: "Not Found"
        case 411: "Length Required"
        case 413: "Content Too Large"
        case 429: "Too Many Requests"
        case 431: "Request Header Fields Too Large"
        case 499: "Client Closed Request"
        case 504: "Gateway Timeout"
        case 500: "Internal Server Error"
        default: "Internal Server Error"
        }
    }

    private func posixError(_ operation: String) -> ServiceError {
        .internalFailure("\(operation) 失败：\(String(cString: strerror(errno)))")
    }
}
