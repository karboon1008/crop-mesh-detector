import Foundation

enum APIError: LocalizedError {
    case notConfigured
    case http(Int, String)
    case transport(Error)

    var errorDescription: String? {
        switch self {
        case .notConfigured: "Set the server address and farm code in Settings."
        case .http(401, _): "The farm code was not accepted. Check it in Settings."
        case let .http(code, body): "Server error \(code): \(body)"
        case let .transport(error): error.localizedDescription
        }
    }
}

/// REST client for services/alerts/app.py. Every call carries the farm's
/// app token as a bearer token.
struct APIClient {
    let baseURL: URL
    let token: String
    var session: URLSession = .shared

    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        decoder.dateDecodingStrategy = .custom { decoder in
            let raw = try decoder.singleValueContainer().decode(String.self)
            if let date = ISO8601.plain.date(from: raw) ?? ISO8601.fractional.date(from: raw) { return date }
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Bad date \(raw)"))
        }
        return decoder
    }()

    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    /// `path` is relative to the server root and may carry a query ("api/alerts?status=open").
    private func url(_ path: String) throws -> URL {
        var root = baseURL.absoluteString
        while root.hasSuffix("/") { root.removeLast() }
        guard let url = URL(string: "\(root)/\(path)") else { throw APIError.notConfigured }
        return url
    }

    private func request(_ path: String, method: String = "GET", body: Data? = nil) async throws -> Data {
        var request = URLRequest(url: try url(path), timeoutInterval: 15)
        request.httpMethod = method
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let result: (Data, URLResponse)
        do {
            result = try await session.data(for: request)
        } catch {
            throw APIError.transport(error)
        }
        let code = (result.1 as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            throw APIError.http(code, String(data: result.0, encoding: .utf8) ?? "")
        }
        return result.0
    }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        try Self.decoder.decode(T.self, from: try await request(path))
    }

    func me() async throws -> FarmProfile { try await get("api/me") }
    func openAlerts() async throws -> [DiseaseAlert] { try await get("api/alerts?status=open") }
    func closedAlerts() async throws -> [DiseaseAlert] { try await get("api/alerts?status=closed") }
    func alert(_ id: Int) async throws -> DiseaseAlert { try await get("api/alerts/\(id)") }
    func nodes() async throws -> [FieldNode] { try await get("api/nodes") }
    func nearbyRisks() async throws -> [NearbyRisk] { try await get("api/nearby-risks") }

    func image(path: String) async throws -> Data {
        try await request(path.hasPrefix("/") ? String(path.dropFirst()) : path)
    }

    func updateStatus(_ id: Int, status: AlertStatus, note: String?) async throws -> DiseaseAlert {
        struct Body: Encodable { let status: String; let note: String? }
        let body = try Self.encoder.encode(Body(status: status.rawValue, note: note))
        return try Self.decoder.decode(DiseaseAlert.self, from: try await request("api/alerts/\(id)", method: "PATCH", body: body))
    }

    func registerDevice(token deviceToken: String) async throws {
        struct Body: Encodable { let token: String; let platform: String }
        let body = try Self.encoder.encode(Body(token: deviceToken, platform: "ios"))
        _ = try await request("api/devices", method: "POST", body: body)
    }
}

enum ISO8601 {
    static let plain: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    static let fractional: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()
}
