import CoreLocation
import Foundation

/// Mirrors services/alerts/app.py. JSON is snake_case; APIClient decodes with
/// `.convertFromSnakeCase` and ISO-8601 dates.

enum AlertStatus: String, Codable, CaseIterable, Identifiable {
    case new, acknowledged, treated
    case falseAlarm = "false_alarm"

    var id: String { rawValue }
    var isOpen: Bool { self == .new || self == .acknowledged }

    var title: String {
        switch self {
        case .new: "New"
        case .acknowledged: "Seen"
        case .treated: "Treated"
        case .falseAlarm: "False alarm"
        }
    }

    var symbol: String {
        switch self {
        case .new: "exclamationmark.triangle.fill"
        case .acknowledged: "eye.fill"
        case .treated: "checkmark.seal.fill"
        case .falseAlarm: "xmark.circle.fill"
        }
    }
}

struct DetectionRecord: Codable, Hashable {
    let capturedAt: Date
    let cropConfidence: Double
    let diseaseConfidence: Double
    let latitude: Double
    let longitude: Double
}

struct DiseaseAlert: Codable, Identifiable, Hashable {
    let id: Int
    let nodeId: String
    let fieldName: String
    let crop: String
    let disease: String
    var status: AlertStatus
    let confidence: Double
    let latitude: Double
    let longitude: Double
    let firstSeen: Date
    let lastSeen: Date
    let detectionCount: Int
    var note: String?
    let imagePath: String?
    var detections: [DetectionRecord]?

    var coordinate: CLLocationCoordinate2D { .init(latitude: latitude, longitude: longitude) }
    var title: String { "\(DisplayName.pretty(crop)) · \(DisplayName.pretty(disease))" }
}

struct FieldNode: Codable, Identifiable, Hashable {
    let nodeId: String
    let name: String
    let latitude: Double?
    let longitude: Double?
    let lastSeen: Date?
    let batteryPct: Double?
    let online: Bool

    var id: String { nodeId }
    var coordinate: CLLocationCoordinate2D? {
        guard let latitude, let longitude else { return nil }
        return .init(latitude: latitude, longitude: longitude)
    }
}

/// Another farm's outbreak near ours: rounded (~1 km) location, no identity.
struct NearbyRisk: Codable, Identifiable, Hashable {
    let crop: String
    let disease: String
    let latitude: Double
    let longitude: Double
    let distanceKm: Double
    let lastSeen: Date

    var id: String { "\(crop)|\(disease)|\(latitude)|\(longitude)" }
    var coordinate: CLLocationCoordinate2D { .init(latitude: latitude, longitude: longitude) }
}

struct FarmProfile: Codable {
    let farmId: String
    let farmName: String
}

enum DisplayName {
    /// "Pepper,_bell" -> "Pepper bell", "Tomato_Yellow_Leaf_Curl_Virus" -> "Tomato Yellow Leaf Curl Virus"
    static func pretty(_ raw: String) -> String {
        raw.replacingOccurrences(of: ",", with: "")
            .replacingOccurrences(of: "_", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }
}

enum Coordinates {
    static func decimal(_ c: CLLocationCoordinate2D) -> String {
        String(format: "%.5f, %.5f", c.latitude, c.longitude)
    }

    /// 4°28'19.6"N 101°22'44.9"E
    static func dms(_ c: CLLocationCoordinate2D) -> String {
        func part(_ value: Double, _ pos: String, _ neg: String) -> String {
            let v = abs(value)
            let deg = Int(v)
            let minFull = (v - Double(deg)) * 60
            let min = Int(minFull)
            let sec = (minFull - Double(min)) * 60
            return String(format: "%d°%02d'%04.1f\"%@", deg, min, sec, value >= 0 ? pos : neg)
        }
        return "\(part(c.latitude, "N", "S")) \(part(c.longitude, "E", "W"))"
    }

    static func appleMapsURL(_ c: CLLocationCoordinate2D, label: String) -> URL {
        var components = URLComponents(string: "https://maps.apple.com/")!
        components.queryItems = [
            .init(name: "ll", value: "\(c.latitude),\(c.longitude)"),
            .init(name: "q", value: label),
        ]
        return components.url!
    }
}
