import Foundation

/// What a detected disease means and what to do first, bundled with the app
/// (Resources/treatments.json) so it works with no signal in the field.
/// Deliberately general first-response guidance: chemical treatments depend
/// on local registration and label rules, so each guide ends by pointing to
/// the local agricultural extension officer.
struct DiseaseGuide: Codable {
    enum Urgency: String, Codable {
        case high, medium, low

        var rank: Int { self == .high ? 3 : self == .medium ? 2 : 1 }
        var title: String {
            switch self {
            case .high: "Act today"
            case .medium: "Act this week"
            case .low: "Monitor"
            }
        }
    }

    let name: String
    let cause: String
    let urgency: Urgency
    let spreads: String
    let lookFor: [String]
    let doNow: [String]
    let prevent: [String]
}

@MainActor
final class TreatmentGuide {
    static let shared = TreatmentGuide()

    private let guides: [String: DiseaseGuide]

    private init() {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        if let url = Bundle.main.url(forResource: "treatments", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let decoded = try? decoder.decode([String: DiseaseGuide].self, from: data) {
            guides = decoded
        } else {
            guides = [:]
        }
    }

    func guide(for disease: String) -> DiseaseGuide {
        guides[disease] ?? DiseaseGuide(
            name: DisplayName.pretty(disease), cause: "Unknown", urgency: .medium,
            spreads: "Unknown, so treat it as spreading until an expert has looked.",
            lookFor: ["Compare the plant with the photo in this alert."],
            doNow: ["Keep people and tools away from the plant until it has been checked.",
                    "Contact your local agricultural extension officer."],
            prevent: []
        )
    }

    static let disclaimer = "General first steps only. Before spraying anything, check with your local agricultural extension officer, and follow the product label and local rules."
}
