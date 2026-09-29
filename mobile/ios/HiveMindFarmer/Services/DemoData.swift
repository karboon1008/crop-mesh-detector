import Foundation

/// Built-in sample farm (same farm and nodes as services/alerts/config.example.json),
/// so the app can be shown with no server and no signal.
enum DemoData {
    struct Snapshot {
        let farmName: String
        let openAlerts: [DiseaseAlert]
        let closedAlerts: [DiseaseAlert]
        let nodes: [FieldNode]
        let risks: [NearbyRisk]
    }

    static let firstDemoAlertID = 900

    static func snapshot(now: Date = .now) -> Snapshot {
        func ago(_ minutes: Double) -> Date { now.addingTimeInterval(-minutes * 60) }

        let open = [
            DiseaseAlert(id: 101, nodeId: "node_0", fieldName: "Tomato greenhouse A", crop: "Tomato",
                         disease: "Late_blight", status: .new, confidence: 0.91, latitude: 4.47219, longitude: 101.37921,
                         firstSeen: ago(95), lastSeen: ago(6), detectionCount: 5, note: nil, imagePath: nil,
                         detections: [
                            DetectionRecord(capturedAt: ago(6), cropConfidence: 0.97, diseaseConfidence: 0.91, latitude: 4.47219, longitude: 101.37921),
                            DetectionRecord(capturedAt: ago(40), cropConfidence: 0.95, diseaseConfidence: 0.86, latitude: 4.47215, longitude: 101.37918),
                            DetectionRecord(capturedAt: ago(95), cropConfidence: 0.93, diseaseConfidence: 0.78, latitude: 4.47212, longitude: 101.37913),
                         ]),
            DiseaseAlert(id: 102, nodeId: "node_2", fieldName: "Pepper row east", crop: "Pepper,_bell",
                         disease: "Bacterial_spot", status: .new, confidence: 0.83, latitude: 4.47036, longitude: 101.38617,
                         firstSeen: ago(180), lastSeen: ago(150), detectionCount: 2, note: nil, imagePath: nil, detections: nil),
            DiseaseAlert(id: 103, nodeId: "node_1", fieldName: "Potato plot north", crop: "Potato",
                         disease: "Early_blight", status: .acknowledged, confidence: 0.77, latitude: 4.47583, longitude: 101.38296,
                         firstSeen: ago(60 * 20), lastSeen: ago(60 * 3), detectionCount: 9, note: nil, imagePath: nil, detections: nil),
        ]
        let closed = [
            DiseaseAlert(id: 98, nodeId: "node_0", fieldName: "Tomato greenhouse A", crop: "Tomato",
                         disease: "Leaf_Mold", status: .treated, confidence: 0.88, latitude: 4.47210, longitude: 101.37909,
                         firstSeen: ago(60 * 24 * 6), lastSeen: ago(60 * 24 * 5), detectionCount: 4,
                         note: "Opened side vents, removed lower leaves", imagePath: nil, detections: nil),
            DiseaseAlert(id: 97, nodeId: "node_2", fieldName: "Pepper row east", crop: "Pepper,_bell",
                         disease: "Bacterial_spot", status: .falseAlarm, confidence: 0.64, latitude: 4.47031, longitude: 101.38622,
                         firstSeen: ago(60 * 24 * 9), lastSeen: ago(60 * 24 * 9), detectionCount: 1,
                         note: "Sun scald, not disease", imagePath: nil, detections: nil),
        ]
        let nodes = [
            FieldNode(nodeId: "node_0", name: "Tomato greenhouse A", latitude: 4.47212, longitude: 101.37913,
                      lastSeen: ago(2), batteryPct: 88, online: true),
            FieldNode(nodeId: "node_1", name: "Potato plot north", latitude: 4.47580, longitude: 101.38301,
                      lastSeen: ago(4), batteryPct: 61, online: true),
            FieldNode(nodeId: "node_2", name: "Pepper row east", latitude: 4.47031, longitude: 101.38622,
                      lastSeen: ago(140), batteryPct: 12, online: false),
        ]
        let risks = [
            NearbyRisk(crop: "Tomato", disease: "Late_blight", latitude: 4.49, longitude: 101.39,
                       distanceKm: 2.5, lastSeen: ago(300)),
        ]
        return Snapshot(farmName: "Tanah Rata Farm (demo)", openAlerts: open, closedAlerts: closed, nodes: nodes, risks: risks)
    }

    /// What a node reporting a new disease would produce, for the demo's "Simulate detection".
    static func simulatedAlert(id: Int, now: Date = .now) -> DiseaseAlert {
        DiseaseAlert(id: id, nodeId: "node_0", fieldName: "Tomato greenhouse A", crop: "Tomato",
                     disease: "Tomato_Yellow_Leaf_Curl_Virus", status: .new, confidence: 0.89,
                     latitude: 4.47226, longitude: 101.37902, firstSeen: now, lastSeen: now, detectionCount: 1,
                     note: nil, imagePath: nil,
                     detections: [DetectionRecord(capturedAt: now, cropConfidence: 0.96, diseaseConfidence: 0.89,
                                                  latitude: 4.47226, longitude: 101.37902)])
    }
}
