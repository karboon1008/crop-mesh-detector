import MapKit
import SwiftUI

/// The whole farm at a glance: open alerts (pins), field cameras (sensors),
/// and outbreaks reported by nearby farms (rounded ~1 km circles).
struct FarmMapView: View {
    @EnvironmentObject private var store: AlertStore
    @EnvironmentObject private var router: AppRouter
    @State private var position: MapCameraPosition = .automatic
    @State private var selection: String?
    @State private var showResolved = false

    private var shownAlerts: [DiseaseAlert] {
        showResolved ? store.openAlerts + store.closedAlerts : store.openAlerts
    }

    /// Nodes with a known position (a node that never reported one can't be drawn).
    private var placedNodes: [(node: FieldNode, coordinate: CLLocationCoordinate2D)] {
        store.nodes.compactMap { node in node.coordinate.map { (node: node, coordinate: $0) } }
    }

    private var selectedAlert: DiseaseAlert? {
        guard let selection, selection.hasPrefix("alert-"), let id = Int(selection.dropFirst(6)) else { return nil }
        return store.alert(id: id)
    }

    var body: some View {
        NavigationStack {
            Map(position: $position, selection: $selection) {
                ForEach(store.risks) { risk in
                    MapCircle(center: risk.coordinate, radius: 1000)
                        .foregroundStyle(.orange.opacity(0.18))
                        .stroke(.orange, lineWidth: 1)
                    Annotation("\(DisplayName.pretty(risk.disease)) nearby", coordinate: risk.coordinate) {
                        Image(systemName: "dot.radiowaves.left.and.right")
                            .padding(6)
                            .background(.orange, in: Circle())
                            .foregroundStyle(.white)
                    }
                }
                ForEach(placedNodes, id: \.node.id) { placed in
                    Annotation(placed.node.name, coordinate: placed.coordinate) {
                        Image(systemName: "sensor.fill")
                            .padding(5)
                            .background(placed.node.online ? Color.blue : Color.gray, in: RoundedRectangle(cornerRadius: 6))
                            .foregroundStyle(.white)
                    }
                }
                ForEach(shownAlerts) { alert in
                    Marker(alert.title, systemImage: alert.status.isOpen ? "leaf.fill" : alert.status.symbol,
                           coordinate: alert.coordinate)
                        .tint(alert.status.color)
                        .tag("alert-\(alert.id)")
                }
                UserAnnotation()
            }
            .mapStyle(.hybrid(elevation: .flat))
            .mapControls {
                MapUserLocationButton()
                MapCompass()
                MapScaleView()
            }
            .safeAreaInset(edge: .bottom) {
                if let alert = selectedAlert {
                    selectedCard(alert)
                }
            }
            .navigationTitle("Farm map")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Toggle(isOn: $showResolved) { Label("Show resolved", systemImage: "clock.arrow.circlepath") }
                        .toggleStyle(.button)
                }
            }
            .onAppear { LocationManager.shared.start() }
        }
    }

    private func selectedCard(_ alert: DiseaseAlert) -> some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(alert.title).font(.headline)
                Text("\(alert.fieldName) · \(Int(alert.confidence * 100))%").font(.subheadline).foregroundStyle(.secondary)
            }
            Spacer()
            Button("Open") { router.openAlert(alert.id) }
                .buttonStyle(.borderedProminent)
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        .padding()
    }
}
