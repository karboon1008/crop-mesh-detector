import SwiftUI

/// Health of the field cameras: a camera that is offline or flat can't
/// warn anyone, so this is the first place to look when things go quiet.
struct NodesView: View {
    @EnvironmentObject private var store: AlertStore

    var body: some View {
        NavigationStack {
            List {
                if store.isOffline {
                    Section { OfflineBanner(lastUpdated: store.lastUpdated) }
                }
                let attention = store.nodes.filter { !$0.online || ($0.batteryPct ?? 100) < 20 }
                if !attention.isEmpty {
                    Section {
                        Label("\(attention.count) camera\(attention.count == 1 ? " needs" : "s need") attention",
                              systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                    }
                }
                Section("Field cameras") {
                    ForEach(store.nodes) { node in NodeRow(node: node, openAlerts: openAlerts(for: node)) }
                }
            }
            .navigationTitle("Fields")
            .navigationBarTitleDisplayMode(.inline)
            .refreshable { await store.refresh() }
        }
    }

    private func openAlerts(for node: FieldNode) -> Int {
        store.openAlerts.filter { $0.nodeId == node.nodeId }.count
    }
}

struct NodeRow: View {
    let node: FieldNode
    let openAlerts: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Circle().fill(node.online ? Color.green : Color.gray).frame(width: 10, height: 10)
                Text(node.name).font(.headline)
                Spacer()
                if openAlerts > 0 {
                    Text("\(openAlerts) alert\(openAlerts == 1 ? "" : "s")")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Color.red.opacity(0.15), in: Capsule())
                        .foregroundStyle(.red)
                }
            }
            HStack(spacing: 14) {
                if let lastSeen = node.lastSeen {
                    Label { Text(lastSeen, style: .relative) + Text(" ago") } icon: { Image(systemName: "clock") }
                } else {
                    Label("Never connected", systemImage: "clock")
                }
                if let battery = node.batteryPct {
                    Label("\(Int(battery))%", systemImage: batterySymbol(battery))
                        .foregroundStyle(battery < 20 ? .red : .secondary)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            if let coordinate = node.coordinate {
                Text(Coordinates.decimal(coordinate)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private func batterySymbol(_ pct: Double) -> String {
        switch pct {
        case ..<20: "battery.0percent"
        case ..<50: "battery.25percent"
        case ..<75: "battery.50percent"
        default: "battery.100percent"
        }
    }
}
