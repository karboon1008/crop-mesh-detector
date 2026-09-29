import SwiftUI

struct AlertsView: View {
    @EnvironmentObject private var store: AlertStore
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var settings: AppSettings

    var body: some View {
        NavigationStack(path: $router.alertPath) {
            List {
                if store.isOffline {
                    Section { OfflineBanner(lastUpdated: store.lastUpdated) }
                }
                if let error = store.errorMessage, !store.isOffline {
                    Section { Label(error, systemImage: "exclamationmark.circle").foregroundStyle(.red) }
                }

                if !store.risks.isEmpty {
                    Section("Near your farm") {
                        ForEach(store.risks) { risk in
                            Button { router.tab = .map } label: { RiskRow(risk: risk) }
                                .buttonStyle(.plain)
                        }
                    }
                }

                Section("Needs attention") {
                    if store.openAlerts.isEmpty {
                        ContentUnavailableView("No open alerts", systemImage: "leaf.fill",
                                               description: Text("Your field cameras haven't found any disease."))
                    }
                    ForEach(store.openAlerts) { alert in
                        NavigationLink(value: alert.id) { AlertRow(alert: alert) }
                    }
                }

                if !store.closedAlerts.isEmpty {
                    Section("Resolved") {
                        ForEach(store.closedAlerts) { alert in
                            NavigationLink(value: alert.id) { AlertRow(alert: alert) }
                        }
                    }
                }
            }
            .navigationTitle(store.farmName.isEmpty ? "Alerts" : store.farmName)
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: Int.self) { id in AlertDetailView(alertID: id) }
            .refreshable { await store.refresh() }
            .overlay { if store.isLoading && store.openAlerts.isEmpty { ProgressView() } }
        }
    }
}

struct AlertRow: View {
    let alert: DiseaseAlert

    var body: some View {
        let guide = TreatmentGuide.shared.guide(for: alert.disease)
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: alert.status.symbol)
                .font(.title2)
                .foregroundStyle(alert.status.color)
                .frame(width: 30)
            VStack(alignment: .leading, spacing: 4) {
                Text(alert.title).font(.headline)
                Text(alert.fieldName).font(.subheadline).foregroundStyle(.secondary)
                HStack(spacing: 6) {
                    if alert.status.isOpen { UrgencyBadge(urgency: guide.urgency) } else { StatusBadge(status: alert.status) }
                    Text("\(Int(alert.confidence * 100))%").font(.caption).foregroundStyle(.secondary)
                    if alert.detectionCount > 1 {
                        Text("· seen \(alert.detectionCount)×").font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            Spacer()
            Text(alert.lastSeen, style: .relative)
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
        .padding(.vertical, 4)
    }
}

struct RiskRow: View {
    let risk: NearbyRisk

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "dot.radiowaves.left.and.right")
                .font(.title2)
                .foregroundStyle(.orange)
                .frame(width: 30)
            VStack(alignment: .leading, spacing: 2) {
                Text("\(DisplayName.pretty(risk.disease)) about \(risk.distanceKm, specifier: "%g") km away")
                    .font(.subheadline.weight(.semibold))
                Text("On \(DisplayName.pretty(risk.crop).lowercased()) at another farm. Check your \(DisplayName.pretty(risk.crop).lowercased()) plants.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}
