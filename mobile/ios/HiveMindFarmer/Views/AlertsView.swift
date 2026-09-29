import SwiftUI

/// How the farmer arranges the alert lists. Each has a natural order (newest, most urgent,
/// nearest first); `reversed` flips it.
enum AlertSort: String, CaseIterable, Identifiable {
    case detectionTime, actionNeeded, nearest

    var id: String { rawValue }

    var title: String {
        switch self {
        case .detectionTime: "Detection time"
        case .actionNeeded: "Action needed"
        case .nearest: "Nearest"
        }
    }

    var symbol: String {
        switch self {
        case .detectionTime: "clock"
        case .actionNeeded: "exclamationmark.triangle"
        case .nearest: "location"
        }
    }

    func orderTitle(reversed: Bool) -> String {
        switch self {
        case .detectionTime: reversed ? "Oldest first" : "Newest first"
        case .actionNeeded: reversed ? "Least urgent first" : "Most urgent first"
        case .nearest: reversed ? "Farthest first" : "Nearest first"
        }
    }
}

struct AlertsView: View {
    @EnvironmentObject private var store: AlertStore
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var location: LocationManager
    @AppStorage("alertSort") private var sort: AlertSort = .actionNeeded
    @AppStorage("alertSortReversed") private var reversed = false

    var body: some View {
        NavigationStack(path: $router.alertPath) {
            List {
                if store.isOffline {
                    Section { OfflineBanner(lastUpdated: store.lastUpdated) }
                }
                if let error = store.errorMessage, !store.isOffline {
                    Section { Label(error, systemImage: "exclamationmark.circle").foregroundStyle(.red) }
                }

                // A disease on this farm comes first; with none, nearby farms' outbreaks lead.
                if store.openAlerts.isEmpty {
                    nearbySection
                    needsAttentionSection
                } else {
                    needsAttentionSection
                    nearbySection
                }

                if !store.closedAlerts.isEmpty {
                    Section("Resolved") {
                        ForEach(arranged(store.closedAlerts)) { alert in
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
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) { sortMenu }
            }
            .onAppear { if sort == .nearest { location.start() } }
            .onChange(of: sort) { _, newSort in if newSort == .nearest { location.start() } }
        }
    }

    // MARK: - Sorting

    private var sortMenu: some View {
        Menu {
            Picker("Sort by", selection: $sort) {
                ForEach(AlertSort.allCases) { option in
                    Label(option.title, systemImage: option.symbol).tag(option)
                }
            }
            Picker("Order", selection: $reversed) {
                Text(sort.orderTitle(reversed: false)).tag(false)
                Text(sort.orderTitle(reversed: true)).tag(true)
            }
        } label: {
            Label("Sort", systemImage: "arrow.up.arrow.down")
        }
    }

    /// Metres from the farmer to the plant, or nil before the phone knows where it is.
    private func distance(to alert: DiseaseAlert) -> Double? {
        location.route(to: alert.coordinate)?.meters
    }

    private func arranged(_ alerts: [DiseaseAlert]) -> [DiseaseAlert] {
        let natural: [DiseaseAlert]
        switch sort {
        case .detectionTime:
            natural = alerts.sorted { $0.lastSeen > $1.lastSeen }
        case .actionNeeded:
            natural = alerts  // the store already orders alerts by attention
        case .nearest:
            natural = alerts.sorted { (distance(to: $0) ?? .infinity) < (distance(to: $1) ?? .infinity) }
        }
        return reversed ? Array(natural.reversed()) : natural
    }

    private var arrangedRisks: [NearbyRisk] {
        let natural: [NearbyRisk]
        switch sort {
        case .detectionTime: natural = store.risks.sorted { $0.lastSeen > $1.lastSeen }
        case .actionNeeded: natural = store.risks
        case .nearest: natural = store.risks.sorted { $0.distanceKm < $1.distanceKm }
        }
        return reversed ? Array(natural.reversed()) : natural
    }

    @ViewBuilder
    private var nearbySection: some View {
        if !store.risks.isEmpty {
            Section("Near your farm") {
                ForEach(arrangedRisks) { risk in
                    Button { router.tab = .map } label: { RiskRow(risk: risk) }
                        .buttonStyle(.plain)
                }
            }
        }
    }

    private var needsAttentionSection: some View {
        Section {
            if store.openAlerts.isEmpty {
                ContentUnavailableView("No open alerts", systemImage: "leaf.fill",
                                       description: Text("Your field cameras haven't found any disease."))
            }
            ForEach(arranged(store.openAlerts)) { alert in
                NavigationLink(value: alert.id) { AlertRow(alert: alert) }
            }
        } header: {
            HStack {
                Text("Needs attention")
                Spacer()
                Text(sort.orderTitle(reversed: reversed)).textCase(nil)
            }
        } footer: {
            if sort == .nearest && location.location == nil && !store.openAlerts.isEmpty {
                Text("Turn on location to sort by distance.")
            }
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
