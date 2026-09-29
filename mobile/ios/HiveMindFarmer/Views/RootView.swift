import SwiftUI

struct RootView: View {
    @EnvironmentObject private var store: AlertStore
    @EnvironmentObject private var router: AppRouter

    var body: some View {
        TabView(selection: $router.tab) {
            HomeView()
                .tabItem { Label("Home", systemImage: "house.fill") }
                .tag(AppRouter.Tab.home)
            AlertsView()
                .tabItem { Label("Alerts", systemImage: "exclamationmark.bubble.fill") }
                .badge(store.newAlertCount)
                .tag(AppRouter.Tab.alerts)
            FarmMapView()
                .tabItem { Label("Map", systemImage: "map.fill") }
                .tag(AppRouter.Tab.map)
            NodesView()
                .tabItem { Label("Fields", systemImage: "sensor.fill") }
                .tag(AppRouter.Tab.fields)
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape.fill") }
                .tag(AppRouter.Tab.settings)
        }
        .task {
            // While the app is open, check the farm server every 5 s; new alerts show up in the
            // list and as a notification (AlertStore.notifyNewAlerts). Demo mode just reloads.
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                await store.refresh()
            }
        }
    }
}

// MARK: - Shared pieces

struct StatusBadge: View {
    let status: AlertStatus

    var body: some View {
        Label(status.title, systemImage: status.symbol)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(status.color.opacity(0.15), in: Capsule())
            .foregroundStyle(status.color)
    }
}

struct UrgencyBadge: View {
    let urgency: DiseaseGuide.Urgency

    var body: some View {
        Text(urgency.title)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(urgency.color.opacity(0.15), in: Capsule())
            .foregroundStyle(urgency.color)
    }
}

struct OfflineBanner: View {
    let lastUpdated: Date?

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "wifi.slash")
            if let lastUpdated {
                Text("Offline. Showing data from \(lastUpdated, style: .relative) ago.")
            } else {
                Text("Offline. No data yet.")
            }
        }
        .font(.footnote)
        .foregroundStyle(.secondary)
    }
}

extension AlertStatus {
    var color: Color {
        switch self {
        case .new: .red
        case .acknowledged: .orange
        case .treated: .green
        case .falseAlarm: .gray
        }
    }
}

extension DiseaseGuide.Urgency {
    var color: Color {
        switch self {
        case .high: .red
        case .medium: .orange
        case .low: .blue
        }
    }
}
