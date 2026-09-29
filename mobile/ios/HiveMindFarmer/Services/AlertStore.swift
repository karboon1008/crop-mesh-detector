import Foundation

/// The app's single source of truth for alerts, field nodes and nearby risks.
///
/// Offline first: every successful refresh is cached to disk, and when the
/// server can't be reached (common in the field) the last snapshot is shown
/// with an "offline" banner. Status changes made offline are applied on the
/// phone straight away and queued, then sent on the next refresh that reaches
/// the server.
@MainActor
final class AlertStore: ObservableObject {
    static let shared = AlertStore()

    @Published private(set) var openAlerts: [DiseaseAlert] = []
    @Published private(set) var closedAlerts: [DiseaseAlert] = []
    @Published private(set) var nodes: [FieldNode] = []
    @Published private(set) var risks: [NearbyRisk] = []
    @Published private(set) var farmName: String = ""
    @Published private(set) var lastUpdated: Date?
    @Published private(set) var isOffline = false
    @Published private(set) var isLoading = false
    @Published var errorMessage: String?

    private var pendingUpdates: [PendingUpdate] = []
    private var imageCache: [Int: Data] = [:]
    private let settings = AppSettings.shared

    private struct PendingUpdate: Codable {
        let alertId: Int
        let status: AlertStatus
        let note: String?
    }

    private struct Snapshot: Codable {
        var farmName: String
        var openAlerts: [DiseaseAlert]
        var closedAlerts: [DiseaseAlert]
        var nodes: [FieldNode]
        var risks: [NearbyRisk]
        var lastUpdated: Date
        var pendingUpdates: [PendingUpdate]
    }

    private init() {
        loadCache()
    }

    var newAlertCount: Int { openAlerts.filter { $0.status == .new }.count }

    func alert(id: Int) -> DiseaseAlert? {
        openAlerts.first { $0.id == id } ?? closedAlerts.first { $0.id == id }
    }

    // MARK: - Refresh

    func refresh() async {
        if settings.demoMode {
            loadDemo()
            return
        }
        guard !isLoading else { return }  // the 5 s poll and a pull-to-refresh can overlap
        guard let client = settings.client else {
            errorMessage = APIError.notConfigured.errorDescription
            return
        }
        isLoading = true
        defer { isLoading = false }
        await flushPending(client)
        do {
            async let me = client.me()
            async let open = client.openAlerts()
            async let closed = client.closedAlerts()
            async let nodes = client.nodes()
            async let risks = client.nearbyRisks()
            let (profile, openList, closedList, nodeList, riskList) = try await (me, open, closed, nodes, risks)
            farmName = profile.farmName
            openAlerts = sortForAttention(applyPending(openList))
            closedAlerts = closedList
            self.nodes = nodeList
            self.risks = riskList
            lastUpdated = .now
            isOffline = false
            errorMessage = nil
            saveCache()
            notifyNewAlerts(openList)
        } catch {
            isOffline = true
            errorMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    /// Full alert with its detection history (falls back to the list copy offline).
    func detail(id: Int) async -> DiseaseAlert? {
        guard let client = settings.client else { return alert(id: id) }
        do {
            let full = try await client.alert(id)
            replace(full)
            return full
        } catch {
            return alert(id: id)
        }
    }

    func imageData(for alert: DiseaseAlert) async -> Data? {
        if let cached = imageCache[alert.id] { return cached }
        guard let path = alert.imagePath, let client = settings.client,
              let data = try? await client.image(path: path) else { return nil }
        imageCache[alert.id] = data
        return data
    }

    // MARK: - Farmer feedback

    func setStatus(_ status: AlertStatus, note: String? = nil, for id: Int) async {
        guard var alert = alert(id: id) else { return }
        alert.status = status
        if let note, !note.isEmpty { alert.note = note }
        replace(alert)  // optimistic: the farmer sees it change immediately

        guard !settings.demoMode, let client = settings.client else { return }
        do {
            replace(try await client.updateStatus(id, status: status, note: note))
        } catch {
            pendingUpdates.removeAll { $0.alertId == id }
            pendingUpdates.append(PendingUpdate(alertId: id, status: status, note: note))
            isOffline = true
        }
        saveCache()
    }

    private func flushPending(_ client: APIClient) async {
        var remaining: [PendingUpdate] = []
        for update in pendingUpdates {
            do {
                _ = try await client.updateStatus(update.alertId, status: update.status, note: update.note)
            } catch {
                remaining.append(update)
            }
        }
        pendingUpdates = remaining
    }

    private func applyPending(_ alerts: [DiseaseAlert]) -> [DiseaseAlert] {
        alerts.map { alert in
            guard let update = pendingUpdates.last(where: { $0.alertId == alert.id }) else { return alert }
            var copy = alert
            copy.status = update.status
            return copy
        }
    }

    private func replace(_ alert: DiseaseAlert) {
        openAlerts.removeAll { $0.id == alert.id }
        closedAlerts.removeAll { $0.id == alert.id }
        if alert.status.isOpen {
            openAlerts = sortForAttention(openAlerts + [alert])
        } else {
            closedAlerts.insert(alert, at: 0)
        }
    }

    /// New before seen, then most urgent disease, then most recent.
    private func sortForAttention(_ alerts: [DiseaseAlert]) -> [DiseaseAlert] {
        alerts.sorted {
            if ($0.status == .new) != ($1.status == .new) { return $0.status == .new }
            let u0 = TreatmentGuide.shared.guide(for: $0.disease).urgency.rank
            let u1 = TreatmentGuide.shared.guide(for: $1.disease).urgency.rank
            if u0 != u1 { return u0 > u1 }
            return $0.lastSeen > $1.lastSeen
        }
    }

    // MARK: - Demo + cache

    private func loadDemo() {
        let demo = DemoData.snapshot()
        farmName = demo.farmName
        if lastUpdated == nil {  // first load only, so the farmer's demo taps (treated, seen...) stick
            openAlerts = sortForAttention(demo.openAlerts)
            closedAlerts = demo.closedAlerts
        }
        nodes = demo.nodes
        risks = demo.risks
        lastUpdated = .now
        isOffline = false
        errorMessage = nil
    }

    /// Demo mode: a simulated detection lands as a brand-new alert.
    func addDemoAlert(_ alert: DiseaseAlert) {
        replace(alert)
    }

    // MARK: - Notifications for new server alerts

    private static let notifiedKey = "notifiedAlertIDs"
    private static let primedKey = "notifiedAlertsPrimed"

    /// Shows a notification for each alert the server has that this phone hasn't announced yet.
    /// Works with a free Apple ID: the app polls the server (every 5 s while open, see RootView)
    /// and posts a local notification, instead of the server pushing through APNs. The first
    /// load only records what's already there, so opening the app doesn't replay old alerts.
    private func notifyNewAlerts(_ alerts: [DiseaseAlert]) {
        let defaults = UserDefaults.standard
        var notified = Set(defaults.array(forKey: Self.notifiedKey) as? [Int] ?? [])
        let primed = defaults.bool(forKey: Self.primedKey)
        for alert in alerts where !notified.contains(alert.id) {
            notified.insert(alert.id)
            if primed && alert.status == .new {
                NotificationAction.scheduleAlertNotification(for: alert, delay: 0)
            }
        }
        defaults.set(Array(notified), forKey: Self.notifiedKey)
        defaults.set(true, forKey: Self.primedKey)
    }

    func resetForModeChange() {
        UserDefaults.standard.removeObject(forKey: Self.notifiedKey)
        UserDefaults.standard.removeObject(forKey: Self.primedKey)
        openAlerts = []
        closedAlerts = []
        nodes = []
        risks = []
        lastUpdated = nil
        imageCache = [:]
        if !settings.demoMode { loadCache() }
    }

    private var cacheURL: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appending(path: "alerts-cache.json")
    }

    private func saveCache() {
        guard !settings.demoMode, let lastUpdated else { return }
        let snapshot = Snapshot(farmName: farmName, openAlerts: openAlerts, closedAlerts: closedAlerts,
                                nodes: nodes, risks: risks, lastUpdated: lastUpdated, pendingUpdates: pendingUpdates)
        if let data = try? APIClient.encoder.encode(snapshot) {
            try? data.write(to: cacheURL, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        }
    }

    private func loadCache() {
        guard !settings.demoMode,
              let data = try? Data(contentsOf: cacheURL),
              let snapshot = try? APIClient.decoder.decode(Snapshot.self, from: data) else { return }
        farmName = snapshot.farmName
        openAlerts = snapshot.openAlerts
        closedAlerts = snapshot.closedAlerts
        nodes = snapshot.nodes
        risks = snapshot.risks
        lastUpdated = snapshot.lastUpdated
        pendingUpdates = snapshot.pendingUpdates
    }
}
