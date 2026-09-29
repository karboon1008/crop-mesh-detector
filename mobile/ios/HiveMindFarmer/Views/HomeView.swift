import MapKit
import SwiftUI
import UIKit

/// The farm at a glance: a satellite map with every field camera's zone, the hive's view of
/// outbreaks on nearby farms, what the farmer's feedback teaches the next model round, and
/// the latest activity.
struct HomeView: View {
    @EnvironmentObject private var store: AlertStore
    @EnvironmentObject private var router: AppRouter
    @State private var position: MapCameraPosition = .automatic
    @State private var selectedNodeID: String?

    private var allAlerts: [DiseaseAlert] { store.openAlerts + store.closedAlerts }
    private var coverages: [DiseaseCoverage] { store.openAlerts.map { DiseaseCoverage(alert: $0) } }

    private var placedNodes: [(node: FieldNode, coordinate: CLLocationCoordinate2D)] {
        store.nodes.compactMap { node in node.coordinate.map { (node: node, coordinate: $0) } }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    if store.isOffline {
                        OfflineBanner(lastUpdated: store.lastUpdated)
                            .padding(.horizontal)
                    }
                    mapHero
                    cropChips
                    VStack(alignment: .leading, spacing: 16) {
                        HiveIntelligenceCard(risks: store.risks, alerts: allAlerts)
                        LearningCard(open: store.openAlerts, closed: store.closedAlerts)
                        lastActivities
                    }
                    .padding(.horizontal)
                }
                .padding(.bottom, 24)
            }
            .background(Theme.canvas)
            .refreshable { await store.refresh() }
            .navigationTitle(store.farmName.isEmpty ? "HiveMind" : store.farmName)
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(Theme.brand, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .onAppear(perform: fitFarm)
            .onChange(of: store.nodes.count) { _, _ in fitFarm() }
        }
    }

    // MARK: - Map

    private var mapHero: some View {
        Map(position: $position, interactionModes: []) {
            ForEach(store.risks) { risk in
                MapCircle(center: risk.coordinate, radius: 1000)
                    .foregroundStyle(.orange.opacity(0.18))
                    .stroke(.orange, lineWidth: 1)
            }
            // what each camera watches: a neutral outline, so it isn't mistaken for disease
            ForEach(placedNodes, id: \.node.id) { placed in
                MapCircle(center: placed.coordinate, radius: 60)
                    .foregroundStyle(.white.opacity(0.08))
                    .stroke(placed.node.online ? .white.opacity(0.7) : .gray, lineWidth: 1)
            }
            // where each open disease has spread, estimated from its detections
            ForEach(coverages) { coverage in
                let color = coverage.alert.status.color
                if let outline = coverage.outline {
                    MapPolygon(coordinates: outline)
                        .foregroundStyle(color.opacity(0.35))
                        .stroke(color, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                } else {
                    MapCircle(center: coverage.center, radius: coverage.radius)
                        .foregroundStyle(color.opacity(0.35))
                        .stroke(color, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                }
            }
            ForEach(placedNodes, id: \.node.id) { placed in
                Annotation(placed.node.name, coordinate: placed.coordinate) {
                    Button { withAnimation(.snappy) { selectedNodeID = placed.node.id } } label: {
                        Image(systemName: "sensor.fill")
                            .font(.caption)
                            .padding(6)
                            .background(placed.node.online ? Theme.brand : Color.gray, in: RoundedRectangle(cornerRadius: 6))
                            .foregroundStyle(.white)
                    }
                }
                .annotationTitles(.hidden)
            }
            ForEach(coverages) { coverage in
                Annotation(coverage.alert.title, coordinate: coverage.center, anchor: .bottom) {
                    Button { router.openAlert(coverage.alert.id) } label: {
                        VStack(spacing: 0) {
                            Text(DisplayName.pretty(coverage.alert.disease)).font(.caption2.weight(.bold))
                            Text("≈ " + coverage.areaText).font(.caption2.monospacedDigit())
                        }
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(coverage.alert.status.color, in: RoundedRectangle(cornerRadius: 6))
                        .foregroundStyle(.white)
                    }
                }
                .annotationTitles(.hidden)
            }
        }
        .mapStyle(.hybrid(elevation: .flat))
        .frame(height: 340)
        .overlay(alignment: .topTrailing) { hiveStats.padding(10) }
        .overlay(alignment: .topLeading) {
            if let node = store.nodes.first(where: { $0.id == selectedNodeID }) {
                FieldCard(node: node, openAlerts: store.openAlerts.filter { $0.nodeId == node.nodeId }.count) {
                    withAnimation(.snappy) { selectedNodeID = nil }
                }
                .padding(10)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .overlay(alignment: .bottomTrailing) {
            Button { router.tab = .alerts } label: {
                Label("\(store.openAlerts.count)", systemImage: "exclamationmark.triangle.fill")
                    .font(.subheadline.weight(.bold).monospacedDigit())
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(.white, in: Capsule())
                    .foregroundStyle(store.newAlertCount > 0 ? .red : .orange)
            }
            .padding(10)
            .opacity(store.openAlerts.isEmpty ? 0 : 1)
        }
        .overlay(alignment: .bottomLeading) {
            Button { router.tab = .map } label: {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
                    .font(.subheadline.weight(.semibold))
                    .padding(8)
                    .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
                    .foregroundStyle(.white)
            }
            .padding(10)
        }
    }

    private var hiveStats: some View {
        let online = store.nodes.filter(\.online).count
        let lowestBattery = store.nodes.compactMap(\.batteryPct).min()
        return VStack(alignment: .trailing, spacing: 6) {
            StatChip(symbol: "sensor.fill", value: "\(online)/\(store.nodes.count)",
                     tint: online < store.nodes.count ? .orange : .green)
            if let lowestBattery {
                StatChip(symbol: "battery.25percent", value: "\(Int(lowestBattery))%",
                         tint: lowestBattery < 20 ? .red : .white)
            }
            StatChip(symbol: "leaf.fill", value: "\(store.openAlerts.count)",
                     tint: store.openAlerts.isEmpty ? .green : .red)
            StatChip(symbol: "dot.radiowaves.left.and.right", value: "\(store.risks.count)",
                     tint: store.risks.isEmpty ? .white : .orange)
        }
    }

    /// Frames the farm's own cameras and alerts; nearby farms' outbreaks can sit off screen.
    private func fitFarm() {
        let coordinates = placedNodes.map(\.coordinate) + coverages.flatMap { $0.outline ?? [$0.center] }
        guard let minLat = coordinates.map(\.latitude).min(), let maxLat = coordinates.map(\.latitude).max(),
              let minLon = coordinates.map(\.longitude).min(), let maxLon = coordinates.map(\.longitude).max()
        else { return }
        let center = CLLocationCoordinate2D(latitude: (minLat + maxLat) / 2, longitude: (minLon + maxLon) / 2)
        // extra width on the right keeps the stat chips clear of the pins
        let span = MKCoordinateSpan(latitudeDelta: max((maxLat - minLat) * 1.8, 0.004),
                                    longitudeDelta: max((maxLon - minLon) * 2.2, 0.004))
        position = .region(MKCoordinateRegion(center: center, span: span))
    }

    // MARK: - Crops

    private var crops: [(name: String, open: Int)] {
        var counts: [String: Int] = [:]
        for alert in allAlerts { counts[alert.crop, default: 0] += alert.status.isOpen ? 1 : 0 }
        return counts.map { (name: $0.key, open: $0.value) }
            .sorted { $0.open != $1.open ? $0.open > $1.open : $0.name < $1.name }
    }

    @ViewBuilder
    private var cropChips: some View {
        if !crops.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(crops, id: \.name) { crop in
                        Button { router.tab = .alerts } label: { CropChip(name: crop.name, open: crop.open) }
                            .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal)
            }
            .padding(.top, -8)
        }
    }

    // MARK: - Last activities

    private var recent: [DiseaseAlert] {
        Array(allAlerts.sorted { $0.lastSeen > $1.lastSeen }.prefix(5))
    }

    private var lastActivities: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Last activities").font(.title3.weight(.semibold))
            if recent.isEmpty {
                Text("No detections yet.").foregroundStyle(.secondary)
            }
            ForEach(recent) { alert in
                Button { router.openAlert(alert.id) } label: { ActivityRow(alert: alert) }
                    .buttonStyle(.plain)
                if alert.id != recent.last?.id { Divider() }
            }
        }
        .cardStyle()
    }
}

// MARK: - Disease coverage

/// The area a disease covers, estimated from where it was detected: a circle that grows with
/// how often it was seen (and reaches every detection), or, when the detections spread wider
/// than that, the outline around them plus a margin.
struct DiseaseCoverage: Identifiable {
    let alert: DiseaseAlert
    let center: CLLocationCoordinate2D
    let outline: [CLLocationCoordinate2D]?
    let radius: CLLocationDistance
    let areaM2: Double

    var id: Int { alert.id }

    /// Beyond the outermost detection: a leaf that was seen sits in a patch that wasn't.
    static let margin: Double = 12

    init(alert: DiseaseAlert) {
        self.alert = alert
        let points = (alert.detections ?? []).map { CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude) }
            + [alert.coordinate]
        let center = CLLocationCoordinate2D(latitude: points.map(\.latitude).reduce(0, +) / Double(points.count),
                                            longitude: points.map(\.longitude).reduce(0, +) / Double(points.count))
        self.center = center

        // metres east (x) and north (y) of the centre; fine at field scale
        let metresPerLat = 111_320.0
        let metresPerLon = 111_320.0 * cos(center.latitude * .pi / 180)
        let local = points.map { CGPoint(x: ($0.longitude - center.longitude) * metresPerLon,
                                         y: ($0.latitude - center.latitude) * metresPerLat) }
        let hull = Self.convexHull(local)
        let spread = Double(local.map { hypot($0.x, $0.y) }.max() ?? 0)
        let circleRadius = max(spread + Self.margin, min(15 + 4 * Double(alert.detectionCount), 60))
        let circleArea = Double.pi * circleRadius * circleRadius

        var polygon: (outline: [CLLocationCoordinate2D], radius: Double, area: Double)?
        if hull.count >= 3 {
            let grown = hull.map { p -> CGPoint in
                let d = hypot(p.x, p.y)
                let scale = d == 0 ? 1 : (d + CGFloat(Self.margin)) / d
                return CGPoint(x: p.x * scale, y: p.y * scale)
            }
            let twiceArea = zip(grown, Array(grown.dropFirst()) + [grown[0]])
                .reduce(CGFloat(0)) { $0 + ($1.0.x * $1.1.y - $1.1.x * $1.0.y) }
            polygon = (
                grown.map { CLLocationCoordinate2D(latitude: center.latitude + Double($0.y) / metresPerLat,
                                                   longitude: center.longitude + Double($0.x) / metresPerLon) },
                Double(grown.map { hypot($0.x, $0.y) }.max() ?? 0),
                Double(abs(twiceArea)) / 2
            )
        }

        // the outline only when it says more than the detection count does, so a disease
        // seen more often never looks smaller
        if let polygon, polygon.area > circleArea {
            outline = polygon.outline
            radius = polygon.radius
            areaM2 = polygon.area
        } else {
            outline = nil
            radius = circleRadius
            areaM2 = circleArea
        }
    }

    var areaText: String {
        areaM2 < 10_000 ? "\(Int((areaM2 / 10).rounded()) * 10) m²" : String(format: "%.2f ha", areaM2 / 10_000)
    }

    /// Andrew's monotone chain, counter-clockwise, without repeated points.
    private static func convexHull(_ points: [CGPoint]) -> [CGPoint] {
        var seen = Set<SIMD2<Double>>()
        var unique: [CGPoint] = []
        for p in points where seen.insert(SIMD2<Double>(Double(p.x), Double(p.y))).inserted {
            unique.append(p)
        }
        let sorted = unique.sorted { (a: CGPoint, b: CGPoint) -> Bool in
            a.x != b.x ? a.x < b.x : a.y < b.y
        }
        guard sorted.count >= 3 else { return sorted }
        func cross(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }
        var lower: [CGPoint] = []
        for p in sorted {
            while lower.count >= 2 && cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 { lower.removeLast() }
            lower.append(p)
        }
        var upper: [CGPoint] = []
        for p in sorted.reversed() {
            while upper.count >= 2 && cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 { upper.removeLast() }
            upper.append(p)
        }
        return Array(lower.dropLast()) + Array(upper.dropLast())
    }
}

// MARK: - Map overlays

private struct StatChip: View {
    let symbol: String
    let value: String
    var tint: Color = .white

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: symbol).foregroundStyle(tint)
            Text(value).foregroundStyle(.white)
        }
        .font(.subheadline.weight(.semibold).monospacedDigit())
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct FieldCard: View {
    let node: FieldNode
    let openAlerts: Int
    let close: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(node.name).font(.headline).lineLimit(1)
                Spacer(minLength: 8)
                Button(action: close) { Image(systemName: "chevron.up") }
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 6) {
                Text(node.online ? "Online" : "Offline")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background((node.online ? Theme.brand : Color.gray), in: Capsule())
                    .foregroundStyle(.white)
                if openAlerts > 0 {
                    Text("\(openAlerts) open")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Color.red.opacity(0.15), in: Capsule())
                        .foregroundStyle(.red)
                }
            }
            HStack(spacing: 12) {
                if let battery = node.batteryPct {
                    Label("\(Int(battery))%", systemImage: "battery.50percent")
                        .foregroundStyle(battery < 20 ? .red : .secondary)
                }
                if let lastSeen = node.lastSeen {
                    Label { Text(lastSeen, style: .relative) } icon: { Image(systemName: "clock") }
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(width: 230, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .shadow(color: .black.opacity(0.15), radius: 8, y: 3)
    }
}

private struct CropChip: View {
    let name: String
    let open: Int

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "leaf.fill")
            Text(DisplayName.pretty(name)).fontWeight(.semibold)
            if open > 0 {
                Text("\(open)")
                    .font(.caption.weight(.bold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 1)
                    .background(.red, in: Capsule())
                    .foregroundStyle(.white)
            } else {
                Image(systemName: "checkmark").font(.caption.weight(.bold))
            }
        }
        .font(.subheadline)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Theme.brand.opacity(0.14), in: Capsule())
        .foregroundStyle(Theme.brandDark)
    }
}

// MARK: - Hive intelligence

/// Outbreaks the hive reports from nearby farms (disease, crop, ~1 km location only) and a
/// week of outbreak activity, on this farm and around it.
private struct HiveIntelligenceCard: View {
    let risks: [NearbyRisk]
    let alerts: [DiseaseAlert]

    private enum Level {
        case high, medium, low

        var title: String { self == .high ? "High" : self == .medium ? "Medium" : "Low" }
        var color: Color { self == .high ? .red : self == .medium ? .orange : .green }
    }

    /// High: a crop this farm grows, within 3 km. Medium: one of the two. Low: neither.
    private func level(_ risk: NearbyRisk) -> Level {
        let grown = alerts.contains { $0.crop == risk.crop }
        let close = risk.distanceKm <= 3
        if grown && close { return .high }
        return grown || close ? .medium : .low
    }

    private struct Day: Identifiable {
        let id: Int
        let label: String
        let local: Int
        let nearby: Int
    }

    private var week: [Day] {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: .now)
        let real = alerts.filter { $0.status != .falseAlarm }
        return (0..<7).reversed().map { offset in
            let start = calendar.date(byAdding: .day, value: -offset, to: today)!
            let end = calendar.date(byAdding: .day, value: 1, to: start)!
            return Day(
                id: offset,
                label: offset == 0 ? "Today" : start.formatted(.dateTime.weekday(.abbreviated)),
                local: real.filter { $0.firstSeen < end && $0.lastSeen >= start }.count,
                nearby: risks.filter { $0.lastSeen >= start && $0.lastSeen < end }.count
            )
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "dot.radiowaves.left.and.right").foregroundStyle(Theme.brand)
                Text("Hive intelligence").font(.title3.weight(.semibold))
            }

            if risks.isEmpty {
                Label("No outbreaks reported by nearby farms.", systemImage: "checkmark.shield.fill")
                    .font(.subheadline)
                    .foregroundStyle(Theme.brand)
            }
            ForEach(risks.prefix(2)) { risk in
                let riskLevel = level(risk)
                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Pathogen").font(.caption).foregroundStyle(.secondary)
                            Text(DisplayName.pretty(risk.disease)).font(.headline)
                        }
                        Spacer()
                        VStack(alignment: .trailing, spacing: 2) {
                            Text("Risk to your farm").font(.caption).foregroundStyle(.secondary)
                            Text(riskLevel.title).font(.headline).foregroundStyle(riskLevel.color)
                        }
                    }
                    Text("On \(DisplayName.pretty(risk.crop).lowercased()) · \(risk.distanceKm, specifier: "%g") km away · ")
                        .font(.caption).foregroundStyle(.secondary)
                    + Text(risk.lastSeen, style: .relative).font(.caption).foregroundStyle(.secondary)
                    + Text(" ago").font(.caption).foregroundStyle(.secondary)
                }
            }

            Divider()

            Grid(horizontalSpacing: 6, verticalSpacing: 6) {
                GridRow {
                    ForEach(week) { day in
                        Text(day.label).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                row(title: "Your cameras") { $0.local }
                row(title: "Nearby farms") { $0.nearby }
            }
            .frame(maxWidth: .infinity)
        }
        .cardStyle()
    }

    @ViewBuilder
    private func row(title: String, count: @escaping (Day) -> Int) -> some View {
        GridRow {
            Text(title).font(.caption).foregroundStyle(.secondary).gridCellColumns(7)
        }
        GridRow {
            ForEach(week) { day in
                RoundedRectangle(cornerRadius: 4)
                    .fill(color(for: count(day)))
                    .frame(width: 26, height: 26)
            }
        }
    }

    private func color(for outbreaks: Int) -> Color {
        switch outbreaks {
        case 0: .green
        case 1: .yellow
        case 2: .orange
        default: .red
        }
    }
}

// MARK: - Learning together

/// The farmer's reports are HiveMind's labels: confirmed and false-alarm alerts feed the next
/// continual-learning round, while photos stay on the farm.
private struct LearningCard: View {
    let open: [DiseaseAlert]
    let closed: [DiseaseAlert]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "brain.head.profile").foregroundStyle(Theme.brand)
                Text("Learning together").font(.title3.weight(.semibold))
            }
            HStack {
                stat("\(closed.filter { $0.status == .treated }.count)", "Confirmed", .green)
                stat("\(closed.filter { $0.status == .falseAlarm }.count)", "False alarms", .gray)
                stat("\(open.count)", "To review", .orange)
            }
            Label("Photos stay on your farm. The hive shares what the models learned, never images.",
                  systemImage: "lock.shield.fill")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Label("Your reports train the next model round.", systemImage: "arrow.triangle.2.circlepath")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .cardStyle()
    }

    private func stat(_ value: String, _ title: String, _ color: Color) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.title2.weight(.bold).monospacedDigit()).foregroundStyle(color)
            Text(title).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Activity rows

private struct ActivityRow: View {
    let alert: DiseaseAlert

    var body: some View {
        HStack(spacing: 12) {
            AlertThumbnail(alert: alert)
            VStack(alignment: .leading, spacing: 3) {
                Text(DisplayName.pretty(alert.disease))
                    .font(.headline)
                    .foregroundStyle(Theme.brand)
                    .lineLimit(1)
                Label(DisplayName.pretty(alert.crop) + " · " + alert.fieldName, systemImage: "leaf")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 8)
            VStack(alignment: .trailing, spacing: 6) {
                Text(alert.lastSeen, format: .dateTime.day().month(.twoDigits))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                StatusBadge(status: alert.status)
            }
        }
        .contentShape(Rectangle())
    }
}

private struct AlertThumbnail: View {
    let alert: DiseaseAlert
    @EnvironmentObject private var store: AlertStore
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                LinearGradient(colors: [Theme.brand, Theme.brandDark], startPoint: .topLeading, endPoint: .bottomTrailing)
                Image(systemName: "leaf.fill").font(.title3).foregroundStyle(.white.opacity(0.9))
            }
        }
        .frame(width: 52, height: 52)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .task(id: alert.id) {
            if let data = await store.imageData(for: alert) { image = UIImage(data: data) }
        }
    }
}
