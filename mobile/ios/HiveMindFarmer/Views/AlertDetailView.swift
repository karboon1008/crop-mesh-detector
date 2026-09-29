import MapKit
import SwiftUI
import UIKit

/// Everything a farmer needs to act on one alert: where the plant is (map,
/// coordinates, distance, turn-by-turn), how sure the system is, the leaf
/// photo, what to do first, and buttons to report back.
struct AlertDetailView: View {
    let alertID: Int

    @EnvironmentObject private var store: AlertStore
    @EnvironmentObject private var location: LocationManager
    @State private var full: DiseaseAlert?
    @State private var photo: UIImage?
    @State private var pendingStatus: AlertStatus?
    @State private var note = ""
    @State private var copied = false

    private var alert: DiseaseAlert? { store.alert(id: alertID) ?? full }

    var body: some View {
        Group {
            if let alert {
                content(alert)
            } else {
                ContentUnavailableView("Alert not found", systemImage: "questionmark.circle",
                                       description: Text("Pull to refresh the alert list."))
            }
        }
        .navigationTitle("Alert")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            location.start()
            full = await store.detail(id: alertID)
            if let alert, alert.status == .new {
                await store.setStatus(.acknowledged, for: alertID)  // opening it counts as seen
            }
            if let alert, let data = await store.imageData(for: alert) { photo = UIImage(data: data) }
        }
        .sheet(item: $pendingStatus) { status in
            noteSheet(for: status)
        }
    }

    @ViewBuilder
    private func content(_ alert: DiseaseAlert) -> some View {
        let guide = TreatmentGuide.shared.guide(for: alert.disease)
        List {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Text(alert.title).font(.title2.bold())
                    HStack {
                        StatusBadge(status: alert.status)
                        if alert.status.isOpen { UrgencyBadge(urgency: guide.urgency) }
                    }
                    Text("\(alert.fieldName) · \(Int(alert.confidence * 100))% confidence")
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }

            locationSection(alert)

            Section("Evidence") {
                if let photo {
                    Image(uiImage: photo)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                } else {
                    Label(alert.imagePath == nil ? "No photo sent with this alert" : "Loading photo…",
                          systemImage: "photo")
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Seen", value: "\(alert.detectionCount) time\(alert.detectionCount == 1 ? "" : "s")")
                LabeledContent("First seen") { Text(alert.firstSeen, format: .dateTime.day().month().hour().minute()) }
                LabeledContent("Last seen") { Text(alert.lastSeen, format: .dateTime.day().month().hour().minute()) }
                LabeledContent("Camera", value: alert.nodeId)
                if let detections = (full ?? alert).detections, detections.count > 1 {
                    DisclosureGroup("Detection history (\(detections.count))") {
                        ForEach(detections, id: \.self) { d in
                            HStack {
                                Text(d.capturedAt, format: .dateTime.day().month().hour().minute())
                                Spacer()
                                Text("\(Int(min(d.cropConfidence, d.diseaseConfidence) * 100))%")
                                    .foregroundStyle(.secondary)
                            }
                            .font(.subheadline)
                        }
                    }
                }
            }

            guideSection(guide)

            Section {
                if alert.status.isOpen {
                    Button { pendingStatus = .treated } label: {
                        Label("I've treated it", systemImage: "checkmark.seal.fill")
                    }
                    Button(role: .destructive) { pendingStatus = .falseAlarm } label: {
                        Label("Not a disease (false alarm)", systemImage: "xmark.circle")
                    }
                } else {
                    Button { Task { await store.setStatus(.acknowledged, for: alert.id) } } label: {
                        Label("Reopen", systemImage: "arrow.uturn.backward")
                    }
                }
                if let note = alert.note, !note.isEmpty {
                    LabeledContent("Note", value: note)
                }
            } header: {
                Text("Report back")
            } footer: {
                Text("False alarms help retrain the farm cameras so they make fewer mistakes.")
            }
        }
    }

    @ViewBuilder
    private func locationSection(_ alert: DiseaseAlert) -> some View {
        Section("Where") {
            Map(initialPosition: .region(MKCoordinateRegion(center: alert.coordinate, latitudinalMeters: 250,
                                                             longitudinalMeters: 250))) {
                Marker(alert.fieldName, systemImage: "leaf.fill", coordinate: alert.coordinate)
                    .tint(alert.status.color)
                UserAnnotation()
            }
            .mapStyle(.hybrid)
            .frame(height: 220)
            .listRowInsets(EdgeInsets())

            VStack(alignment: .leading, spacing: 4) {
                Text(Coordinates.decimal(alert.coordinate)).font(.body.monospacedDigit())
                Text(Coordinates.dms(alert.coordinate)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            if let route = location.route(to: alert.coordinate) {
                Label("\(LocationManager.format(meters: route.meters)) \(route.direction) of you",
                      systemImage: "location.north.line.fill")
            }

            Button {
                NotificationAction.openInMaps(latitude: alert.latitude, longitude: alert.longitude,
                                              name: "\(alert.title) – \(alert.fieldName)")
            } label: {
                Label("Navigate to the plant", systemImage: "figure.walk")
            }
            Button {
                UIPasteboard.general.string = Coordinates.decimal(alert.coordinate)
                copied = true
            } label: {
                Label(copied ? "Copied" : "Copy coordinates", systemImage: copied ? "checkmark" : "doc.on.doc")
            }
            ShareLink(item: Coordinates.appleMapsURL(alert.coordinate, label: alert.title),
                      subject: Text(alert.title),
                      message: Text("\(alert.title) at \(alert.fieldName): \(Coordinates.decimal(alert.coordinate))")) {
                Label("Send location to a worker", systemImage: "square.and.arrow.up")
            }
        }
    }

    @ViewBuilder
    private func guideSection(_ guide: DiseaseGuide) -> some View {
        Section {
            Text(guide.spreads).font(.subheadline)
            DisclosureGroup("What to look for") {
                ForEach(guide.lookFor, id: \.self) { Label($0, systemImage: "eye").font(.subheadline) }
            }
            VStack(alignment: .leading, spacing: 8) {
                Text("Do now").font(.headline)
                ForEach(Array(guide.doNow.enumerated()), id: \.offset) { index, step in
                    HStack(alignment: .top, spacing: 8) {
                        Text("\(index + 1).").bold()
                        Text(step)
                    }
                    .font(.subheadline)
                }
            }
            .padding(.vertical, 4)
            if !guide.prevent.isEmpty {
                DisclosureGroup("Prevent it next time") {
                    ForEach(guide.prevent, id: \.self) { Label($0, systemImage: "shield").font(.subheadline) }
                }
            }
        } header: {
            Text("\(guide.name) · \(guide.cause)")
        } footer: {
            Text(TreatmentGuide.disclaimer)
        }
    }

    private func noteSheet(for status: AlertStatus) -> some View {
        NavigationStack {
            Form {
                Section {
                    TextField(status == .treated ? "What did you do? (optional)" : "What was it really? (optional)",
                              text: $note, axis: .vertical)
                        .lineLimit(3...6)
                }
            }
            .navigationTitle(status == .treated ? "Mark as treated" : "Mark as false alarm")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { pendingStatus = nil } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        let text = note
                        pendingStatus = nil
                        note = ""
                        Task { await store.setStatus(status, note: text, for: alertID) }
                    }
                }
            }
        }
        .presentationDetents([.medium])
    }
}
