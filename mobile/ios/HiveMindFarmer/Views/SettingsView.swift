import SwiftUI
import UIKit
import UserNotifications

struct SettingsView: View {
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var store: AlertStore
    @State private var notificationStatus: UNAuthorizationStatus = .notDetermined
    @State private var testResult: String?
    @State private var demoAlertID = DemoData.firstDemoAlertID

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Demo mode", isOn: $settings.demoMode)
                        .onChange(of: settings.demoMode) { _, _ in
                            store.resetForModeChange()
                            Task {
                                await store.refresh()
                                await AppDelegate.registerDeviceWithServer()
                            }
                        }
                } footer: {
                    Text(settings.demoMode
                         ? "Showing a sample farm. No server needed."
                         : "Showing your farm from the HiveMind alerts server.")
                }

                if !settings.demoMode {
                    Section("Farm server") {
                        TextField("Server address", text: $settings.serverURL)
                            .keyboardType(.URL)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                        SecureField("Farm code", text: $settings.farmToken)
                        Button("Connect") {
                            Task {
                                await store.refresh()
                                await AppDelegate.registerDeviceWithServer()
                                testResult = store.errorMessage ?? "Connected to \(store.farmName)"
                            }
                        }
                        if let testResult { Text(testResult).font(.footnote).foregroundStyle(.secondary) }
                    }
                }

                Section {
                    switch notificationStatus {
                    case .authorized, .provisional, .ephemeral:
                        Label("Notifications on", systemImage: "bell.badge.fill").foregroundStyle(.green)
                        if !settings.demoMode {
                            Label(settings.deviceRegistered ? "This phone is registered with the farm server"
                                                            : "Not yet registered with the farm server",
                                  systemImage: settings.deviceRegistered ? "checkmark.circle" : "clock")
                                .font(.footnote)
                        }
                    case .denied:
                        Label("Notifications are off", systemImage: "bell.slash.fill").foregroundStyle(.red)
                        Button("Open iPhone Settings") {
                            if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                        }
                    default:
                        Button {
                            Task {
                                _ = await NotificationAction.requestPermission()
                                await refreshNotificationStatus()
                            }
                        } label: { Label("Turn on disease alerts", systemImage: "bell.fill") }
                    }
                    if settings.demoMode {
                        Button {
                            let alert = DemoData.simulatedAlert(id: demoAlertID)
                            demoAlertID += 1
                            store.addDemoAlert(alert)
                            NotificationAction.scheduleAlertNotification(for: alert)
                        } label: { Label("Simulate a detection (in 5 s)", systemImage: "wand.and.stars") }
                    }
                } header: {
                    Text("Alerts")
                } footer: {
                    Text(settings.demoMode
                         ? "Tap Simulate, then lock the phone: the alert arrives like a real one. Tap it to open the plant's location."
                         : "You're notified when a field camera finds a disease, and when one is found near your farm.")
                }

                Section("Privacy") {
                    Text("Your cameras never send photos of your farm to other farms. Neighbours only hear which disease was found, roughly where (to about 1 km), and how far away.")
                        .font(.footnote)
                }

                Section {
                    LabeledContent("Version", value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0")
                } footer: {
                    Text("HiveMind: crop disease detection that learns across farms without sharing their data.")
                }
            }
            .navigationTitle("Settings")
            .task { await refreshNotificationStatus() }
        }
    }

    private func refreshNotificationStatus() async {
        notificationStatus = await UNUserNotificationCenter.current().notificationSettings().authorizationStatus
    }
}
