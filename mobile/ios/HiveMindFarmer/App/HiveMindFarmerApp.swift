import SwiftUI

@main
struct HiveMindFarmerApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @Environment(\.scenePhase) private var scenePhase

    @StateObject private var store = AlertStore.shared
    @StateObject private var settings = AppSettings.shared
    @StateObject private var router = AppRouter.shared
    @StateObject private var location = LocationManager.shared

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(store)
                .environmentObject(settings)
                .environmentObject(router)
                .environmentObject(location)
                .tint(Theme.brand)
                .task { await store.refresh() }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                Task { await store.refresh() }
            }
        }
    }
}
