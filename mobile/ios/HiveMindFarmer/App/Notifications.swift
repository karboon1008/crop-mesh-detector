import MapKit
import UIKit
import UserNotifications

/// Which tab is showing and which alert is open; a notification tap sets it.
@MainActor
final class AppRouter: ObservableObject {
    static let shared = AppRouter()

    enum Tab: Hashable { case alerts, map, fields, settings }

    @Published var tab: Tab = .alerts
    @Published var alertPath: [Int] = []

    func openAlert(_ id: Int) {
        tab = .alerts
        alertPath = [id]
    }
}

enum NotificationAction {
    static let diseaseCategory = "DISEASE_ALERT"  // set by services/alerts in aps.category
    static let nearbyCategory = "NEARBY_RISK"
    static let navigate = "NAVIGATE"
    static let acknowledge = "ACKNOWLEDGE"
    static let viewMap = "VIEW_MAP"

    static func registerCategories() {
        let navigate = UNNotificationAction(identifier: navigate, title: "Navigate to plant",
                                            options: [.foreground], icon: .init(systemImageName: "location.fill"))
        let acknowledge = UNNotificationAction(identifier: acknowledge, title: "Mark as seen",
                                               options: [], icon: .init(systemImageName: "eye"))
        let viewMap = UNNotificationAction(identifier: viewMap, title: "Show on map",
                                           options: [.foreground], icon: .init(systemImageName: "map"))
        UNUserNotificationCenter.current().setNotificationCategories([
            UNNotificationCategory(identifier: diseaseCategory, actions: [navigate, acknowledge], intentIdentifiers: []),
            UNNotificationCategory(identifier: nearbyCategory, actions: [viewMap], intentIdentifiers: []),
        ])
    }

    /// Asks once; registers for remote notifications whenever allowed.
    static func requestPermission() async -> Bool {
        let center = UNUserNotificationCenter.current()
        let granted = (try? await center.requestAuthorization(options: [.alert, .sound, .badge])) ?? false
        if granted {
            await MainActor.run { UIApplication.shared.registerForRemoteNotifications() }
        }
        return granted
    }

    static func openInMaps(latitude: Double, longitude: Double, name: String) {
        let placemark = MKPlacemark(coordinate: .init(latitude: latitude, longitude: longitude))
        let item = MKMapItem(placemark: placemark)
        item.name = name
        item.openInMaps(launchOptions: [MKLaunchOptionsDirectionsModeKey: MKLaunchOptionsDirectionsModeWalking])
    }

    /// Demo mode: the same notification the server would push, delivered locally
    /// after `delay` seconds (lock the phone to see it on the lock screen).
    static func scheduleDemoNotification(for alert: DiseaseAlert, delay: TimeInterval = 5) {
        let content = UNMutableNotificationContent()
        content.title = "Disease detected: \(DisplayName.pretty(alert.crop)) \(DisplayName.pretty(alert.disease).lowercased())"
        content.body = "\(alert.fieldName) (\(Coordinates.decimal(alert.coordinate))), \(Int(alert.confidence * 100))% confidence. Tap to see where."
        content.sound = .default
        content.categoryIdentifier = diseaseCategory
        content.threadIdentifier = "alert-\(alert.id)"
        content.userInfo = ["kind": "disease_alert", "alert_id": alert.id,
                            "latitude": alert.latitude, "longitude": alert.longitude]
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: delay, repeats: false)
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: "demo-\(alert.id)", content: content, trigger: trigger))
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        UNUserNotificationCenter.current().delegate = self
        NotificationAction.registerCategories()
        Task {
            let settings = await UNUserNotificationCenter.current().notificationSettings()
            if settings.authorizationStatus == .authorized {
                await MainActor.run { application.registerForRemoteNotifications() }
            }
        }
        return true
    }

    func application(_ application: UIApplication, didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        let hex = deviceToken.map { String(format: "%02x", $0) }.joined()
        Task { @MainActor in
            AppSettings.shared.deviceToken = hex
            await Self.registerDeviceWithServer()
        }
    }

    func application(_ application: UIApplication, didFailToRegisterForRemoteNotificationsWithError error: Error) {
        // Simulator without push support, or no aps-environment entitlement: demo notifications still work.
        print("Push registration failed: \(error.localizedDescription)")
    }

    /// Sends the APNs token to the farm's alerts service so it can push to this phone.
    @MainActor
    static func registerDeviceWithServer() async {
        let settings = AppSettings.shared
        guard let token = settings.deviceToken, let client = settings.client else { return }
        do {
            try await client.registerDevice(token: token)
            settings.deviceRegistered = true
        } catch {
            settings.deviceRegistered = false
        }
    }

    // Show the banner even when the app is open, and pull the new alert in.
    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification) async
        -> UNNotificationPresentationOptions {
        await AlertStore.shared.refresh()
        return [.banner, .list, .sound, .badge]
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse) async {
        let info = response.notification.request.content.userInfo
        let kind = info["kind"] as? String
        let alertID = (info["alert_id"] as? Int) ?? (info["alert_id"] as? NSNumber)?.intValue
        let latitude = (info["latitude"] as? NSNumber)?.doubleValue
        let longitude = (info["longitude"] as? NSNumber)?.doubleValue

        await MainActor.run {
            switch response.actionIdentifier {
            case NotificationAction.navigate:
                if let latitude, let longitude {
                    NotificationAction.openInMaps(latitude: latitude, longitude: longitude,
                                                  name: response.notification.request.content.title)
                }
            case NotificationAction.viewMap:
                AppRouter.shared.tab = .map
            default:
                if kind == "nearby_risk" {
                    AppRouter.shared.tab = .map
                } else if let alertID {
                    AppRouter.shared.openAlert(alertID)
                }
            }
        }
        await AlertStore.shared.refresh()
        if response.actionIdentifier == NotificationAction.acknowledge, let alertID {
            await AlertStore.shared.setStatus(.acknowledged, for: alertID)
        }
    }
}
