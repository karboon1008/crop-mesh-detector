import Foundation
import Security

/// Server address and demo mode live in UserDefaults; the farm code (the
/// app token that gives access to this farm's alerts) lives in the Keychain.
@MainActor
final class AppSettings: ObservableObject {
    static let shared = AppSettings()

    @Published var serverURL: String {
        didSet { UserDefaults.standard.set(serverURL, forKey: Keys.serverURL) }
    }
    @Published var farmToken: String {
        didSet { Keychain.set(farmToken, for: Keys.farmToken) }
    }
    /// Demo mode shows built-in sample farm data and needs no server, so the
    /// app can be shown anywhere. On by default until a server is set up.
    @Published var demoMode: Bool {
        didSet { UserDefaults.standard.set(demoMode, forKey: Keys.demoMode) }
    }
    /// The APNs device token, hex-encoded, once iOS hands it over.
    @Published var deviceToken: String? {
        didSet { UserDefaults.standard.set(deviceToken, forKey: Keys.deviceToken) }
    }
    @Published var deviceRegistered = false

    private enum Keys {
        static let serverURL = "serverURL"
        static let farmToken = "farmToken"
        static let demoMode = "demoMode"
        static let deviceToken = "deviceToken"
    }

    private init() {
        let defaults = UserDefaults.standard
        serverURL = defaults.string(forKey: Keys.serverURL) ?? "http://192.168.1.10:8080"
        farmToken = Keychain.get(Keys.farmToken) ?? ""
        demoMode = defaults.object(forKey: Keys.demoMode) as? Bool ?? true
        deviceToken = defaults.string(forKey: Keys.deviceToken)
    }

    /// nil in demo mode or while the server address / farm code is missing.
    var client: APIClient? {
        guard !demoMode, !farmToken.isEmpty,
              let url = URL(string: serverURL.trimmingCharacters(in: .whitespaces)), url.scheme != nil
        else { return nil }
        return APIClient(baseURL: url, token: farmToken)
    }
}

enum Keychain {
    private static let service = "org.hivemind.farmer"

    static func set(_ value: String, for key: String) {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                    kSecAttrService as String: service,
                                    kSecAttrAccount as String: key]
        SecItemDelete(query as CFDictionary)
        guard !value.isEmpty else { return }
        var add = query
        add[kSecValueData as String] = Data(value.utf8)
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(add as CFDictionary, nil)
    }

    static func get(_ key: String) -> String? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                    kSecAttrService as String: service,
                                    kSecAttrAccount as String: key,
                                    kSecReturnData as String: true,
                                    kSecMatchLimit as String: kSecMatchLimitOne]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess, let data = item as? Data else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }
}
