import SwiftUI
import UIKit

/// HiveMind Farmer colours and the card style shared by the Home screen.
enum Theme {
    /// Field green, used for the navigation bar, tint and brand accents.
    static let brand = Color(red: 0.34, green: 0.52, blue: 0.33)
    static let brandDark = Color(red: 0.22, green: 0.38, blue: 0.22)
    static let canvas = Color(.systemGroupedBackground)
    static let card = Color(.secondarySystemGroupedBackground)
}

struct CardStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.card, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .shadow(color: .black.opacity(0.06), radius: 8, y: 2)
    }
}

extension View {
    func cardStyle() -> some View { modifier(CardStyle()) }
}
