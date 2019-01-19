/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

import Foundation
import Shared
import XCGLogger

private let log = Logger.browserLogger

struct HomePageConstants {
    static let HomePageURLPrefKey = "HomePageURLPref"
}

class HomePageHelper {

    let prefs: Prefs

    var currentURL: URL? {
        get {
            return HomePageAccessors.getHomePage(prefs)
        }
        set {
            if let url = newValue, url.isWebPage(includeDataURIs: false) && !url.isLocal {
                prefs.setString(url.absoluteString, forKey: HomePageConstants.HomePageURLPrefKey)
            } else {
                prefs.removeObjectForKey(HomePageConstants.HomePageURLPrefKey)
            }
        }
    }

    var isHomePageAvailable: Bool { return currentURL != nil }

    init(prefs: Prefs) {
        self.prefs = prefs
    }

    func openHomePage(_ tab: Tab) {
        guard let url = currentURL else {
            // this should probably never happen.
            log.error("User requested a homepage that wasn't a valid URL")
            return
        }
        tab.loadRequest(URLRequest(url: url))
    }

    func openHomePage(inTab tab: Tab, presentAlertOn viewController: UIViewController?) {
        if isHomePageAvailable {
            openHomePage(tab)
        } else {
            setHomePage(toTab: tab, presentAlertOn: viewController)
        }
    }

    func setHomePage(toTab tab: Tab, presentAlertOn viewController: UIViewController?) {
        let alertController = UIAlertController(
            title: Strings.SetHomePageDialogTitle,
            message: Strings.SetHomePageDialogMessage,
            preferredStyle: .alert)
        alertController.addAction(UIAlertAction(title: Strings.SetHomePageDialogNo, style: .cancel, handler: nil))
        alertController.addAction(UIAlertAction(title: Strings.SetHomePageDialogYes, style: .default) { _ in
            self.currentURL = tab.url as URL?
        })
        viewController?.present(alertController, animated: true, completion: nil)
    }
}
