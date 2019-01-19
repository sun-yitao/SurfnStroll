import UIKit
import MapKit

class MapScreen: UIViewController, CLLocationManagerDelegate {
    
    var window: UIWindow?
    var mapView: MKMapView?
    var locationManager: CLLocationManager?
    //The range (meter) of how much we want to see arround the user's location
    let distanceSpan: Double = 10000 //or 500
    var previousLocation: CLLocation?
    let geoCoder = CLGeocoder()
    var directionsArray: [MKDirections] = []
    
    let addressLabel = UILabel(frame: CGRect(x: 100, y: 686, width: 314, height: 60))
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.window = UIWindow(frame: UIScreen.main.bounds)
        self.view.backgroundColor = UIColor.white
        self.mapView = MKMapView(frame: CGRect(x: 0, y: 20, width: (self.window?.frame.width)!, height: 650))
        mapView?.delegate = self
        self.view.addSubview(self.mapView!)
        
        
        let backButton = UIButton(frame: CGRect(x: 0, y: 686, width: 100, height: 50))
        backButton.setTitleColor(.blue, for: .normal)
        backButton.setTitle("Back", for: .normal)
        backButton.addTarget(self, action: #selector(backButtonAction), for: .touchUpInside)
        self.view.addSubview(backButton)
        
        let goButton = UIButton(frame: CGRect(x: 334, y: 606, width: 60, height: 60))
        goButton.backgroundColor = .green
        goButton.setTitle("Go", for: .normal)
        goButton.addTarget(self, action: #selector(goButtonAction), for: .touchUpInside)
        self.view.addSubview(goButton)
        
        addressLabel.center = CGPoint(x: 257, y: 711)
        addressLabel.textAlignment = NSTextAlignment.center
        addressLabel.text = "Address"
        self.view.addSubview(addressLabel)

        self.locationManager = CLLocationManager()
        if let locationManager = self.locationManager {
            locationManager.delegate = self
            locationManager.desiredAccuracy = kCLLocationAccuracyBestForNavigation
            locationManager.requestWhenInUseAuthorization()
            locationManager.distanceFilter = 50
            locationManager.startUpdatingLocation()
        }
        goButton.layer.cornerRadius = goButton.frame.size.height/2
        startTackingUserLocation()
    }
    
    func startTackingUserLocation() {
        mapView?.showsUserLocation = true
        centerViewOnUserLocation()
        locationManager?.startUpdatingLocation()
        previousLocation = getCenterLocation(for: mapView!)
    }
    
    func centerViewOnUserLocation() {
        if let location = locationManager?.location?.coordinate {
            let region = MKCoordinateRegionMakeWithDistance(location, distanceSpan, distanceSpan)
            mapView!.setRegion(region, animated: true)
        }
    }
    
    func locationManager(manager: CLLocationManager, didUpdateToLocation newLocation: CLLocation, fromLocation oldLocation: CLLocation) {
        if let mapView = self.mapView {
            let region = MKCoordinateRegionMakeWithDistance(newLocation.coordinate, self.distanceSpan, self.distanceSpan)
            mapView.setRegion(region, animated: true)
            mapView.showsUserLocation = true
        }
    }
    func getCenterLocation(for mapView: MKMapView) -> CLLocation {
        let latitude = mapView.centerCoordinate.latitude
        let longitude = mapView.centerCoordinate.longitude
        
        return CLLocation(latitude: latitude, longitude: longitude)
    }
    
    func getDirections() {
        guard let location = locationManager?.location?.coordinate else {
            //TODO: Inform user we don't have their current location
            return
        }
        
        let request = createDirectionsRequest(from: location)
        let directions = MKDirections(request: request)
        resetMapView(withNew: directions)
        
        directions.calculate { [unowned self] (response, error) in
            //TODO: Handle error if needed
            guard let response = response else { return } //TODO: Show response not available in an alert
            
            for route in response.routes {
                self.mapView?.add(route.polyline)
                self.mapView?.setVisibleMapRect(route.polyline.boundingMapRect, animated: true)
            }
        }
    }
    
    func createDirectionsRequest(from coordinate: CLLocationCoordinate2D) -> MKDirections.Request {
        let destinationCoordinate       = getCenterLocation(for: mapView!).coordinate
        let startingLocation            = MKPlacemark(coordinate: coordinate)
        let destination                 = MKPlacemark(coordinate: destinationCoordinate)
        
        let request                     = MKDirections.Request()
        request.source                  = MKMapItem(placemark: startingLocation)
        request.destination             = MKMapItem(placemark: destination)
        request.transportType           = .walking
        request.requestsAlternateRoutes = true
        
        return request
        
    }
    
    
    func resetMapView(withNew directions: MKDirections) {
        mapView!.removeOverlays((mapView?.overlays)!)
        directionsArray.append(directions)
        let _ = directionsArray.map { $0.cancel() }
    }
    
    @objc func goButtonAction(sender: UIButton!) {
        getDirections()
    }
    
    @objc func backButtonAction(sender: UIButton!) {
        self.dismiss(animated: true, completion: nil)
    }
    
}

extension MapScreen: MKMapViewDelegate {
    
    func mapView(_ mapView: MKMapView, regionDidChangeAnimated animated: Bool) {
        let center = getCenterLocation(for: mapView)
        
        guard let previousLocation = self.previousLocation else { return }
        
        guard center.distance(from: previousLocation) > 50 else { return }
        self.previousLocation = center
        
        geoCoder.cancelGeocode()
        
        geoCoder.reverseGeocodeLocation(center) { [weak self] (placemarks, error) in
            guard let self = self else { return }
            
            if let _ = error {
                //TODO: Show alert informing the user
                return
            }
            
            guard let placemark = placemarks?.first else {
                //TODO: Show alert informing the user
                return
            }
            
            let streetNumber = placemark.subThoroughfare ?? ""
            let streetName = placemark.thoroughfare ?? ""
            
            DispatchQueue.main.async {
                self.addressLabel.text = "\(streetNumber) \(streetName)"
            }
        }
    }
    
    func mapView(_ mapView: MKMapView, rendererFor overlay: MKOverlay) -> MKOverlayRenderer {
        let renderer = MKPolylineRenderer(overlay: overlay as! MKPolyline)
        renderer.strokeColor = .blue
        
        return renderer
    }
}
