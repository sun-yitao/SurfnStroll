# Surf & Stroll
<img src="https://github.com/sun-yitao/SurfnStroll/blob/master/logo.png" alt="logo" width="200" height="200"/>
An app that helps people who use the phone while walking to do so more safely. 

It is a fully functional mobile browser that sends an alert to the user when there is an obstacle in front so the user does not walk into it or trip on it. There is also a gps functionality that allows the user to see where he or she is walking and route a path to get from point A to point B. 

Demo : [](https://youtu.be/f0exMuqcsCs)

### How we built the app
We started by collecting data using mobile phones in video format, which we extracted in jpgs at 5 frames per second. We then trained a ResNet18 model in keras which yielded 96% validation accuracy. After which, we converted the .h5 checkpoint to .coreml using coremltools. We used ResNet18 to train due to a lack of data which led to overfitting with bigger models. We then managed to add the video capturing and image recognition to a open source browser called brave (unfortunately iOS doesn't allow expensive computation and camera functions to carry out in the background when the app is not active). The reason we chose a web browser to build on is because it retains most of the smartphone's functionality including writing emails, browsing social media and using various other web apps. For the GPS routing function, it was written using the MapKit framework provided by Apple.

### Instructions for installation
For iOS only, install directly from [zachmane.github.io](zachmane.github.io), the app should begin to install in the background.

To build from source (requires mac with xcode and apple developer account to sideload the app):
1. Install the latest Xcode developer tools from Apple. (Xcode 10 and up required)
    
2. Install homebrew https://brew.sh
    
3. ```brew update``` <br />
  ``` brew install carthage```  <br />
5. ```brew install swiftlint``` <br />
6. ```git clone https://github.com/sun-yitao/collision_detector.git``` <br />
7. ```cd collision_detector``` <br /> 
   ```sh ./bootstrap.sh``` <br />
8. Open Client.xcodeproj in Xcode.
9. Build using Fennec scheme in Xcode.

Machine learning script we used and the keras .h5 model is in the Python_ML directory.

### Teammates

Sun Yitao <br />
Quinn Ng Wan Ying <br />
Noel Kwan Zhi Kai
    
### License

This project is licensed under the MPL 2.0 License
