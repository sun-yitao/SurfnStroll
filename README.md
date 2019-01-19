# Surf & Stroll

An app that helps people who use the phone while walking to do so more safely. 

It is a fully functional mobile browser that sends a alert to the user when there is a obstacle in front so the user does not walk into it or trip on it. There is also a gps functionality that allows the user to see where he or she is walking and route a path to get from point A to point B. 

### How we built the app
We started by collecting data using mobile phones in video format, which we extracted in jpgs at 5 frames per second. We then trained a ResNet18 model in keras which yielded 96% validation accuracy. We then converted the .h5 checkpoint to .coreml using coremltools. Our app runs the camera and image detection asynchronously in the background when the app is running (unfortunately iOS doesnt allow expensive computation and camera functions to carry out in the background when the app is not active).

### Instructions for installation

Build from source:
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

The Script for training the keras model and the best checkpoint is included in the folder Python_ML. Reason we used resnet 18 was due to a lack of data which lead to overfitting with bigger models.

### Authors

Yitao
Quinn
Noel
    
### License

This project is licensed under the MPL 2.0 License
