# VehicleNet
This is a convolutional neural network implementation in TensorFlow for classifying vehicles of various types. The classes include 2 levels of identification - general vehicle type, and specific vehicle type. For example, a picture of a Boeing 747 would be classified under (air vehicle, airplane). 

## Data
The data was collected from [image-net.org](http://www.image-net.org), using their API for collecting image links. The code for downloading the sets of images is located in `datascraper.py`.
