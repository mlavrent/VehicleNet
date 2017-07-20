# VehicleNet
This is a convolutional neural network implementation in TensorFlow for classifying vehicles of various types. The current classes being used are: airplane, bus, tank, and yacht.

## Data
The data was collected from [image-net.org](http://www.image-net.org), using their API for collecting image links. The code for downloading the sets of images is located in `data_scraper.py`. Additional data was obtained from splitting various YouTube videos into seperate frames (e.g. compilation of airplanes landing/taking off). In order to do this, the video was downloaded from YouTube and then split using the command `ffmpeg -i input_video.mp4 -r 4/1 final_folder/%06d.png`. This would create four frames per second of video, which then were manually checked for quality.

#### Augmentation
In order to generate more data for training, the data was augmented in meaninful ways. Currently, for each image in the dataset, a left-right mirror image of that image is also added.

![Original](http://i.imgur.com/uiSxChy.jpg "Original") ![Flipped](http://i.imgur.com/1QoMe2h.png "Flipped")

#### Preparation
The neural network accepts fixed size images. In order to prepare an image, its largest dimension is scaled down to fit within the given image box (currently 150 wide by 100 tall) and the other dimension is scaled down to preserve the aspect ratio. Then, the image is placed on a black background the size of the given box. An example is below:

![Original image](http://i.imgur.com/oBKVSmk.jpg "Original") ![Modified image](http://i.imgur.com/qTp6CrB.png "Modified")
