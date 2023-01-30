# Object blurring for HARAI class

Blur objects such as mobile phones in a video stream. By default we blur mobile phone, laptop, and tv.

## Installation

Execute the script below to install DepthAI on Linux systems:
```
sudo wget -qO- https://docs.luxonis.com/install_depthai.sh | bash
```

You may need to do ```sudo apt install python3.8-venv``` and run install_depthai.sh again, if you get an error 

Then install additional dependencies
```
pip3 install depthai blobconverter opencv-python
```

## Run

Now you can run the script
```
python3 blur_objects.py
```

Make sure your OAK-D camera is connected.
