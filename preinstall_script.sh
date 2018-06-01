#!/bin/bash
# May need to uncomment and update to find current packages
apt-get update
sudo apt-get install -y cuda-libraries-9-0
# Required for demo script! #
pip install scikit-video

# Add your desired packages for each workspace initialization
#          Add here!          #
conda install opencv==3.3.1
pip install pandas
pip install scikit-learn
pip install pillow
pip install tensorflow-gpu==1.8.0
pip install keras