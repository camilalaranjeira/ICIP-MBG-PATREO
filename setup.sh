apt-get update
apt-get install -y libsm6 libxrender-dev
pip install pandas lxml seaborn 
rm -r /usr/include/opencv*
rm -r /usr/include/boost/compute/interop/opencv*
rm -r /usr/bin/opencv*
rm -r /usr/share/licenses/opencv3
rm /usr/lib/x86_64-linux-gnu/pkgconfig/opencv.pc
rm /usr/lib/x86_64-linux-gnu/libopencv*  
pip install opencv-python==4.2.0.34
