# orc
Material for lab sessions of "Learning and Optimization for Robot Control" @ UniTN

## Installation instructions for native Ubuntu machine

Follow these instructions if you have a computer with an Ubuntu Operating System, or you already have an Ubuntu virtual machine that you would rather use (e.g., to save space). Acceptable versions of Ubuntu are either 20.04 or 22.04. 

Open a terminal and execute the following commands:
```
sudo apt install terminator python3-numpy python3-scipy python3-matplotlib spyder3 curl

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"

sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/wip/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"

curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
sudo apt-get update
```
On Ubuntu 20.04 install these packages:
```
sudo apt install robotpkg-py38-pinocchio robotpkg-py38-example-robot-data robotpkg-urdfdom robotpkg-py38-qt5-gepetto-viewer-corba robotpkg-py38-quadprog robotpkg-py38-tsid
```
For other versions of the Ubuntu OS you might need to use a different version of the python packages (e.g., on Ubuntu 22.04 you need to use py310 instead of py38). Configure the environment variables by adding the following lines to your file ~/.bashrc (you can use the software gedit to do so):
```
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export ROS_PACKAGE_PATH=/opt/openrobots/share
export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:<folder_containing_orc>
```
where <folder_containing_orc> is the folder containing the "orc" folder, which in turns contains all the python code of this class. Pay attention to the python version (e.g. python3.8)  in the name of the python folder (PYTHONPATH variable), which may be different from the one you have on your machine, depending on which OS version you have. 

For using the meshcat viewer you need to install it with:
```
pip install meshcat
```
