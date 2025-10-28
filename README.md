Material for the lab sessions of the course "Learning and Optimization for Robot Control", at the University of Trento.

# Installation Instructions

There are different ways to install the software:
* native installation (recommended for maximum computational speed)
* docker (probably the easiest)
* nix-shell 

All the detailed instructions for the different kinds of installation can be found below.

The easiest way to install the dependencies is probably via Docker. Docker is in many ways similar to a virtual machine, but it is lighter and faster. On the downside, it is a bit more complex to use, so expect a slightly steeper learning curve in the beginning. Moreover, since the provided docker image is for an x86 architecture, if your laptop is a macbook with Apple chip (M1 or M2), the docker image is going to be quite slow.

As an alternative to docker, if you have a Linux or Mac OS, then you can decide to install all the software directly on your computer (native installation). Even if you are using Windows, thanks to WSL (Windows Subsystem for Linux) you can still install all the software natively on your machine. This has the advantage of having faster code execution.

## Installation using Docker
Docker should be extremely straightforward to set up for Linux users, and also for Windows users that are using the Windows Subsystem for Linux ([WSL](https://learn.microsoft.com/en-us/windows/wsl/install)). Mac users should go through a few extra steps, which are described below.

First, download and install Docker (Desktop) following the instructions on the [website](https://docs.docker.com/get-docker/).
Download the docker image using the following command:
```
docker pull andreadelprete/orc24:v1
```
NOTE: the password for the user "student" is: iamarobot

On Mac OS, you can run that docker image using this command:
```
docker run  --platform linux/amd64 -v /tmp/.X11-unix/:/tmp/.X11-unix/ --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --name ubuntu_bash --env="DISPLAY=host.docker.internal:0" --privileged -p 127.0.0.1:7000:7000 --shm-size 2g --rm -i -t --user=student --workdir=/home/student andreadelprete/orc24:v1 bash
````
On Windows/Linux, you can instead use this command:
```
docker run  -v /tmp/.X11-unix/:/tmp/.X11-unix/ --name ubuntu_bash --env="DISPLAY=$DISPLAY" --privileged -p 127.0.0.1:7000:7000 --shm-size 2g --rm -i -t --user=student --workdir=/home/student andreadelprete/orc24:v1 bash
```

### Docker for Mac OS
On Mac OS, to be able to run GUI applications from inside docker, you also need to install [XQuartz](https://www.xquartz.org). After installing it, activate the option "Allow connections from network clients" in xQuartz's settings. Finally, before running the docker image, run the following command:
```
xhost 127.0.0.1
```
You can find a more detailed guide on this at this link.

### File Sharing between Docker and Host
One important feature of Docker is file sharing. How do I see my source code inside my container? Suppose you run an Ubuntu container with 
```
docker run -it ubuntu bash
```
You’ll quickly find that (1) the container has its own filesystem, based on the filesystem in the Ubuntu image; (2) you can create, delete and modify files, but your changes are local to the container and are lost when the container is deleted; (3) the container doesn’t have access to any other files on your host computer.

So the natural next questions are, how can I see my other files? And how can my container write data that I can read later and maybe use in other containers? This is where bind mounts come in. It uses the -v flag to docker run to specify some files to share with the container. For example, if you run:
```
docker run -it -v /Users/adelprete/devel:/home/student/shared ubuntu bash
````
then the files at `/Users/adelprete/devel` will be available at `/home/student/shared` in the docker container, and you can read and write to them there.

### Additional Information
If we want to make the changes inside your docker image permanent, follow these steps:
https://github.com/mfocchi/lab-docker#committing-a-docker-image-locally-only-for-advanced-users

Some information regarding potential problems related to docker and how to solve them:
https://github.com/mfocchi/lab-docker/blob/master/install_docker.md#docker_issues
https://github.com/mfocchi/lab-docker


## Installation instructions for native MAC OS
While on Mac OS you can still use docker, it may be worth it to go through the initial effort of setting up a native installation to gain computational speed. For instance, on my M2 mac, using docker it took about 13 s to solve a simple optimal control problem for a robot manipulator. When switching to a native installation the time went down to 2 s. 

Use pip to install these packages:
```
pip install numpy matplotlib casadi==3.6.5 example-robot-data 'adam-robotics[casadi]' tsid meshcat quadprog
```
The latest version of casadi would be 3.6.6, but it gave me problems, so I suggest you use 3.6.5.

After that you should configure your environment variables by editing the file `.zshrc` that is located in your home folder. For instance you can open it with Visual Studio Code by using the following command:
```
code ~/.zshrc
```
Inside the file add the following line:
```
export PYTHONPATH=$PYTHONPATH:<path_to_folder_containing_orc>
```
where `<path_to_folder_containing_orc>` must be replaced with the path to the folder containing the ORC repository.


## Installation instructions for native Ubuntu machine

Follow these instructions if you have a computer with an Ubuntu Operating System, or you already have an Ubuntu virtual machine that you would rather use (e.g., to save space). Acceptable versions of Ubuntu are either 20.04, 22.04, or 24.04. 

Open a terminal and execute the following commands:
```
sudo apt install python3-numpy python3-scipy python3-matplotlib curl

sudo mkdir -p /etc/apt/keyrings

curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
     | sudo tee /etc/apt/keyrings/robotpkg.asc

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
     | sudo tee /etc/apt/sources.list.d/robotpkg.list

sudo apt update
```
On Ubuntu 22.04 install these packages:
```
sudo apt install robotpkg-py310-pinocchio robotpkg-py310-example-robot-data robotpkg-urdfdom robotpkg-py310-qt5-gepetto-viewer-corba robotpkg-py310-quadprog robotpkg-py310-tsid
```
For other versions of the Ubuntu OS you might need to use a different version of the python packages (e.g., on Ubuntu 24.04 you need to use py312). Configure the environment variables by adding the following lines to your file ~/.bashrc (you can use the software gedit to do so):
```
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export ROS_PACKAGE_PATH=/opt/openrobots/share
export PYTHONPATH=$PYTHONPATH:/opt/openrobots/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:<folder_containing_orc>
```
where <folder_containing_orc> is the folder containing the "orc" folder, which in turns contains all the python code of this class. Pay attention to the python version (e.g. python3.8)  in the name of the python folder (PYTHONPATH variable), which may be different from the one you have on your machine, depending on which OS version you have. 

For using the adam-robotics library and the meshcat viewer you need to install them with pip:
```
pip install adam-robotics[casadi] meshcat
```

## Installation instructions for nix-shell

Another alternative to install the software is to use nix-shell. At [this link](https://github.com/lucaSartore/nixos-config/tree/main/shells%2Fros-olrc) you can find a flake to run a nix-shell containing all the dependencies. 
The shell can be executed with the command
```
nix develop PATH_CONTAINING_FLAKE.NIX
```
This will create the environment from scratch (without virtualization overhead). Once you exit the shell, the local environment will remain unaffected. This has only been tested in NixOS, and not in other Linux distros or macOS.


# Running the software
Note for users using docker: If you installed the docker image, inside the home folder of the default user (`/home/student`) of the provided docker image you can find the "orc" folder, already configured to be used. However, we recommend you not to use that folder, but instead to put the orc folder in the shared folder, so that your changes to the code are not lost every time you quit the container. For doing this follow the following instructions.

You can clone the orc folder with the following command:
```
git clone https://github.com/andreadelprete/orc.git
```
To make sure you have the most updated version of the code you should run this command (inside the orc folder) before starting to modify the code:
```
git pull
```
Trying to execute this command after you have already modified the code will not work.

You can execute a python script directly from the terminal using the following command:
````
python3 script_name.py
````

If you want to keep interacting with the interpreter after the execution of the script use the following command:
````
python3 -i script_name.py
````

You can use the script "test_software.py" to check whether your environment is working fine:
````
python3 test_software.py
````
After running the script you should be able to see a robot manipulator moving in a simulation environment (meshcat). To be able to see the viewer you must open a window on your browser at the specified URL (127.0.0.1:7000). If you are using docker, you do not need to open the browser inside the docker image, but you can open it directly on your host machine.

## Using an IDE
Rather than running scripts from the terminal, it is more convenient to use a customized python editor. For instance, you can use the software Visual Studio Code, or spyder3. I suggest you run the IDE in your host machine rather than inside the docker image, and then use the docker image just for launching the script.

If you want instead to run the script inside the IDE you should run the IDE (e.g., spyder3) from the terminal by typing:
```
spyder3
```
Once spyder3 is open, you can use "File->Open" to open a python script, and then click on the "Run file" button (green "play" shape) to execute the script. The first time that you run a script in spyder3, you must set up the configuration options. In particular, you must choose the type of console between these 3 options:
* current console
* dedicated console
* external system terminal

Typically option 1 (which is the default choice) does not work, so you should use either option 2 or 3. I typically use option 2, but option 3 is fine as well. If you have already run a file on spyder3 and you want to change the console to use, you can do it via the menu "Run -> Configuration per file".
Side note: depending on your OS version, option 2 and/or option 3 also allow you to check the option "Interact with the Python console after execution", which is useful to explore the value of the script variables after the execution has ended.