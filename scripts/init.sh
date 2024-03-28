# development initializer script for headless docker container

pip install -r requirements.txt
git config --global --add safe.directory /app
apt-get update
apt-get install libglfw3
apt-get install libgl1-mesa-glx
apt-get install libosmesa6
apt install xvfb