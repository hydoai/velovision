# in Ubuntu,
# open 'Startup Applications'
# create a new script like this:
# gnome-terminal -- bash -c "/home/ryan/Gimondi/gi-edge/jetson_nx_startup.sh"
cd ~/Gimondi/gi-edge/computer_vision
python3 vision.py -f yolox_exps/nx-foxtrot.py --production_hardware --physical_switches
