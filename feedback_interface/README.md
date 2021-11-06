# Feedback Interface: Interacting with speakers and displays

## Setting Default Speaker
Stack overflow:
```
EDIT (05/03/2020):
It seems that @phanky5 figured out a simpler solution. Please check it before you try this one.

Here is a well explained tutorial to set a default audio input/output.
First: List the audio output devices using

pactl list short sources  
Example of the output:

pactl list short sources
0   alsa_output.pci-0000_02_00.1.hdmi-stereo.monitor    module-alsa-card.c  s16le 2ch 44100Hz   SUSPENDED
1   alsa_input.usb-AVerMedia_Technologies__Inc._Live_Gamer_Portable_2_5202050100060-03.analog-stereo    module-alsa-card.c  
2   alsa_output.usb-Blue_Microphones_Yeti_Stereo_Microphone_REV8-00.analog-stereo.monitor   module-alsa-card.c  s16le 2ch 44100Hz   SUSPENDED
3   alsa_input.usb-Blue_Microphones_Yeti_Stereo_Microphone_REV8-00.analog-stereo    module-alsa-card.c  s16le 2ch 44100Hz   RUNNING
Second: To set a default output device run the command

pactl set-default-sink <'output_device_name'>
Example: pactl set-default-sink 'alsa_output.pci-0000_00_1f.3.analog-stereo'

Now, to make this work at every restart, follow this :

First, open the file /etc/pulse/default.pa using :

sudo -H gedit /etc/pulse/default.pa
Then scroll to the bottom of the file, where two lines starting with set- will be commented out.

Now, uncomment these lines and replace the words input and output with the number of the sink (for output) / source (for input) that you want to be the default.

Example (sets both default input and output):

### Make some devices default
set-default-sink 3
set-default-source 3
PS: As discussed in the comments with Bim, it is also possible (according to him) to put the input/output name in /etc/pulse/default.pa instead of the input/output number.

Example:

### Make some devices default
set-default-sink alsa_output.pci-0000_00_1f.3.analog-stereo
set-default-source alsa_output.pci-0000_00_1f.3.analog-stereo.monitor
After doing this, save and exit. Then, delete the ~/.config/pulse directory by running rm -r ~/.config/pulse, and then reboot the system. Once the system reboots, the appropriate devices should now be set as the defaults.


EDIT:
As mentioned by ahmorris in his answer, some had to comment this line load-module module-switch-on-connect in the file /etc/pulse/default.pa to be # load-module module-switch-on-connect in order to make the changes persistent.
```
