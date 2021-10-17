# Control Interface: Registering user input

## GPIO-based digital toggle/push button switch read

Simple one button switch electrical diagram and sample code:
![](jetson-gpio-basic.png)

Applied code: shutdown all gstreamer streams (sending EOS to video recording, thereby making the files complete): [safe_shutdown.py](safe_shutdown.py). This shut
