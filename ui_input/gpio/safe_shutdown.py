# Read push button input
# Send End of Stream message to recording streams
# Shutdown 
import os
import time
import RPi.GPIO as GPIO

in_pin = 15
PRE_SHUTDOWN_CMDS = [
        "pkill -e --signal SIGINT gst-launch-1.0", # sending SIGINT will trigger End of Stream on currently recording videos
        "echo GST streams ended"
        ]

GPIO.setmode(GPIO.BOARD) # using the same numbers as printed on the board itself
GPIO.setup(in_pin, GPIO.IN)

while True:
    inp = GPIO.input(15)
    pressed = not inp # inp is 1 when switch open, 0 when closed.
    print(pressed)
    time.sleep(0.01)

    if pressed:
        for CMD in PRE_SHUTDOWN_CMDS:
            os.system(CMD)
            print("Pre-shutdown commands have been executed")
            print("Shutting down in 3 seconds")
            time.sleep(3)
            os.system('shutdown -h now')



