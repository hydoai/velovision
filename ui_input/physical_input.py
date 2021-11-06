from threading import Thread
import os
import time
from time import sleep
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD) # use board pin number (consecutive) instead of GPIO numbers.

def safe_shutdown():
    PRE_SHUTDOWN_CMDS= [
            "pkill -e --signal SIGINT gst-launch-1.0", # send SIGINT will trigger End of Stream on currently recording videos.
            "echo GST streams ended"
            ]
    for CMD in PRE_SHUTDOWN_CMDS:
        os.system(CMD)
        print("Pre-shutdown commands have run. Shutting down in 3 seconds.")
        time.sleep(3)
        os.system("shutdown -h  now")
class Pins:
    '''
    Interface for watching GPIO binary inputs within another script.

    Usage: Create a Pins object, then write a function that will run whenver a button is pressed (True). Then call start().
    Choose among: 'shutdown', 'front_toggle', 'rear_toggle'

    def safe_shutdown():
        ...
        ...

    pins = Pins()
    pins.setup_function('shutdown', safe_shutdown)
    pins.start()

    If there are any arguments to the functions, pass them as a tuple as a third argument to setup_function()

    If the script is looped, and you want to access the boolean values at each loop, it can be accessed with:

    while True:
        ... something ...

        is_toggle_pin_true = pins.bool('front_toggle')

        ... do something with boolean switch value ...


    '''
    def __init__(self, 
            black_button_pin = 7,
            red_button_pin = 15,
            front_toggle_pin = 29,
            rear_toggle_pin = 31,
            ):
        self.pins = {
                'black_button': black_button_pin,
                'red_button' : red_button_pin,
                'front_toggle' : front_toggle_pin,
                'rear_toggle' : rear_toggle_pin,
                }
        self.bool = {
                'black_button': False,
                'red_button': False,
                'front_toggle' : False,
                'rear_toggle' : False,
                }
        self.fns = {
                'black_button' : None,
                'red_button' : None,
                'front_toggle' : None,
                'rear_toggle' : None,
                }
        self.fn_args = {
                'black_button':None,
                'red_button':None,
                'front_toggle' : None,
                'rear_toggle' : None,
                }
        self.flip_true= {
                'black_button':False,
                'red_button':False,
                'front_toggle' : False,
                'rear_toggle' : False,
                }

        for button in self.pins:
            pin = self.pins[button]
            GPIO.setup(pin, GPIO.IN)

    def setup_function(self, pin_desc, function, function_args=None, flip_true=False):
        '''
        Set up function to run when pin is ever true

        Args:
            pin desc (string): pin description (see self.pins dictionary keys) ('shutdown', 'front_toggle', 'rear_toggle')
            function (function): python function
            function_args (list): list of function arguments
            flip_true (boolean): in case switch has been installed the wrong way, flip the booleans so that 'function' runs on 'False' input.
        '''
        self.fns[pin_desc] = function

        if function_args is not None:
            if type(function_args) != list:
                function_args = list(function_args)
        self.fn_args[pin_desc] = function_args
        
        self.flip_true[pin_desc] = flip_true

    def start(self):
        def pin_loop():
            while True:
                sleep(0.1)
                try:
                    for key in self.pins:
                        function_args = self.fn_args[key]
                        function = self.fns[key]
                        flip_true = self.flip_true[key]
                        activation = GPIO.input(self.pins[key])

                        # genuine boolean is XOR of flip_true and activation 
                        self.bool[key] = bool(activation) != bool(flip_true)
                        if bool(self.bool[key]):
                            if function is not None:
                                if function_args is not None:
                                    function(*function_args)
                                else: 
                                    function()

                except KeyboardInterrupt:
                    for source in self.bool[key]:
                        pin = self.pins[source]
                        GPIO.cleanup(pin)
        p = Thread(target=pin_loop, args=([]))
        p.start()

if __name__ == "__main__":
    def demo_print(text, text2):
        print(f"WOAHH this is something you wrote: {text}. Also, {text2}")

    pins = Pins()

    pins.setup_function('red_button', demo_print, ('Mr. and Mrs. Bob Vance', 'I send it back'))

    pins.start()
    while True:
        sleep(0.1)
        for button in pins.bool:
            print(button, ': ', pins.bool[button])

