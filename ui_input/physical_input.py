from multiprocessing import Process
from time import sleep
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD) # use board pin number (consecutive) instead of GPIO numbers.

class Pins:
    '''
    Interface for watching GPIO binary inputs within another script.

    Usage:

    '''
    def __init__(self, 
            power_on_pin = 7,
            shutdown_pin = 15,
            front_toggle_pin = 29,
            rear_toggle_pin = 31,
            ):
        self.pins = {
                'power_on' : power_on_pin,
                'shutdown' : shutdown_pin,
                'front_toggle' : front_toggle_pin,
                'rear_toggle' : rear_toggle_pin,
                }
        self.fns = {
                'power_on' : None,
                'shutdown' : None,
                'front_toggle' : None,
                'rear_toggle' : None,
                }
        self.fn_args = {
                'power_on' : None,
                'shutdown' : None,
                'front_toggle' : None,
                'rear_toggle' : None,
                }
        self.flip_true= {
                'power_on' : False,
                'shutdown' : False,
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
            pin desc (string): pin description (see self.pins dictionary keys) ('power_on', 'shutdown', 'front_toggle', 'rear_toggle')
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

                        # flip_true and activation are XOR 
                        print(key, bool(activation))
                        print(bool(activation) != bool(flip_true))
                        if bool(activation) != bool(flip_true):
                            if function is not None:
                                print('function:')
                                print(function)
                                print(type(function))
                                if function_args is not None:
                                    function(function_args)
                                else: 
                                    function()

                except KeyboardInterrupt:
                    for source in self.activation:
                        pin = self.pins[source]
                        GPIO.cleanup(pin)
        p = Process(target=pin_loop, args=([]))
        p.start()

if __name__ == "__main__":
    def simulated_shutdown():
        print("going to sleep!")
        sleep(1)
        print('3')
        sleep(1)
        print('2')
        sleep(1)
        print('1')
        sleep(1)
        import sys
        sys.exit()
    pins = Pins()
    pins.setup_function('shutdown', simulated_shutdown)
    pins.start()
