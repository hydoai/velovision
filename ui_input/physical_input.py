import RPi.GPIO as GPIO

config = {
        'push_button_1' : 7,
        'push_button_2' : 15,
        'toggle_switch_1' : 29,
        'toggle_switch_2' : 31,
        }

GPIO.setmode(GPIO.BOARD)

for source in config:
    pin = config[source]
    GPIO.setup(pin, GPIO.IN)

while True:
    try:
        push_button_1 = GPIO.input(config['push_button_1'])
        push_button_2 = GPIO.input(config['push_button_2'])
        toggle_switch_1 = GPIO.input(config['toggle_switch_1'])
        toggle_switch_2 = GPIO.input(config['toggle_switch_2'])

        print(f"push_button_1:{push_button_1}")
        print(f"push_button_2:{push_button_2}")
        print(f"toggle_switch_1:{toggle_switch_1}")
        print(f"toggle_switch_2:{toggle_switch_2}")
    except KeyboardInterrupt:
        for source in config:
            pin = config[source]
            GPIO.cleanup(pin)


