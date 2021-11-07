'''
Module to interface with speaker
'''
import os
import inspect
import pygame

class GiSpeaker:
    '''
    Project Gimondi hardware - specific speaker interface

    IMPORTANT: Audio files themselves need to have right or left Pan.
    In Audacity, the L-R slider next to the waveforms does this.
    '''
    def __init__(self, left_sound_filename='FYOONG-left.mp3', right_sound_filename='high-three-chirp-right.mp3'):
        pygame.mixer.init()

        audio_files_path = os.path.join(os.path.dirname(inspect.getfile(GiSpeaker)), 'audio_files')
       # audio_files_path = '/home/halpert/Gimondi/gi-edge/feedback_interface/audio_files'

        self.right_sound_path = os.path.join(audio_files_path, right_sound_filename)
        self.left_sound_path = os.path.join(audio_files_path, left_sound_filename)

        self.right_sound = pygame.mixer.Sound(self.right_sound_path)
        self.left_sound = pygame.mixer.Sound(self.left_sound_path)

        self.left_channel = pygame.mixer.Channel(1)
        self.right_channel = pygame.mixer.Channel(2)

    def play_left(self):
        if self.left_channel.get_busy(): # i don't think these actually work
            self.left_channel.stop() # intended to stop sound and start from the beginning if a new threat is detected
        self.left_channel.play(self.left_sound)

    def play_right(self):
        if self.right_channel.get_busy():
            self.right_channel.stop()
        self.right_channel.play(self.right_sound)

if __name__ == "__main__":
    import time
    # testing
    gi_speaker = GiSpeaker(left_sound_filename= 'xylo-up-left.mp3', right_sound_filename= 'FYOONG-right.mp3')
    while True:
        print('play left')
        gi_speaker.play_left()
        time.sleep(2)
        print('play right')
        gi_speaker.play_right()
        time.sleep(2)


