from collections import deque
import skimage.transform
from constants import constants
import numpy as np

class EnvObs:
    def __init__(self):
        self.buffer = deque(maxlen=4)

    def observation(self, s):
        """
        input: state from vizdoom game
        """
        if s == None:
            obs = np.zeros(constants['FRAME_SHAPE'])
        else:
            obs = s.screen_buffer
            obs = skimage.transform.resize(obs, constants['FRAME_SHAPE'])
        self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=-1)
        return obsNew.astype(np.float32)

    def reset(self, s):
        """
        input: state from vizdoom game
        """
        obs = s.screen_buffer
        obs = skimage.transform.resize(obs, constants['FRAME_SHAPE'])
        self.buffer.clear()
        for _ in range(4-1):    # duplicate first frame
            self.buffer.append(obs)
        self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=-1)
        return obsNew.astype(np.float32)