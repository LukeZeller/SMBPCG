import json
import numpy as np

import config.config_mgr as config_mgr

LEVEL_DATA_REL = 'data/levels'
LEVEL_DATA_DIR_PATH = config_mgr.get_absolute_path(LEVEL_DATA_REL)
LEVEL_DATA_DIR = str(LEVEL_DATA_DIR_PATH)

DEFAULT_LEVEL_HEIGHT = 14
DEFAULT_LEVEL_WIDTH = 28

char_int_map = {
    'X' : 0,
    'S' : 1,
    '-' : 2,
    '?' : 3,
    'Q' : 4,
    'E' : 5,
    '<' : 6,
    '>' : 7,
    '[' : 8,
    ']' : 9,
    'o' : 10,
    'B' : 11,
    'b' : 12
}

int_char_map = {
    0 : 'X',
    1 : 'S',
    2 : '-',
    3 : '?',
    4 : 'Q',
    5 : 'E',
    6 : '<',
    7 : '>',
    8 : '[',
    9 : ']',
    10 : 'o',
    11 : 'B',
    12 : 'b'
}

def load_level_from_json(json_fname):
    with open(config_mgr.get_absolute_path(
            'json/' + json_fname, LEVEL_DATA_DIR_PATH), 'r') as json_f:
        level_json = json.loads(json_f.read())
    return Level(data=level_json)

class Level(object):
    # Initialize Level object with width / height or data
    # Note: If data array is present, width / height parameters are ignored.
    #       In this case, Level should be initialized as Level(data = ...)
    def __init__(self, width = DEFAULT_LEVEL_WIDTH, height = DEFAULT_LEVEL_HEIGHT, data = None):
        if data is not None:
            self.__initialize_from_data(data)
        else:
            self.width = width
            self.height = height
            # Initialize array with 1's ( = 'S') so level is empty space
            self.__tiles = np.ones((height, width), dtype=np.int)

    # Currently, data be python list or numpy array
    def __initialize_from_data(self, data):
        if type(data) is list:
            self.width = len(data[0])
            self.height = len(data)
            self.__tiles = np.array(data)
        elif type(data) is np.ndarray:
            self.width = data.shape[1]
            self.height = data.shape[0]
            self.__tiles = np.copy(data)
        else:
            raise TypeError("Parameter data must be python list or numpy array.")

    # Check that position parameters (x, y) are in bounds
    def __bounds_check(self, x, y):
        if x < 0 or x >= self.width:
            raise ValueError("Horizontal position parameter x is out of bounds.")
        if y < 0 or y >= self.height:
            raise ValueError("Vertical position parameter y is out of bounds.")

    def get_data(self):
        return self.__tiles.tolist()

    def get_tile_char(self, x, y):
        self.__bounds_check(x, y)
        return int_char_map[int(self.__tiles[y, x])]

    def get_tile_int(self, x, y):
        self.__bounds_check(x, y)
        return int(self.__tiles[y, x])
    
    def set_tile_char(self, x, y, tile_c):
        self.__bounds_check(x, y)
        if tile_c not in char_int_map:
            raise ValueError("Invalid tile character provided.")
        self.tiles[y, x] = char_int_map[tile_c]

    def set_tile_int(self, x, y, tile_i):
        self.__bounds_check(x, y)
        if tile_i not in int_char_map:
            raise ValueError("Invalid tile integer provided.")
        self.tiles[y, x] = tile_i
