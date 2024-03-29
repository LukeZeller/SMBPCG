import json
import numpy as np

import config.config_mgr as config_mgr

config_mgr.setup_environment()
import jnius

# --- Java class names ---
MARIO_LEVEL_VIEWER_CLASS = 'viewer.MarioLevelViewer'
LEVEL_PARSER_CLASS = 'ch.idsia.mario.engine.level.LevelParser'

# Project classes
_J_MarioLevelViewer = jnius.autoclass(MARIO_LEVEL_VIEWER_CLASS)
_J_LevelParser = jnius.autoclass(LEVEL_PARSER_CLASS)

# Standard Library classes
_J_ArrayList = jnius.autoclass('java.util.ArrayList')
_J_Integer = jnius.autoclass('java.lang.Integer')

LEVEL_DATA_REL = 'data/full_levels'
LEVEL_DATA_DIR_PATH = config_mgr.get_absolute_path(LEVEL_DATA_REL)
LEVEL_DATA_DIR = str(LEVEL_DATA_DIR_PATH)

LEVEL_OUTPUT_REL = 'data/generated_levels'
LEVEL_OUTPUT_DIR_PATH = config_mgr.get_absolute_path(LEVEL_OUTPUT_REL)
LEVEL_OUTPUT_DIR = str(LEVEL_OUTPUT_DIR_PATH)


DEFAULT_LEVEL_HEIGHT = 14
DEFAULT_LEVEL_WIDTH = 28

char_int_map = {
    'X': 0,
    'S': 1,
    '-': 2,
    '?': 3,
    'Q': 4,
    'E': 5,
    '<': 6,
    '>': 7,
    '[': 8,
    ']': 9,
    'o': 10,
    'B': 11,
    'b': 12
}

int_char_map = {
    0: 'X',
    1: 'S',
    2: '-',
    3: '?',
    4: 'Q',
    5: 'E',
    6: '<',
    7: '>',
    8: '[',
    9: ']',
    10: 'o',
    11: 'B',
    12: 'b'
}

java_level_cache = {}
def _get_java_level(level):
    if level in java_level_cache:
        return java_level_cache[level].copy()

    # j_level_arraylist is a Java ArrayList of rows in the level
    j_level_list = _J_ArrayList(level.height)
    for y in range(level.height):
        # Each row is a List of Integers representing tiles
        j_row_list = _J_ArrayList(level.width)
        for x in range(level.width):
            # Second argument expects java Object but python int converts to
            # java primitive int, so we must wrap in Integer
            j_row_list.add(_J_Integer(level.get_tile_int(x, y)))
        j_level_list.add(j_row_list)

    res = _J_LevelParser.createLevelJson(j_level_list)
    java_level_cache[level] = res
    return res.copy()


def load_level_from_json(json_fname):
    with open(config_mgr.get_absolute_path(
            'json/' + json_fname, LEVEL_DATA_DIR_PATH), 'r') as json_f:
        level_json = json.loads(json_f.read())
    return Level(data=level_json)

def load_level_from_ascii(ascii_fname):
    with open(config_mgr.get_absolute_path(
            'text/' + ascii_fname, LEVEL_DATA_DIR_PATH), 'r') as ascii_f:
        return load_level_from_ascii_str(ascii_f.read())

def load_level_from_ascii_str(s):
    lines = [line.strip() for line in s.split('\n')]
    while lines and not lines[-1]:
        lines.pop()
    level = Level(width = len(lines[0]), height = len(lines))
    for y, row in enumerate(lines):
        for x, char in enumerate(row):
            level.set_tile_char(x, y, char)
    return level

# The following functions each return a string representation of the level.
# If the fname parameter is specified, they will also attempt to write the output
# to data/level/X/fname where X IN ["json", "text"].

def _save_string(rel_path, contents):
    with open(config_mgr.get_absolute_path(
            rel_path, LEVEL_OUTPUT_DIR_PATH), 'w') as f:
        f.write(contents)

def level_to_json_str(level, fname = None):
    res_str = str(level.get_data())
    if fname is not None:
        rel_path = 'json/' + fname
        _save_string(rel_path, res_str)
    return res_str

def level_to_ascii_str(level, fname = None):
    # Warning: Do not read next 4 lines
    res_str = '\n'.join(map(
        lambda row: ''.join(map(lambda i: int_char_map[i], row)),
        level.get_data()
    )) + '\n'
    # Thanks.
    if fname is not None:
        rel_path = 'text/' + fname
        _save_string(rel_path, res_str)
    return res_str

def level_to_jpg(level, fname, trim_buffer):
    if ".jpg" in fname:
        fname = fname[:-4]
    _J_MarioLevelViewer.saveLevel(_get_java_level(level), fname, trim_buffer)


class Level(object):
    # Initialize Level object with width / height or data
    # Note: If data array is present, width / height parameters are ignored.
    #       In this case, Level should be initialized as Level(data = ...)
    def __init__(self, width=DEFAULT_LEVEL_WIDTH, height=DEFAULT_LEVEL_HEIGHT, data=None):
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

    def get_data(self, as_ndarray = False):
        if as_ndarray:
            return np.copy(self.__tiles)
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
            raise ValueError("Invalid tile character [{0}] provided".format(tile_c))
        self.__tiles[y, x] = char_int_map[tile_c]

    def set_tile_int(self, x, y, tile_i):
        self.__bounds_check(x, y)
        if tile_i not in int_char_map:
            raise ValueError("Invalid tile integer [{0}] provided".format(tile_i))
        self.__tiles[y, x] = tile_i
        
    def __add__(self, other):
        assert self.height == other.height, "You can only append levels with the same height"
        level = Level(self.width + other.width, self.height)
        for i in range(self.height):
            for j in range(self.width):
                level.__tiles[i][j] = self.__tiles[i][j]
        for i in range(other.height):
            for j in range(other.width):
                level.__tiles[i][j + self.width] = other.__tiles[i][j]
        return level
        
