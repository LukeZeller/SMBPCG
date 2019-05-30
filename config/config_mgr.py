import os, pathlib, yaml

# Name of Config YAML file
CONFIG_FILE = 'config.yml'
# Relative path (from project root) to java classes
JAVA_CLASSPATH_REL = 'simulator/target/classes'

# Absolute path to project config/ directory (as Path object)
CONFIG_DIR_PATH = pathlib.Path(__file__).resolve().parent
# String version of CONF_DIR_PATH
CONFIG_DIR = str(CONFIG_DIR_PATH)

# Absolute path to project root directory (as Path object)
ROOT_DIR_PATH = CONFIG_DIR_PATH.parent
# String version of ROOT_DIR_PATH
ROOT_DIR = str(ROOT_DIR_PATH)

# Global variables for config data
conf_data = None
git_placeholder = None

def load_config(force_load=False):
    global conf_data, git_placeholder
    # Do not reload config unless force_load is True
    if conf_data is not None and not force_load:
        return
    with open(get_absolute_path(CONFIG_FILE, CONFIG_DIR), 'r') as conf_f:
        conf_data = yaml.safe_load(conf_f)
    git_placeholder = conf_data['git_placeholder']

# ---- General Helper Functions ----

# Returns absolute path to file
# Arg type must be Path or str
def get_absolute_path(rel_p, base_p = ROOT_DIR_PATH):
    if type(rel_p) is str:
        rel_p = pathlib.Path(rel_p)
    if type(base_p) is str:
        base_p = pathlib.Path(base_p)
    return str((base_p / rel_p).resolve())

# ---- Environment Helper Functions ----

# TODO: Implement default placeholder to use 

def setup_environment():
    load_config()

    # Set/update environment for current python VM
    for var, val in conf_data['environment'].items():
        if val == git_placeholder:
            raise ValueError(
                ("Version control placeholder value present in configuration key '"
                 + var + "'. Update " + CONFIG_FILE + " locally to reflect proper "
                 "configuration information."
                )
            )
        _env_set_or_append(var, val)
        # print('JAVA_HOME=' + str(os.environ['JAVA_HOME']))


    # ---- Some hardcoded default behaviors ----

    # If JAVA_HOME is set but PATH is not in conf_data, use %JAVA_HOME%/server
    # as PATH to JVM
    if 'JAVA_HOME' in os.environ and 'PATH' not in conf_data['environment']:
        _env_set_or_append('PATH', get_absolute_path("server", os.environ['JAVA_HOME'].strip(';')))

    # If CLASSPATH is not in conf_data, use [Directory of config.py] + JAVA_CLASSPATH_REL
    if 'CLASSPATH' not in conf_data['environment']:
        _env_set_or_append('CLASSPATH', get_absolute_path(JAVA_CLASSPATH_REL))

def _env_set_or_append(var, val):
    # Argument val is assumed not to have terminating semicolon
    val += ';'
    if var in os.environ:
        if os.environ[var][-1] != ';':
            os.environ[var] += ';'
        # Withoug this condition, identical values can be repeatedly appended to the same
        # environment variable. This can occur, for example, if multiple processes with
        # distinct memory but shared environment import this module
        if val not in os.environ[var]:
            os.environ[var] += val
    else:
        os.environ[var] = val
