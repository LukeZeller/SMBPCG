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

# OS-dependent delimiter for environment variables -- : (colon) for Posix,
# ; (semicolon) for Windows. Conveniently, the distinction is identical to
# the PosixPath vs. WindowsPath distinction in pathlib so we have basically
# no work to do.
if isinstance(ROOT_DIR_PATH, pathlib.PosixPath):
    SYS_ENV_DELIMITER = ':'
elif isinstance(ROOT_DIR_PATH, pathlib.WindowsPath):
    SYS_ENV_DELIMITER = ';'
else:
    raise RuntimeError( (
        "Unexpected error: System appears to be neither Windows nor posix-like. "
        "Are you attempting to run this code on a TI-84?" )
    )

# Global variables for config data
conf_data = None
git_placeholder = None
default_placeholder = None

def load_config(force_load=False):
    global conf_data, git_placeholder, default_placeholder
    # Do not reload config unless force_load is True
    if conf_data is not None and not force_load:
        return
    with open(get_absolute_path(CONFIG_FILE, CONFIG_DIR), 'r') as conf_f:
        conf_data = yaml.safe_load(conf_f)
    git_placeholder = conf_data['git_placeholder']
    default_placeholder = conf_data['default_placeholder']

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

# TODO: Implement default placeholder for unused/preset/default fields

def setup_environment():
    load_config()

    # Set/update environment for current python VM
    for var, cfg_info in conf_data['environment'].items():
        # print(var + " has value " + str(cfg_info['value']))
        if cfg_info['value'] == default_placeholder:
            # print("Default @ " + var)
            continue
        if cfg_info['value'] == git_placeholder:
            raise ValueError(
                ("Version control placeholder value present in configuration key '"
                 + var + "'. Update " + CONFIG_FILE + " locally to reflect proper "
                 "configuration information."
                )
            )
        # This can be generalized, but for now enforcing the following behavior is sufficient:
        # JAVA_HOME does not allow appending (instead overwriting) but the other two path variables
        # (PATH and CLASSPATH) do.
        _env_set_or_append(var, cfg_info['value'], allow_append=cfg_info['appendable'])
        # print('JAVA_HOME=' + str(os.environ['JAVA_HOME']))

    # ---- Some hardcoded default behaviors ----

    # If JDK_HOME is set but PATH is not in conf_data, try %JAVA_HOME%/jre/bin/server as PATH to JVM.
    # As far as I know, this will only work on Windows installations of Java. However, on posix-like
    # systems the path to the JVM varies is both dependent on the system and the version of Java, so
    # we do not attempt to determine a proper default value. For posix systems, pyjnius tries to
    # determine the proper jvm location (in its setup.py), but we need to update the path variable
    # in Windows. 
    if 'JDK_HOME' in os.environ and not _has_value('PATH'):
        _env_set_or_append('PATH', get_absolute_path("jre/bin/server", os.environ['JDK_HOME'].strip(';')))

    # If CLASSPATH is not provided in conf_data, use [Directory of config.py] + JAVA_CLASSPATH_REL
    if not _has_value('CLASSPATH'):
        _env_set_or_append('CLASSPATH', get_absolute_path(JAVA_CLASSPATH_REL))

def _has_value(var):
    return ( var in conf_data['environment'] and
             conf_data['environment'][var]['value'] != default_placeholder )

def _env_set_or_append(var, val, allow_append=True):
    # Argument val is assumed not to have terminating delimiter
    if var in os.environ and allow_append:
        val += SYS_ENV_DELIMITER
        if os.environ[var][-1] != SYS_ENV_DELIMITER:
            os.environ[var] += SYS_ENV_DELIMITER
        # Withoug this condition, identical values can be repeatedly appended to the same
        # environment variable. This can occur, for example, if multiple processes with
        # distinct memory but shared environment import this module
        if val not in os.environ[var]:
            os.environ[var] += val
    else:
        os.environ[var] = val
