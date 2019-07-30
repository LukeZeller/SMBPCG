from common.constants import DEFAULT_LEVEL_ROOT_DIRECTORY
from common.plotting import _get_unique_file
from common.level import level_to_jpg, level_to_ascii_str

def save_latent_vector(lv, name, fitness = None):
    root_dir = DEFAULT_LEVEL_ROOT_DIRECTORY
    with open(_get_unique_file(f"{root_dir}/latent_vectors/{name}.txt"), 'w') as lv_file:
        lv_as_string = " ".join([str(elem) for elem in lv])
        print(lv_as_string, file = lv_file)
        if fitness:
            print(fitness, file = lv_file)

def save_level(level, name, is_pre_lstm):
    root_dir = DEFAULT_LEVEL_ROOT_DIRECTORY
    lstm_dir = "prelstm" if is_pre_lstm else "postlstm"
    level_to_jpg(level, 
                 _get_unique_file(f"{root_dir}/level_images/{lstm_dir}/{name}"),
                 trim_buffer = False)
    text = level_to_ascii_str(level)
    with open(_get_unique_file(f"{root_dir}/level_asciis/{lstm_dir}/{name}.txt"), 'w') as level_file:
        print(text, file = level_file)