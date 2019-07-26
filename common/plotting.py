import os
import matplotlib.pyplot as plt

def _file_exists(path):
    return os.path.isfile(path)

def _get_unique_file(file):
    if not _file_exists(file):
        return file
    else:
        name, extension = os.path.splitext(file)
        suffix = 1
        while True:
            possible_file = f"{name}_{suffix}{extension}"
            if not _file_exists(possible_file):
                return possible_file
            suffix += 1
            
def plot_to_file(title, xs, ys, xlabel, ylabel, file_path):
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig(_get_unique_file(file_path))
    plt.clf()
