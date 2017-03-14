import os
import datetime
import pickle

def time_str():
    """
    Returns string-formatted local time in format hours:minutes:seconds.
    """
    return "".join(str(datetime.datetime.now().time()).split(".")[0])

def date_str():
    """
    Returns string-formatted local date in format year-month-day.
    """
    return str(datetime.datetime.now().date())

def uniq_filename(dir_path, pattern, ext=""):
    """
    Returns an unique filename in directory path that starts with pattern.
    """
    dir_path = os.path.abspath(dir_path)

    files = [f for f in os.listdir(dir_path) if \
             os.path.exists(os.path.join(dir_path, f))]
    num = len([f for f in files if f.startswith(pattern)])

    filename = pattern + (("_" + str(num)) if num else "") + ext

    return filename

def uniq_filepath(dir_path, pattern, ext=""):
    """
    Returns an unique filepath in directory path that starts with pattern.
    """
    dir_path = os.path.abspath(dir_path)
    return os.path.join(dir_path, uniq_filename(dir_path, pattern, ext))

def get_ext(filepath, sep="."):
    if not sep in filepath:
        return ""
    return filepath.split(sep)[-1]

def open_mp(filepath, *args, **kwargs):
    ext = get_ext(filepath)

    if ext == "gz":
        import gzip.open
        open_f = gzip.open
    elif ext == "bz2":
        import bz2.open
        open_f = bz2.open
    else:
        open_f = open

    return open_f(filepath, *args, **args)

def pickle(obj, filepath, protocol=4):
    with open_mp(filepath, "wb") as f:
        pickle.dump(obj, f, protocol=4)

def unpickle(filepath):
    with open_mp(filepath, "rb") as f:
        return pickle.load(f)
