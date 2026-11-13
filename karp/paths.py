"""Locations of the data files shipped with KARP (karp/data/)."""
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def data_file(*parts):
    """Path of a file under karp/data/, e.g. data_file("masks", "stellar.txt")."""
    return os.path.join(DATA_DIR, *parts)
