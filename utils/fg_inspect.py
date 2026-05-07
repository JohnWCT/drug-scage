import sys
from functools import cache
from unittest import TestCase

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from echo_logger import *

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from _config import pdir
from data_process.compound_tools import get_all_matched_fn_ids_returning_tuple, str_one_fg_matches
from data_process.function_group_constant import FUNCTION_GROUP_LIST_FROM_DAYLIGHT
from data_process.utils_dataprocess import auto_read_list
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as mpl
from rdkit import Chem
from tqdm import tqdm
import json

@cache
def get_spatial_pos_frequency_arr():
    with open(Path(pdir) / 'dump/spatial_pos_status.json') as file:
        data = json.load(file)
        to_return = []
        for id, frequency in data.items():
            to_return.append(20_0000 * 10 / (frequency + 1))
        return np.log(np.array(to_return) + 1)


@cache
def get_pair_distances_frequency_arr():
    with open(Path(pdir) / 'dump/pair_distances_status.json') as file:
        data = json.load(file)
        to_return = []
        for id, frequency in data.items():
            to_return.append(20_0000 * 10 / (frequency + 1))
        return np.log(np.array(to_return) + 1)


@cache
def get_angle_frequency_arr():
    with open(Path(pdir) / 'dump/angle_status_20.json') as file:
        data = json.load(file)
        to_return = []
        for id, frequency in data.items():
            to_return.append(20_0000 / (frequency + 1))
        return np.log(np.array(to_return) + 1)