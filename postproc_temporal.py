import os
from posixpath import join
import sys
import glob
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter

from pyavbp.io.manipbin import readbin

# Definitions
P_REF = 101325.0
FACTOR_HR_2D_TO_3D = 1.0
WINDOW_SIZE = 401
OUTPUT_HDF = 1
OUTPT_CSV = 0

matplotlib.use('Qt5Agg')
plt.style.use("~/cerfacs.mplstyle")
plt.rcParams['axes.grid'] = False

logging.basicConfig(
    # filename='myfirstlog.log',
    level=logging.INFO,
    # level=logging.DEBUG,
    format='\n > %(asctime)s | %(name)s | %(levelname)s \n > %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Functions


def get_hdf5(_file):
    logger.info("Reading %s" % _file)
    data = readbin(_file)
    df = pd.DataFrame.from_dict(data)
    return df


def write_file(_file, dataframe):
    if OUTPUT_HDF:
        logger.info("Output HDF5 file %s" % _file)
        dataframe.to_hdf("%s.h5" % _file, key="AVBP", index=False)
    if OUTPT_CSV:
        logger.info("Output csv file %s" % _file)
        dataframe.to_csv("%s.csv" % _file, index=False)


def postproc_probe(file, df):
    logger.info("Postproc Probe: %s" % file)
    if 'P' in df.columns.values:
        df["overpressure_Pa"] = df["P"] - P_REF
        df["overpressure_mbar"] = 1e3 * 1e-5 * df["overpressure_Pa"]
    return df


def postproc_mmm(file, df):
    logger.info("Postproc avbp_mmm: %s" % file)
    logger.info(df.columns.values)

    # HR tot, lam, sgs and percentage
    if 'thickening_mean' in df.columns.values:
        logger.info('thickening_mean found')
        df['HR_mean'] *= FACTOR_HR_2D_TO_3D
        df['HR_lam_mean'] *= FACTOR_HR_2D_TO_3D
        df['HR_sgs_mean'] = df['HR_mean'] - df['HR_lam_mean']
    else:
        logger.info('thickening_mean NOT found')
        df['HR_lam_mean'] = df['HR_mean']
        df['HR_sgs_mean'] = 0.0 * df['HR_mean']

    df['percentage_res_HR'] = 100.0 * df['HR_lam_mean'] / df['HR_mean']

    # flame position and speed in Omar's simulations
    if 'Pos_y_max' in df.columns.values:
        logger.info(
            'Found Omar simulation, using Pos_y_max to compute velocity')
        col = 'Pos_y_max'
        dummy = df.describe().T
        dy = np.gradient(df[col].rolling(WINDOW_SIZE, left=True).mean())
        dx = np.gradient(df["t"].rolling(WINDOW_SIZE, left=True).mean())
        df['%s_rolling' % col] = df[col].rolling(
            WINDOW_SIZE, center=left).mean()
        df['%s_speed' % col] = dy/dx

    return df


def postproc_track_condition(file, df):
    logger.info('Postproc track condition: %s' % file)
    order = 2
    lst_track = ['x_min', 'x_max', 'y_min', 'y_max',
                 'z_min', 'z_max', 'radius_min', 'radius_max']

    for col in lst_track:
        # Weird: need to describe df to get rid of bug-endian buffer conflicts
        dummy = df.describe().T
        dy = np.gradient(df[col].rolling(WINDOW_SIZE, center=True).mean())
        dx = np.gradient(df["t"].rolling(WINDOW_SIZE, center=True).mean())
        df['%s_rolling' % col] = df[col].rolling(WINDOW_SIZE, center=True).mean()
        df['%s_speed' % col] = dy/dx
        # dy = np.gradient(savgol_filter(df[col], WINDOW_SIZE, order))
        # dx = np.gradient(savgol_filter(df["t"], WINDOW_SIZE, order))
        # df['%s_speed_savgol' % col] = dy/dx
        # df['%s_savgol' % col] = savgol_filter(df[col], WINDOW_SIZE, order)
    return df


def time(file, df):
    logger.info('Postproc, add time and time in ms for convenience: %s' % file)
    df["t"] = df["atime"]
    df["t_ms"] = 1e3 * df["atime"]
    return df


def treatments(file, df):
    logger.info('Seek for treatment: %s' % file)
    df = time(file, df)
    if "probe" in file:
        df = postproc_probe(file, df)
        return df
    elif "avbp_mmm" in file:
        df = postproc_mmm(file, df)
        return df
    elif "avbp_track_condition" in file:
        df = postproc_track_condition(file, df)
        return df
    return df


if __name__ == "__main__":
    logger.info('Start')
    dir_path_all = sorted(glob.glob(sys.argv[1]))
    #dir_path_all = ["2p5D_40CellsAcross"]
    # dir_path_all = ["2D_DNS_Le1"]
    logger.info('Directories found:', dir_path_all)

    for dir_path in dir_path_all:

        if os.path.isdir(dir_path):
            logger.info("Working on directory: %s" % dir_path)
        else:
            raise ValueError('"%s" is not even a directory!' % dir_path)

        # list all non hdf5 files
        temporal_files = [file for file in sorted(
            glob.glob(os.path.join(dir_path, "avbp_*"))) if not file.endswith(".h5") if not file.endswith(".csv")]

        for temporal in temporal_files:
            logger.info("temporal file: %s" % temporal)
            # Read file
            df = get_hdf5(temporal)
            # Add treatments if needed
            treatments(temporal, df)
            # write
            write_file(temporal, df)

    logger.info('Done')
