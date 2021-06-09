import os
import glob
import logging

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')
plt.style.use("~/cerfacs.mplstyle")
plt.rcParams['axes.grid'] = False

logging.basicConfig(
    level=logging.INFO,
    format='\n > %(asctime)s | %(name)s | %(levelname)s \n > %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MARKERS = 0

SHOW = 1
PLOT_FLAME_SPEED = 1
PLOT_RESOLVED_HR = 1
PLOT_OVERPRESSURE = 1
PLOT_FLAME_POSITION = 1
PLOT_F_MEAN_FLAME = 1

factor_figsize = 1.25
FIGSIZE = factor_figsize * 4, factor_figsize * 3
EXT_LIST = ['png']

# Choosing cases to plot and the legend {name: dir}
CASE_DIR = {
    r"2D DNS, $\rm Le = 1$": "2D_DNS_Le1",
    r"10 cells": "2p5D_10CellsAcross",
    r"15 cells": "2p5D_15CellsAcross",
    r"20 cells, $R_0 = 7$ mm": "2p5D_20CellsAcross",
    r"20 cells, $R_0 = 14$ mm": "2p5D_20CellsAcross_R014mm",
    r"40 cells": "2p5D_40CellsAcross",
}

logging.info("Plotting data:", CASE_DIR.keys())

if __name__ == "__main__":

    data = {}
    for (idx, (my_case, my_dir)) in enumerate(CASE_DIR.items()):
        logging.info("Working on case: %s" % my_case)
        data[my_case] = {}

        # Read AVBP temporals
        if PLOT_OVERPRESSURE or PLOT_RESOLVED_HR:
            df_mmm = pd.read_hdf(os.path.join(my_dir, 'avbp_mmm.h5'))
            # df_mmm = df_mmm.dropna()
            data[my_case]['avbp_mmm'] = df_mmm

        if PLOT_FLAME_POSITION or PLOT_FLAME_SPEED:
            track = sorted(glob.glob(os.path.join(
                my_dir, 'avbp_track_condition*.h5')))
            for _tracker in track:
                file = os.path.split(_tracker)[-1]
                file = file.replace(".h5", "")
                print(_tracker, file)
                df_track = pd.read_hdf(os.path.join(_tracker))
                # df_track = df_track.dropna()
                data[my_case][file] = df_track

        if PLOT_OVERPRESSURE:
            probes = sorted(glob.glob(os.path.join(
                my_dir, 'avbp_local_probe*.h5')))
            for probe in probes:
                file = os.path.split(probe)[-1]
                file = file.replace("avbp_local_", "").replace(".h5", "")
                print(probe, file)
                df_probe = pd.read_hdf(os.path.join(probe))
                # df_probe = df_probe.dropna()
                data[my_case][file] = df_probe
            print(data[my_case].keys())

        logging.info("%s: data loaded" % my_case)

    # crop simulation to 40 cm in y_tip
    logging.info("Debug crop simulation at 40cm in ytip")
    for _case in data.keys():
        logging.info("case: %s" % _case)
        if 'cell' in _case:
            logging.info("'cell' found in name, case: %s" % _case)
            time_40_cm = -1.0
            for dataframe_name in data[_case].keys():
                logging.info(dataframe_name)
                if 'track' in dataframe_name:
                    logging.info('track found')
                    dataframe = data[_case][dataframe_name]
                    dataframe = dataframe[dataframe['y_max'] < 4e-1]
                    time_40_cm = max(time_40_cm, dataframe['t'].max())
                    logging.info('time -->', time_40_cm)
            for dataframe_name in data[_case].keys():
                logging.info('crop %s' % dataframe_name)
                dataframe = data[_case][dataframe_name]
                dataframe = dataframe[dataframe['t'] < time_40_cm]
                data[_case][dataframe_name] = dataframe

    # -----------------------------------
    #  Plotting ------------------------
    # -----------------------------------

    if PLOT_OVERPRESSURE:
        # overpressure
        fig_overP, ax_overp = plt.subplots(
            1, 1, sharex=False, figsize=FIGSIZE)
        ax_overp.set_xlabel("Time [s]")
        ax_overp.set_ylabel("Overpressure [mbar]")

    if PLOT_FLAME_SPEED:
        # St vs pos
        fig_speed_vs_pos, ax_st_pos = plt.subplots(1, 1, figsize=FIGSIZE)
        ax_st_pos.set_xlabel("Flame front position [m]")
        ax_st_pos.set_ylabel("Flame front speed [m.s$^{-1}$]")

    if PLOT_F_MEAN_FLAME:
        # flame pos (debug)
        fig_thick, ax_thick = plt.subplots(1, 1, figsize=FIGSIZE)
        ax_thick.set_xlabel("Time [s]")
        ax_thick.set_ylabel(r"Flame Thickness $F$ in flame [-]")

    if PLOT_FLAME_POSITION:
        # flame pos (debug)
        fig_flame_pos, ax_flame_pos = plt.subplots(1, 1, figsize=FIGSIZE)
        ax_flame_pos.set_xlabel("Time [s]")
        ax_flame_pos.set_ylabel(r"Flame front position ($y$) [m]")

    if PLOT_RESOLVED_HR:
        # Resolved Heat Release
        fig_res_hr, axes_res_hr = plt.subplots(
            2, 1, figsize=(5, 5), sharex=True)
        ax_hr, ax_perc_hr = axes_res_hr[0], axes_res_hr[1]
        ax_hr.set_ylabel("Heat Release [$J.s^{-1}$]")
        ax_perc_hr.set_ylabel("Resolved Heat Release [%]")
        ax_perc_hr.set_xlabel(r"Flame front position ($y$) [m]")

    # Plot LES results
    linestyle_str = 2 * [
        ('solid', '-'),  # Same as (0, ()) or '-'
        ('dashed', '--'),  # Same as '--'
        ('dashdot', '-.'),  # Same as '-.'
        ('dotted', ':'),  # Same as (0, (1, 1)) or '.'

        ('densely dashed', (0, (5, 1))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),
        ('densely dotted', (0, (1, 1))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),

        ('loosely dotted', (0, (1, 10))),
        ('loosely dashed', (0, (5, 10))),
        ('loosely dashdotted', (0, (3, 10, 1, 10)))]

    markers = ['o', '+', 'x', 's', 'D', 'v', '^', '<',
               '>', '1', '2', '3', '4', 'd', 'p', 'P', '*', 'D']

    for (idx, (my_case, my_dir)) in enumerate(CASE_DIR.items()):
        name_linstyle, linestyle = linestyle_str[idx]
        print(name_linstyle, linestyle)
        print(my_case)
        _ls = linestyle
        _color = "C%s" % idx

        _marker = None
        if MARKERS:
            _marker = markers[idx]

        if PLOT_OVERPRESSURE:
            df = data[my_case]["probe3"]
            ax_overp.plot(df['t'], df.overpressure_mbar,
                          ls=_ls,
                          color=_color,
                          marker=_marker,
                          markevery=0.05,
                          label=my_case)

        if PLOT_FLAME_SPEED:
            # if track condition is available
            try:
                df = data[my_case]["avbp_track_condition-temperature_gt_800_lt_1200"]
                ax_st_pos.plot(df['y_max_rolling'], df['y_max_speed'],
                               ls=linestyle,
                               color=_color,
                               marker=_marker,
                               #    markevery=0.05,
                               label=my_case)
            # cases from Omar
            except KeyError:
                df = data[my_case]["avbp_mmm"]
                print(df.columns.values)
                ax_st_pos.plot(df['Pos_y_max_rolling'], df['Pos_y_max_speed'],
                               ls=linestyle,
                               color=_color,
                               marker=_marker,
                               #    markevery=0.05,
                               label=my_case)

        if PLOT_F_MEAN_FLAME:
            df = data[my_case]["avbp_mmm"]

            if 'thickening_masked_mean' in df.columns.values:
                ax_thick.plot(df['t'], df['thickening_masked_mean'],
                              ls=linestyle,
                              color=_color,
                              marker=_marker,
                              markevery=0.05,
                              label=my_case)

        if PLOT_RESOLVED_HR:
            df_mmm = data[my_case]["avbp_mmm"]
            try:
                y_tip = data[my_case]["avbp_track_condition-temperature_gt_500_lt_1500"].y_max.values
            except KeyError:
                y_tip = df_mmm.Pos_y_max.values

            my_label_res_hr = None
            my_label_sgs_hr = None
            my_label_tot_hr = None

            # if currently running the 2 dataframe may not have the same length
            length = min(len(y_tip), len(df.HR_mean))

            # Debug
            for col in df_mmm.columns.values:
                if 'HR' in col:
                    print(col)

            if not idx:
                my_label_res_hr = r"$\dot{\omega}_{\rm res}$"
                my_label_sgs_hr = r"$\dot{\omega}_{\rm sgs}$"
                my_label_tot_hr = r"$\dot{\omega}_{\rm tot}$"

            ax_hr.plot(y_tip[:length], df_mmm.HR_lam_mean[:length], '-',
                       color=_color,
                       marker=_marker,
                       markevery=0.05,
                       label=my_label_res_hr)
            ax_hr.plot(y_tip[:length], df_mmm.HR_sgs_mean[:length], '-.',
                       color=_color,
                       marker=_marker,
                       markevery=0.05,
                       label=my_label_sgs_hr)
            # ax_hr.plot(y_tip, df_mmm.HR_mean, '--',
            #             color=_color,
            #             marker=_marker,
            #             markevery=0.05,
            #             label=my_label_tot_hr)

            ax_perc_hr.plot(y_tip[:length], df_mmm.percentage_res_HR[:length],
                            color=_color,
                            marker=_marker,
                            markevery=0.05,
                            label=my_case)

        if PLOT_FLAME_POSITION:
            try:
                df = data[my_case]["avbp_track_condition-temperature_gt_800_lt_1200"]
                ax_flame_pos.plot(df.t, df.y_max,
                                  ls=linestyle,
                                  color=_color,
                                  marker=_marker,
                                  markevery=0.05,
                                  label=my_case)
            except KeyError:
                df = data[my_case]["avbp_mmm"]
                ax_flame_pos.plot(df.t, df["Pos_y_max"],
                                  ls=linestyle,
                                  color=_color,
                                  marker=_marker,
                                  markevery=0.05,
                                  label=my_case)

    my_str = ''
    for idx, my_case in enumerate(CASE_DIR):
        my_str += my_case.replace(" ", "_")
        if idx != len(list(CASE_DIR.keys())) - 1:
            my_str += '_vs_'
        my_str = my_str.replace(",", "")
        my_str = my_str.replace(".", "_")
        my_str = my_str.replace("__", "_")
        my_str = my_str.replace("__", "_")
        my_str = my_str.replace("$", "")
        my_str = my_str.replace(r"\rm", "")

    print(my_str)

    if PLOT_OVERPRESSURE:
        # Save overpressure figure
        ax_overp.legend(ncol=1)
        # ax_overp.set_xlim(left=0)
        # fig_overP.tight_layout()

        # Save overpressure figure
        for _ext in EXT_LIST:
            fig_overP.savefig("Figures/Figure_overP_.%s.%s" % (my_str, _ext),
                              bbox_inches='tight',
                              transparent=False,
                              pad_inches=0.02)

    if PLOT_FLAME_SPEED:
        # Save flame
        fig_speed_vs_pos.tight_layout()
        ax_st_pos.legend(ncol=1)
        # ax_st_pos.set_xlim(left=0)
        # ax_st_pos.set_ylim(ymin=0)
        for _ext in EXT_LIST:
            fig_speed_vs_pos.savefig(r"Figures/Figure_FlameSpeed.%s.%s" % (my_str, _ext),
                                     bbox_inches='tight',
                                     pad_inches=0.02)

    if PLOT_FLAME_POSITION:
        # ax_flame_pos.set_xlim(left=0)
        # ax_flame_pos.set_ylim(bottom=0)
        ax_flame_pos.legend()
        for _ext in EXT_LIST:
            fig_flame_pos.savefig("Figures/Figure_FlamePosition.%s.%s" % (my_str, _ext),
                                  bbox_inches='tight',
                                  pad_inches=0.02)

    if PLOT_F_MEAN_FLAME:
        # ax_thick.set_xlim(left=0)
        # ax_thick.set_ylim(bottom=0)
        ax_thick.legend()
        for _ext in EXT_LIST:
            fig_thick.savefig("Figures/Figure_thickening_masked.%s.%s" % (my_str, _ext),
                              bbox_inches='tight',
                              pad_inches=0.02)

    if PLOT_RESOLVED_HR:
        # ax_hr.set_ylim(bottom=0)
        ax_hr.legend()
        ax_perc_hr.legend()
        # ax_perc_hr.set_ylim(top=100, bottom=40)
        # ax_perc_hr.set_xlim(left=0, right=25)
        for ext in ext_list:
            fig_res_hr.savefig("Figures/Figure_ResolvedHR.%s.%s" % (my_str, ext),
                               bbox_inches='tight',
                               pad_inches=0.02)

    if SHOW:
        plt.show()
