import os
import glob
import logging

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal

matplotlib.use('Qt5Agg')
plt.style.use("~/cerfacs.mplstyle")
plt.rcParams['axes.grid'] = False

logging.basicConfig(
    level=logging.INFO,
    format='> %(asctime)s | %(name)s | %(levelname)s | > %(message)s',
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
# factor_figsize = 0.9
FIGSIZE_X, FIGSIZE_Y = factor_figsize * 4, factor_figsize * 3
FIGSIZE = factor_figsize * 4, factor_figsize * 3
EXT_LIST = ['png']
SL0_REF = 0.4893081637390764  # m/s

# Geometry
X_MAX = 0.54
Y_MAX = 1.0

# # Choosing cases to plot and the legend {name: dir}
# CASE_DIR = {
#     r"2D DNS, $\rm Le = 1$": "2D_DNS_Le1",
#     r"10 cells": "2p5D_10CellsAcross",
#     r"15 cells": "2p5D_15CellsAcross",
#     r"20 cells, $R_0 = 7$ mm": "2p5D_20CellsAcross",
#     r"20 cells, $R_0 = 14$ mm": "2p5D_20CellsAcross_R014mm",
#     # r"20 cells, $R_0 = 14$ mm, PGS (0.09)": "2p5D_20CellsAcross_R014mm_PGS",
#     r"40 cells": "2p5D_40CellsAcross",
# }


# Choosing cases to plot and the legend {name: dir}
CASE_DIR = {
    r"2D DNS, $\rm Le = 1$": "2D_DNS_Le1",
    # r"2D LES, $\rm Le = 1$, 20 cells": "2D_LES_Le1",
    # r"2D LES, $\rm Le = 1$, 25 cells": "2D_LES_Le1_25CellsAcross",
    # r"2D LES, $\rm Le = 1$, 20 cells, efcy += 1.5": "2D_LES_Le1_FakeEfcy1p5",
    # r"2D LES, $\rm Le = 1$, 20 cells, no wall damping": "2D_LES_Le1_noWallDamping",
    # r"2D LES, $\rm Le = 1$, 20 cells, efcy += 1.5, $\beta=0.5$": "2D_LES_Le1_FakeEfcy1p5_beta0p5",
    # r"2D LES, $\rm Le = 1$, 20 cells, wall law (noslip)": "2D_LES_Le1_WallLaw",
    # r"2D LES, $\rm Le = 1$, 20 cells, wall law (slip)": "2D_LES_Le1_WallLawNormal",
    # r"2D LES, $\rm Le = 1$, TTGC, ": "2D_LES_Le1_TTGC",
    r"2D LES, $\rm Le = 1$, 20 cells, WL, NWD, $\beta=0.4$": "2D_LES_Le1_WallLawNoWallDampingBeta0p4",
    r"2D LES, $\rm Le = 1$, 20 cells, WL, NWD, $\beta=0.4$, PGS": "2D_LES_Le1_WallLawNoWallDampingBeta0p4_PGS",
    # r"2D LES, $\rm Le = 1$, 20 cells, WL, NWD, $\beta=0.5$": "2D_LES_Le1_WallLawNoWallDampingBeta0p5",
}

logging.info("Plotting data:")
logging.info(CASE_DIR)

def get_fft(df, var, sig_length=512):
    def get_welch(sig, sampling_freq, sig_length=sig_length):
        freq_welch, Pxx_spec = signal.welch(
            sig, sampling_freq, nperseg=sig_length, scaling='spectrum', return_onesided=True, detrend=False)
        spectrum_rms = np.sqrt(2. * Pxx_spec)
        return freq_welch, spectrum_rms
    # Nyquist-Shannon theorem
    df = df.dropna()
    T_sig = df['t'].max() - df['t'].min()
    sampling_period = T_sig / (len(df['t']) - 1.0)
    sampling_freq = 1. / sampling_period
    logging.info("Sampling Frequency = %3.3e [Hz]" % sampling_freq)
    sig = df[var] - df[var].mean()
    freq_welch, spectrum_rms = get_welch(sig, sampling_freq)
    return freq_welch, spectrum_rms

if __name__ == "__main__":

    data = {}
    for (idx, (my_case, my_dir)) in enumerate(CASE_DIR.items()):
        logging.info("Working on case: %s" % my_case)
        data[my_case] = {}

        # Read AVBP temporals
        if PLOT_OVERPRESSURE or PLOT_RESOLVED_HR or PLOT_FLAME_SPEED:
            df_mmm = pd.read_hdf(os.path.join(my_dir, 'avbp_mmm.h5'))
            # df_mmm = df_mmm.dropna()
            data[my_case]['avbp_mmm'] = df_mmm

        if PLOT_FLAME_POSITION or PLOT_FLAME_SPEED or PLOT_OVERPRESSURE:
            track = sorted(glob.glob(os.path.join(
                my_dir, 'avbp_track_condition*.h5')))
            for _tracker in track:
                file = os.path.split(_tracker)[-1]
                file = file.replace(".h5", "")
                logger.info(_tracker, file)
                df_track = pd.read_hdf(os.path.join(_tracker))
                # df_track = df_track.dropna()
                data[my_case][file] = df_track

        if PLOT_OVERPRESSURE:
            probes = sorted(glob.glob(os.path.join(
                my_dir, 'avbp_local_probe*.h5')))
            for probe in probes:
                file = os.path.split(probe)[-1]
                file = file.replace("avbp_local_", "").replace(".h5", "")
                logger.info(probe, file)
                df_probe = pd.read_hdf(os.path.join(probe))
                # df_probe = df_probe.dropna()
                data[my_case][file] = df_probe
            logger.info(data[my_case].keys())

        logging.info("%s: data loaded" % my_case)

    # # crop simulation to 40 cm in y_tip
    # logging.info("Debug crop simulation at 40cm in ytip")
    # for _case in data.keys():
    #     logging.info("case: %s" % _case)
    #     if 'cell' in _case:
    #         logging.info("'cell' found in name, case: %s" % _case)
    #         time_40_cm = -1.0
    #         for dataframe_name in data[_case].keys():
    #             logging.info(dataframe_name)
    #             if 'track' in dataframe_name:
    #                 logging.info('track found')
    #                 dataframe = data[_case][dataframe_name]
    #                 dataframe = dataframe[dataframe['y_max'] < 4e-1]
    #                 time_40_cm = max(time_40_cm, dataframe['t'].max())
    #                 logging.info('time --> %e' % time_40_cm)
    #         for dataframe_name in data[_case].keys():
    #             logging.info('crop %s' % dataframe_name)
    #             dataframe = data[_case][dataframe_name]
    #             dataframe = dataframe[dataframe['t'] < time_40_cm]
    #             data[_case][dataframe_name] = dataframe

    # -----------------------------------
    #  Plotting ------------------------
    # -----------------------------------

    if PLOT_OVERPRESSURE:
        # overpressure
        fig_overP, ax_overp = plt.subplots(
            1, 1, sharex=False, figsize=FIGSIZE)
        ax_overp.set_xlabel("Time [s]")
        ax_overp.set_ylabel("Overpressure [mbar]")

        fig_overP_vs_xy, axes_overp_xy = plt.subplots(
            1, 2, sharex=False, figsize=(2*FIGSIZE_X, FIGSIZE_Y))
        ax_overp_x, ax_overp_y = axes_overp_xy
        ax_overp_x.set_ylabel("Overpressure [mbar]")
        ax_overp_x.set_xlabel("Flame front speed along $x$ [m.s$^{-1}$]")
        ax_overp_y.set_xlabel("Flame front speed along $y$ [m.s$^{-1}$]")

    if PLOT_FLAME_SPEED:
        # St vs pos
        # fig_speed_vs_pos, ax_st_pos_y = plt.subplots(1, 1, figsize=(7,6))
        fig_speed_vs_pos, axes_st_pos = plt.subplots(
            1, 2, figsize=(2*FIGSIZE_X, FIGSIZE_Y))
        ax_st_pos_x, ax_st_pos_y = axes_st_pos
        ax_st_pos_x.set_xlabel("Flame front position ($x$) [m]")
        ax_st_pos_x.set_ylabel("Flame front speed along $x$ [m.s$^{-1}$]")

        ax_st_pos_y.set_xlabel("Flame front position ($y$) [m]")
        ax_st_pos_y.set_ylabel("Flame front speed along $y$ [m.s$^{-1}$]")

        fig_speed_vs_time, ax_st_time = plt.subplots(1, 1, figsize=FIGSIZE)
        ax_st_time.set_xlabel("Time [s]")
        ax_st_time.set_ylabel("Flame front speed [m.s$^{-1}$]")

        # fig_speed_fft, ax_st_fft = plt.subplots(1, 1, figsize=FIGSIZE)
        # ax_st_fft.set_xlabel("Frequency [Hz]")
        # ax_st_fft.set_ylabel(r"|FFT($s_{\rm tip}$)| [m.s$^{-2}$]")

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
            2, 2, figsize=(1.5*FIGSIZE_X, 1.5*FIGSIZE_Y), sharex=False, sharey=False)
        ax_hr_x, ax_perc_hr_x = axes_res_hr[0, 0], axes_res_hr[1, 0]
        ax_hr_x.set_ylabel("Heat Release [$J.s^{-1}$]")
        ax_perc_hr_x.set_ylabel("Resolved Heat Release [%]")
        ax_perc_hr_x.set_xlabel(r"Flame front position ($x$) [m]")

        ax_hr_y, ax_perc_hr_y = axes_res_hr[0, 1], axes_res_hr[1, 1]
        ax_hr_y.set_ylabel("Heat Release [$J.s^{-1}$]")
        ax_perc_hr_y.set_ylabel("Resolved Heat Release [%]")
        ax_perc_hr_y.set_xlabel(r"Flame front position ($y$) [m]")

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
        # logger.info(name_linstyle, linestyle)
        # logger.info(my_case)
        _ls = linestyle
        _color = "C%s" % idx

        _marker = None
        if MARKERS:
            _marker = markers[idx]

        # Get flame tip location
        df_mmm = data[my_case]["avbp_mmm"]
        try:
            y_tip = data[my_case]["avbp_track_condition-temperature_gt_500_lt_1500"].y_max
            x_tip = data[my_case]["avbp_track_condition-temperature_gt_500_lt_1500"].x_max
        except KeyError:
            y_tip = df_mmm.Pos_y_max
            x_tip = df_mmm.Pos_max

        if PLOT_OVERPRESSURE:
            df = data[my_case]["probe3"]
            ax_overp.plot(df['t'], df.overpressure_mbar,
                          ls='-',
                          color=_color,
                          marker=_marker,
                          markevery=0.05,
                          alpha=0.7,
                          label=my_case)
            # ax_overp.plot(df_mmm['t'], 1e-2 * (df_mmm.P_max - 101325),
            #               ls=_ls,
            #               lw=1.,
            #               color=_color,
            #               marker=_marker,
            #               markevery=0.05,
            #               label='%s (max)' % my_case)

            length = min(len(y_tip), len(df.overpressure_mbar))
            ax_overp_x.plot(x_tip[:length], (df.overpressure_mbar[:length]),
                            ls='-',
                            color=_color,
                            marker=_marker,
                            markevery=0.05,
                            alpha=0.7,
                            label=my_case)
            ax_overp_y.plot(y_tip[:length], (df.overpressure_mbar[:length]),
                            ls='-',
                            color=_color,
                            marker=_marker,
                            markevery=0.05,
                            alpha=0.7,
                            label=my_case)

        if PLOT_FLAME_SPEED:

            # if track condition is available
            try:
                df = data[my_case]["avbp_track_condition-temperature_gt_800_lt_1200"]
                ax_st_pos_x.plot(df['x_max_rolling'], df['x_max_speed'],
                                 ls=linestyle,
                                 color=_color,
                                 marker=_marker,
                                 #    markevery=0.05,
                                 label=my_case)
                ax_st_pos_y.plot(df['y_max_rolling'], df['y_max_speed'],
                                 ls=linestyle,
                                 color=_color,
                                 marker=_marker,
                                 #    markevery=0.05,
                                 label=my_case)
                ax_st_time.plot(df['t'], df['y_max_speed'],
                                ls=linestyle,
                                color=_color,
                                marker=_marker,
                                #    markevery=0.05,
                                label=my_case)

                # _freq, _spec = get_fft(df, 'y_max_speed', sig_length=len(df['t'])/1)
                # ax_st_fft.semilogy(_freq, _spec,
                #                ls=linestyle,
                #                color=_color,
                #                marker=_marker,
                #                #    markevery=0.05,
                #                label=my_case)

            # cases from Omar
            except KeyError:
                df = data[my_case]["avbp_mmm"]
                ax_st_pos_x.plot(df['Pos_max_rolling'], SL0_REF * np.ones_like(df['Pos_max_speed'].values),
                                 ls='--',
                                 lw='1.25',
                                 color="xkcd:light grey",
                                 marker=_marker,
                                 label=r'$S_L^0$')
                ax_st_pos_x.plot(df['Pos_max_rolling'], df['Pos_max_speed'],
                                 ls=linestyle,
                                 color=_color,
                                 marker=_marker,
                                 #    markevery=0.05,
                                 label=my_case)
                ax_st_pos_y.plot(df['Pos_y_max_rolling'], SL0_REF * np.ones_like(df['Pos_y_max_speed'].values),
                                 ls='--',
                                 lw='1.25',
                                 color="xkcd:light gray",
                                 marker=_marker,
                                 label=r'$S_L^0$')
                ax_st_pos_y.plot(df['Pos_y_max_rolling'], df['Pos_y_max_speed'],
                                 ls=linestyle,
                                 color=_color,
                                 marker=_marker,
                                 #    markevery=0.05,
                                 label=my_case)
                ax_st_time.plot(df['t'], SL0_REF * np.ones_like(df['Pos_y_max_speed'].values),
                                ls='--',
                                lw='1.25',
                                color="xkcd:blood",
                                marker=_marker,
                                label=r'$S_L^0$')
                ax_st_time.plot(df['t'], df['Pos_y_max_speed'],
                                ls=linestyle,
                                color=_color,
                                marker=_marker,
                                #    markevery=0.05,
                                label=my_case)

                # # _freq, _spec = get_fft(df, 'Pos_y_max_speed')
                # _freq, _spec = get_fft(df, 'Pos_y_max_speed', sig_length=len(df['t'])/1)
                # ax_st_fft.plot(_freq, _spec,
                #                ls=linestyle,
                #                color=_color,
                #                marker=_marker,
                #                #    markevery=0.05,
                #                label=my_case)

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

            # Avoid plotting garbage at the end of the simualtion
            # x_tip = x_tip[x_tip < X_MAX]
            # y_tip = y_tip[y_tip < Y_MAX]

            my_label_res_hr = None
            my_label_sgs_hr = None
            my_label_tot_hr = None

            # if currently running the 2 dataframe may not have the same length
            length = min(len(y_tip), len(df.HR_mean))

            # Debug
            for col in df_mmm.columns.values:
                if 'HR' in col:
                    logger.info(col)

            if not idx:
                my_label_res_hr = r"$\dot{\omega}_{\rm res}$"
                my_label_sgs_hr = r"$\dot{\omega}_{\rm sgs}$"
                my_label_tot_hr = r"$\dot{\omega}_{\rm tot}$"

            ax_hr_x.plot(x_tip[:length], df_mmm.HR_mean[:length], '-',
                         color=_color,
                         marker=_marker,
                         markevery=0.05,
                         label=my_label_tot_hr)

            ax_perc_hr_x.plot(x_tip[:length], df_mmm.percentage_res_HR[:length],
                              color=_color,
                              marker=_marker,
                              markevery=0.05,
                              label=my_case)

            ax_hr_y.plot(y_tip[:length], df_mmm.HR_mean[:length], '-',
                         color=_color,
                         marker=_marker,
                         markevery=0.05,
                         label=my_label_tot_hr)

            ax_perc_hr_y.plot(y_tip[:length], df_mmm.percentage_res_HR[:length],
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
        my_str = my_str.replace("(", "")
        my_str = my_str.replace(")", "")
        my_str = my_str.replace(r"\rm", "")
        my_str = my_str.replace("cells", "C")
        my_str = my_str.replace("_", "")

    logger.info(my_str)

    if PLOT_OVERPRESSURE:
        # Save overpressure figure
        ax_overp.legend(ncol=1)
        ax_overp.set_xlim(left=0)
        # ax_overp.set_xlim(right=0.4)
        fig_overP.tight_layout()

        ax_overp_x.legend()
        # ax_overp_x.set_xlim(0, X_MAX)
        # ax_overp_y.set_xlim(0, Y_MAX)
        fig_overP_vs_xy.tight_layout()

        # Save overpressure figure
        for _ext in EXT_LIST:
            fig_overP_vs_xy.savefig("Figures/Figure_overP_xy_.%s.%s" % (my_str, _ext),
                                    bbox_inches='tight',
                                    transparent=False,
                                    pad_inches=0.02)

            fig_overP.savefig("Figures/Figure_overP_.%s.%s" % (my_str, _ext),
                              bbox_inches='tight',
                              transparent=False,
                              pad_inches=0.02)

    if PLOT_FLAME_SPEED:
        # Save flame
        fig_speed_vs_pos.tight_layout()
        ax_st_pos_y.legend(ncol=1)
        ax_st_pos_y.set_xlim(left=0, right=Y_MAX)
        ax_st_pos_x.set_xlim(left=0, right=X_MAX)
        ax_st_pos_x.set_ylim(ymin=0)
        ax_st_pos_y.set_ylim(ymin=0)
        for _ext in EXT_LIST:
            fig_speed_vs_pos.savefig(r"Figures/Figure_FlameSpeed.%s.%s" % (my_str, _ext),
                                     bbox_inches='tight',
                                     pad_inches=0.02)
        fig_speed_vs_time.tight_layout()
        ax_st_time.legend(ncol=1)
        ax_st_time.set_xlim(left=0)
        # ax_st_time.set_ylim(ymin=0)
        for _ext in EXT_LIST:
            fig_speed_vs_time.savefig(r"Figures/Figure_FlameSpeed_vs_time.%s.%s" % (my_str, _ext),
                                      bbox_inches='tight',
                                      pad_inches=0.02)
        # fig_speed_fft.tight_layout()
        # ax_st_fft.legend(ncol=1)
        # ax_st_fft.set_xlim(left=0, right=200)
        # ax_st_fft.set_ylim(ymin=1e-6, ymax=1)
        # for _ext in EXT_LIST:
        #     fig_speed_fft.savefig(r"Figures/Figure_FlameSpeed_vs_freq.%s.%s" % (my_str, _ext),
        #                              bbox_inches='tight',
        #                              pad_inches=0.02)

    if PLOT_FLAME_POSITION:
        ax_flame_pos.set_xlim(left=0)
        # ax_flame_pos.set_ylim(bottom=0)
        ax_flame_pos.legend()
        for _ext in EXT_LIST:
            fig_flame_pos.savefig("Figures/Figure_FlamePosition.%s.%s" % (my_str, _ext),
                                  bbox_inches='tight',
                                  pad_inches=0.02)

    if PLOT_F_MEAN_FLAME:
        ax_thick.set_xlim(left=0)
        ax_thick.set_ylim(bottom=1)
        ax_thick.legend()
        for _ext in EXT_LIST:
            fig_thick.savefig("Figures/Figure_thickening_masked.%s.%s" % (my_str, _ext),
                              bbox_inches='tight',
                              pad_inches=0.02)

    if PLOT_RESOLVED_HR:
        ax_hr_x.set_ylim(bottom=0, top=5e5)
        ax_hr_x.set_xlim(left=0, right=X_MAX)
        ax_hr_x.legend()
        ax_hr_y.set_ylim(bottom=0, top=5e5)
        ax_hr_y.set_xlim(left=0, right=Y_MAX)
        ax_perc_hr_x.legend()
        for ext in EXT_LIST:
            fig_res_hr.savefig("Figures/Figure_ResolvedHR.%s.%s" % (my_str, ext),
                               bbox_inches='tight',
                               pad_inches=0.02)

    if SHOW:
        plt.show()
