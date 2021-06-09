import os
import glob

import numpy as np
import pandas as pd


def join_subdirectory_csv_files(prefix, extension):
    """
    1. Seek for csv files according to prefix.extension rule
    2. concatenate all files
    3. drop duplicates
    4. re-index
    5. dump clean concatenated file
    """
    # Find all csv files in subdirectories
    all_filenames = [_file for _file in sorted(glob.glob('*/{}.{}'.format(prefix, extension)))]

    # combine all files in the list
    # combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])

    combined_csv =  pd.read_csv(all_filenames[0])
    for _idx, _file in enumerate(all_filenames):
        if _idx:
            print("\t > %s" % _file)
            _df = pd.read_csv(_file)
            # combined_csv.merge(_df, how="inner")
            combined_csv = pd.merge_ordered(combined_csv, _df, fill_method="ffill")

    # Drop duplicates
    combined_csv = combined_csv.drop_duplicates().reset_index(drop=True)

    # export to csv
    combined_csv.to_csv("%s.csv" % prefix, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    # Join all csv files needed here
    extension = "csv"
    prefixes = ["avbp_local_probe_0", "avbp_mmm", "avbp_venting"]

    for prefix in prefixes:
        print(" > Joining %s.%s" % (prefix, extension))
        join_subdirectory_csv_files(prefix, extension)
