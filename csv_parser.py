#!/usr/bin/env python
import os
import sys
import glob
import time
import datetime
import fire
import pandas as pd

def find_csv_files(workdir):
    return glob.glob(os.path.join(workdir, "*.csv"))

def parse_csv(path):
    csv_data = pd.read_csv(path)
    return csv_data

def filter_csv(csv_data, columns=[]):
    return csv_data

def date_to_UNIX(date_series):
    result = []
    for date in date_series:
        print(date)
        try:
            date = int(time.mktime(datetime.datetime.strptime(date, '%Y-%m-%d').timetuple()))
        except:
            date = 0
        result.append(date)
    return pd.Series(result)

class Parser(object):
    """A simple csv parser class."""

    def parse(self, path=None, columns=None, date_columns=None, output_path=None):
        if path == None:
            # find csv files
            workdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'datasets', 'csv')
            pathes = find_csv_files(workdir)

            # Select csv file index
            for i in range(len(pathes)):
                print(i + 1, pathes[i])
            path_idx = int(input("Select number>")) -1
            csv_path = pathes[path_idx]

        # Read csv
        csv_data = pd.read_csv(os.path.abspath(csv_path))

        # Filter columns
        if columns == None:
            # Select filter csv columns
            for i in range(len(csv_data.columns)):
                print(i + 1, csv_data.columns[i])
            columns = [csv_data.columns[int(n) - 1] for n in input("Select collect column number>").split(',')]
        
        filtered_csv_data = csv_data[columns]

        # Parse date to unix time
        if date_columns == None:
            for i in range(len(filtered_csv_data.columns)):
                print(i + 1, filtered_csv_data.columns[i])
            date_columns = [filtered_csv_data.columns[int(n) - 1] for n in input("Select Date column number>").split(',')]
        print(date_columns)
        date_column_numbers = [list(filtered_csv_data.columns).index(column) for column in date_columns]
        print(date_column_numbers)
        for column_number in date_column_numbers:
            filterd_csv_data = filtered_csv_data.apply(date_to_UNIX, axis=column_number)
        
        # Save CSV Data
        if output_path == None:
            output_path = input("Enter output path>")
        filtered_csv_data.to_csv(os.path.abspath(output_path), header=None, index=None)

        print('finish !!')

        return 0
     
if __name__ == '__main__':
    fire.Fire(Parser)
