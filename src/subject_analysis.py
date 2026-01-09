"""
Analyses json subject files (derived from INSPIRE dataset).
Must first run inspire_dataset_subjects to separate into individual subjects.
"""
import sys
import pandas as pd
import os
from stopwatch import Stopwatch
from sortedcontainers import SortedDict

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

from subject import Subject
import vitals as vitals_module
import parameters
import medications as meds_module

import subject as subject_module


def analyse_subject( subject, vital_item_names ):
    """
    Displays normalised vitals (currently only operation vitals, not ward vitals).
    :param subject:
    :param vital_item_names:
    :return:
    """
    subject_vitals = subject.get_vitals()

    df = pd.DataFrame(subject_vitals)

    # # Check if the 'score' column is of numeric type, returning a boolean series
    # is_numeric = df["value"].str.isnumeric()
    # # Use boolean indexing to select rows that meet the condition
    print(f"Filtering to get only numeric values, shape {df.shape}")
    # df = df[is_numeric]
    # Convert the 'salary' column to numeric type; set non-numeric values to NaN
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["chart_time"] = pd.to_numeric(df["chart_time"], errors="coerce")
    # Drop rows containing NaN in the 'salary' column
    df.dropna(subset=["value"], inplace=True)

    # Sort by chart_time, seaborn will not do this automatically
    print(f"Sorting dataframe by chart_time, shape {df.shape}")
    # df["chart_time"] = pd.to_numeric(df["chart_time"], errors="coerce")
    df = df.sort_values(by='chart_time')
    print(f"After sorting dataframe by chart_time, shape {df.shape}")

    # Standardlise each vital, only this subjects vitals, should use all data for mean/stdev?
    for vital_item_name in vital_item_names:
        if df['item_name'].isin([vital_item_name]).any():
            # vitals_module.standardize_vital(df, vital_item_name)
            vital_df = df[df['item_name'] == vital_item_names]
            mean = vital_df['value'].mean()
            std = vital_df['value'].std()
            df.loc[df['item_name'] == vital_item_names, 'value'] = (vital_df['value'] - mean) / std
        #else:
        #    print(f"Dataframe does not have vital {vital_item_name}, standardisation skipped")

    # Sort the DataFrame by index
    df = df.sort_index()
    df = df.sort_values(by=['op_id', 'chart_time'], ascending=[True, True])
    # I have found the index to be unreliable, same op_id and chart_time but non-consecutive index
    # Also, op_id does not seem to be sorted by chart_time,
    # i.e. larger op_id occurs earlier according to chart_time
    #for index, row in df.iterrows():
    #    print(f"op_id {row['op_id']}, subject_id {row['subject_id']}, chart_time {row['chart_time']}, item_name {row['item_name']}")

    #               op_id  subject_id  chart_time item_name  value
    # 0         435959808   181409183        1985    minvol    4.4
    # 1         435959808   181409183        1985        vt  512.0
    # Plotting the vitals
    # sns.lineplot(x='chart_time', y='value', hue='item_name', data=df)
    sns.scatterplot(x='chart_time', y='value', hue='item_name', data=df)

    operations = subject.get_operations()
    # Write indicator when operation in/out of room and operation starts/stops
    for operation in operations:
        orin_time = int(operation['orin_time'].strip())
        orout_time = int(operation['orout_time'].strip())
        opstart_time = int(operation['opstart_time'].strip())
        opend_time = int(operation['opend_time'].strip())

        plt.axvline(x=orin_time, color='orange')
        plt.axvline(x=orout_time, color='orange')
        plt.axvline(x=opstart_time, color='cyan')
        plt.axvline(x=opend_time, color='cyan')

    # Customizing the plot
    plt.title('Vitals Over Time')
    plt.xlabel('Chart Time')
    plt.ylabel('Value')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # Display the plot
    plt.tight_layout()
    plt.show()




def plot_vitals( subject ):
    operation_vitals = subject.get_vitals()
    ward_vitals = subject.get_ward_vitals()

    # Get lookup for medications (i.e. atc_code)
    atc_code_lookup = meds_module.read_atc_codes()

    """
    CSV File ./dataset/vitals.csv (aka operation vitals)
           op_id subject_id chart_time item_name value
    0  475179926  100033460        700        bt  22.0
    1  475179926  100033460        700   art_mbp   6.0
    2  475179926  100033460        700       cvp   0.0
    3  475179926  100033460        705        bt  22.0
    4  475179926  100033460        710        bt  22.0

    CSV File ./dataset/ward_vitals.csv
             subject_id chart_time item_name  value
    0         104192463        580      spo2   98.0
    1         104192463        580  nibp_sbp  169.0
    2         104192463        580        hr   96.0
    3         104192463        580  nibp_dbp  100.0
    4         104192463        580        bt   36.4
    """
    # df = pd.DataFrame(subject_vitals)
    # print(df.head())
    # Sort by chart_time, seaborn will not do this automatically
    # print(f"Sorting dataframe by chart_time, shape {df.shape}")
    # df["chart_time"] = pd.to_numeric(df["chart_time"], errors="coerce")
    # df = df.sort_values(by='chart_time')

    # heart_features = [
    #     'nibp_dbp',
    #     'nibp_mbp', # not sure mean is useful
    #     'nibp_sbp',
    #     'pap_dbp',
    #     'pap_mbp', # not sure mean is useful
    #     'pap_sbp',
    #     'hr',
    # ]

    # Get parameters to provide user-friendly title for plot
    label2desc, label2unit = parameters.read_parameters()

    operation_series = subject_module.convert_vitals_to_dictionary(operation_vitals)
    ward_series      = subject_module.convert_vitals_to_dictionary(ward_vitals)

    item_names = set()
    for item_name in operation_series.keys():
        item_names.add(item_name)
    for item_name in ward_series.keys():
        item_names.add(item_name)

    first_operation = subject.get_first_operation()

    # Get time subject died (if any, most survive)
    inhosp_death_time = first_operation['inhosp_death_time'].strip()
    if len(inhosp_death_time) > 0:
        inhosp_death_time = int( inhosp_death_time )
    else:
        inhosp_death_time = None

    medications = subject.get_medications()
    labs = subject.get_labs()
    operations = subject.get_operations()
    print(f"Found {len(operations)} operations for subject {subject.get_subject_id()}")
    print(f"Found {len(labs)} labs for subject {subject.get_subject_id()}")
    print(f"Found {len(medications)} medications for subject {subject.get_subject_id()}")

    print(f"Found {len(ward_series)} series {ward_series.keys()}")

    for item_name in item_names:

        fig, (axl, axv, axm) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        if item_name in ward_series:
            item_series = ward_series[item_name]
            # Convert to DataFrame with 'item_name' and 'value' columns
            df = pd.DataFrame(item_series.items(), columns=['chart_time', 'value'])
            sns.scatterplot(x='chart_time', y='value', data=df, label='Ward Vitals', color='blue', ax=axv)

        if item_name in operation_series:
            item_series = operation_series[item_name]
            # Convert to DataFrame with 'item_name' and 'value' columns
            df = pd.DataFrame(item_series.items(), columns=['chart_time', 'value'])
            sns.scatterplot(x='chart_time', y='value', data=df, label='Operation Vitals', color='orange', ax=axv)


        # Denote time subject died
        if inhosp_death_time is not None:
            axv.axvline(x=inhosp_death_time, color='red')


        # Write medications indicators
        y_offset = dict() # Many value overlap at same chart_time, "declutter"
        for medication in medications:
            # subject_id chart_time drug_name route drug_name2 drug_name3 atc_code atc_code2 atc_code3
            chart_time = int( medication['chart_time'] )
            drug_name  = medication['drug_name']
            route      = medication['route']
            atc_code   = medication['atc_code'] # also drug name/atc_code 2 and 3
            # atc_desc   = meds_module.parse_atc_code(atc_code, atc_code_lookup)
            if chart_time not in y_offset:
                y_offset[chart_time] = 0
            else:
                y_offset[chart_time]+=0.14
            axm.text(x=chart_time, y=0.0 + y_offset[chart_time], s=f'{drug_name} ({route})', rotation=30)
            axm.axvline(x=chart_time, color='gray')

        # Write labs indicators
        # subject_id chart_time item_name value
        y_offset = dict()  # Many value overlap at same chart_time, "declutter"
        for lab in labs:
            # subject_id chart_time drug_name route drug_name2 drug_name3 atc_code atc_code2 atc_code3
            chart_time = int(lab['chart_time'])
            lab_name = lab['item_name']
            value = lab['value']
            if chart_time not in y_offset:
                y_offset[chart_time] = 0
            else:
                #-----------------------------------------#
                # This spacing has huge effect on layout of plots
                # too big and does not expand to figure area
                # smaller seems to cause less issues
                # -----------------------------------------#
                y_offset[chart_time] += 0.05 # 0.14
            axl.text(x=chart_time, y=0.0 + y_offset[chart_time], s=f'{lab_name} ({value})', rotation=30)
            axl.axvline(x=chart_time, color='gray')



        # Write indicator when operation in/out of room and operation starts/stops
        for operation in operations:
            orin_time = int(operation['orin_time'].strip())
            orout_time = int(operation['orout_time'].strip())
            opstart_time = int(operation['opstart_time'].strip())
            opend_time = int(operation['opend_time'].strip())

            axv.axvline(x=orin_time, color='orange')
            axv.axvline(x=orout_time, color='orange')
            axv.axvline(x=opstart_time, color='cyan')
            axv.axvline(x=opend_time, color='cyan')

            # also show operation times on medication plot
            axm.axvline(x=orin_time, color='orange')
            axm.axvline(x=orout_time, color='orange')
            axm.axvline(x=opstart_time, color='cyan')
            axm.axvline(x=opend_time, color='cyan')

            # also show operation times on labs plot
            axl.axvline(x=orin_time, color='orange')
            axl.axvline(x=orout_time, color='orange')
            axl.axvline(x=opstart_time, color='cyan')
            axl.axvline(x=opend_time, color='cyan')


        # Customizing the plot
        desc = item_name # some abbreviations are not in parameters, e.g. "cpat"
        if item_name in label2desc:
            desc = label2desc[item_name]
        axv.set_title(f'Item {item_name} - {desc}')
        axv.set_xlabel('Chart Time')
        axv.set_ylabel('Value')

        axm.set_xlabel('Chart Time')
        axm.set_ylabel('Medication')

        axl.set_xlabel('Chart Time')
        axl.set_ylabel('Labs')

        # subplots sharex defaults to only showing x labels for bottom plot
        # have to explicitly set if you want for other subplots
        axv.tick_params(labelbottom=True)
        axv.tick_params(axis='x', labelrotation=45)

        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

        # Display the plot
        plt.tight_layout()
        plt.show()
        # plt.clf()  # Clears the figure for the next iteration


def plot_series_dashboard(series, title='Interactive Time Series Viewer'):
    """
    Plots the values allowing to slect which label is shown
    :param series: dict[label] -> dict[chart_time]->value
    :return:
    """

    # series = {
    #     'nepi': {730: 0.1, 735: 0.3, 755: 0.3, 765: 0.3, 780: 0.3},
    #     'phe': {745: 100.0, 750: 100.0, 755: 100.0, 760: 100.0, 765: 100.0, 780: 100.0},
    #     'ebl': {770: 150.0, 785: 150.0},
    #     'ns': {810: 550.0},
    #     'psa': {810: 800.0},
    # }

    n_series = len(series)

    # ----- Step 2: Set up the Plot -----
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.25)  # Leave space for checkboxes

    # Plot all lines and store references
    lines = []
    labels = list(series.keys())

    for label in labels:
        chart_times = list()
        vital_series = series[label]
        # print(f"{label} -> {vital_series}")
        for chart_time in vital_series.keys():
            chart_times.append(chart_time)
        values = list(series[label].values())
        # line, _ = ax.plot(chart_times, values, label=label, linewidth=1)
        line = ax.scatter(x=chart_times, y=values, label=label)
        # line = sns.scatterplot(x=chart_times, y=values, label=label)
        lines.append(line)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    # ----- Step 3: Create CheckButtons Widget -----
    # Position: [left, bottom, width, height]
    rax = plt.axes([0.01, 0.1, 0.2, 0.8])
    visibility = [False] * n_series  # Start with all hidden
    check = CheckButtons(rax, labels, visibility)

    # ----- Step 4: Define Callback for Interactivity -----
    def toggle_visibility(label):
        index = labels.index(label)
        lines[index].set_visible(not lines[index].get_visible())
        plt.draw()

    check.on_clicked(toggle_visibility)

    # Hide all lines initially
    for line in lines:
        line.set_visible(False)

    plt.show()



def test_main():
    parent_dir = '../inspire_subjects'
    # subjects = read_all_subjects(parent_dir)

    stopwatch = Stopwatch()
    subjects = subject_module.read_subjects(parent_dir)
    print(f"Reading took {stopwatch.elapsedTime() / 60:.2f} minutes")

    more_than_one_operation = 0
    died = 0
    survived = 0
    for subject_id in subjects.keys():
        subject = subjects[subject_id]
        operations = subject.get_operations()
        if len(operations) > 1:
            more_than_one_operation += 1
        if subject.died():
            died += 1
        else:
            survived += 1
    print(f"{more_than_one_operation} subjects had more than one operation")
    print(f"Subjects {died} died")
    print(f"Subjects {survived} survived")

"""
Reads all packets from pcap into memory.
"""
def main():

    # if len(sys.argv) != 2:
    #     print("Usage <filepath for subject json>")
    #     print("Example ../inspire_subjects/126736540.json")
    #     return
    # filename = sys.argv[1]

    # filepath = "../inspire_subjects/died/100033460.json"
    # filepath = "../inspire_subjects/died/100221250.json"
    # filepath = "../inspire_subjects/died/126736540.json"
    # filepath = "../inspire_subjects/died/140696052.json"
    # filepath = '../inspire_subjects/died/126736540.json'

    # No chart_times near last operations???
    # filepath = '../inspire_subjects/survived/100002413.json'
    filepath = '../inspire_subjects/survived/100004062.json'


    # Two values for preop_creatinine at same chart time
    # filepath = '../inspire_subjects/survived/121782671.json'

    # Key preop_glucose not equal 264.0 != 225.0
    # Equal 156112884: False

    # Two values for Key preop_albumin not equal 3.7 != 3.5
    # filepath = '../inspire_subjects/survived/138538862.json'

    # Two values for preop_glucose, one nan and other 132.0
    # filepath = '../inspire_subjects/survived/195061453.json'

    # Mismatching bmi, Key bmi not equal 0.002081165452653486 != 20.811654526534856
    # 'weight': '50', 'height': '155'
    # filepath = '../inspire_subjects/died/140696052.json'

    # No time series?
    #filepath = '../inspire_subjects/survived/100002413.json'


    print(f"Reading subject from file {filepath}")
    subject = Subject(None)
    subject.fromJSON(filepath)
    subject_id = subject.get_subject_id()

    print(f"OPERATION: number {len(subject.get_operations())}")

    # operation = subject.get_first_operation()
    # print(f"OPERATION: {operation}")

    vital_item_names = vitals_module.get_vital_names()

    operation_vitals = subject.get_vitals()
    ward_vitals = subject.get_ward_vitals()

    operation_series = subject_module.convert_vitals_to_dictionary(operation_vitals)
    ward_series = subject_module.convert_vitals_to_dictionary(ward_vitals)

    # plot_series_dashboard(operation_series, title=f'{subject_id} Operation Vitals')
    # plot_series_dashboard(ward_series,      title=f'{subject_id} Ward Vitals')

    analyse_subject(subject, vital_item_names)
    plot_vitals(subject)


if __name__ == "__main__":
    main()
    # test_main()
