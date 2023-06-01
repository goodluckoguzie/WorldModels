import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Number of bins for non-binary metrics
number_non_binary_bin = 10

# Define file paths for the first comparison
file_paths1 = {
    "MASPMduelingDQNAverages": "RESULTS/MASPMduelingDQNresults.csv",
    "ADTSPHSMduelingDQN": "RESULTS/ADTSPHSMduelingDQNresults.csv",
    "DuelingDQNTwoStepAhead": "RESULTS/duelingDQNTwoStepAheadresults.csv",
    "WMduelingDQN": "RESULTS/WMduelingDQNresults.csv",

}

# Load the data
data1 = {method: pd.read_csv(file_path) for method, file_path in file_paths1.items()}

# Define file paths for the second comparison
file_paths2 = {
    "RVO2": "RESULTS/RVO2results.csv",
    "duelingDQN": "RESULTS/duelingDQNresults.csv",
    "SFM": "RESULTS/SFMresults.csv",
    "WMduelingDQN": "RESULTS/WMduelingDQNresults.csv",
    "ADTSPHSMduelingDQN": "RESULTS/ADTSPHSMduelingDQNresults.csv",
}


# Print the reward values for the WMduelingDQN
print(data1['WMduelingDQN']['Reward'])


# Load the data
data2 = {method: pd.read_csv(file_path) for method, file_path in file_paths2.items()}

# List of data for the comparisons
data_list = [data1, data2]

# Define metrics
metrics = ['Human Discomfort', 'Jerk Counts', 'Velocities', 'Distance Traveled', 
           'Simulation Time', 'Wall Collisions', 'Human Collisions', 
           'Max Steps', 'Reward', 'Successful Run',
           'Idle Time', 'Personal Space Compliances Rate']

# Define binary and non-binary metrics
binary_metrics = ['Wall Collisions','Successful Run','Max Steps','Human Collisions']
non_binary_metrics = [ 'Jerk Counts','Idle Time', 'Velocities', 'Human Discomfort',  'Distance Traveled', 'Simulation Time', 'Reward', 'Personal Space Compliances Rate']

# Title of the figures
titles = ['Comparison of MASPMduelingDQNAverages, ADTSPHSMduelingDQN and DuelingDQNTwoStepAhead',
          'Comparison of RVO2, duelingDQN, SFM, WMduelingDQN and ADTSPHSMduelingDQN',
          'Comparison of all methods']

# Define units for metrics
units = {'Human Discomfort': 'Count',
         'Jerk Counts': 'Count',
         'Velocities': 'Meters per second',
         'Distance Traveled': 'Meters',
         'Simulation Time': 'Seconds',
         'Wall Collisions': 'Count',
         'Human Collisions': 'Count',
         'Max Steps': 'Count',
         'Reward': 'Score',
         'Successful Run': 'Count',
         'Idle Time': 'Seconds',
         'Personal Space Compliances Rate': 'Compliance Rate'}

# Define maximum values for certain metrics
max_values = {'Human Discomfort': 90,
              'Velocities': 0.1,
              'Distance Traveled': 80,
              'Idle Time': 5,
              'Jerk Counts': 8,
            #   'Reward': 1,
              'Simulation Time': 200,
              }

for data, title in zip(data_list, titles):
    fig, axs = plt.subplots(3, 4, figsize=(20, 7))
    plt.suptitle(f'Histograms for Episode Metrics - {title}', fontsize=10)

    handles = []  # List to store the handles for the legend
    labels = list(data.keys())  # List to store the labels for the legend

    # Iterate over each metric and plot on each subplot
    for i, metric in enumerate(metrics):
        ax = axs[i//4, i%4]  # Determine the subplot for the current metric

        max_value = max_values.get(metric, None)  # Get the max value for the current metric, if one is defined

        for j, (method, df) in enumerate(data.items()):  # Iterate over each dataset
            if max_value:
                df = df[df[metric] <= max_value]  # Filter out values greater than the max value for the current metric
            bins = np.histogram(df[metric], bins=number_non_binary_bin, range=(0, max_value) if max_value else None)[1]
            counts, _ = np.histogram(df[metric], bins=bins)
            column_width = ((bins[1]-bins[0])/(len(data)+1))
            method_offset = (j+0.5) * (bins[1]-bins[0])/(len(data)+1)
            handle = ax.bar(bins[:-1] + method_offset, counts, width=column_width, alpha=1., label=method)
            if metric in binary_metrics:
                ax.set_xticks([0, 1])

            if i == 0:  # We only need to add handles once, so we do it on the first iteration
                handles.append(handle)

        ax.set_title(f'{metric.capitalize()}' ) #Metrics')
        ax.set_xlabel(units[metric])  # Set x-axis label to be the units of the metric
        ax.set_yticks([0, max(counts)])  # Set yticks to have only two values, 0 and max of the counts

    # Create a single legend for the figure with the collected handles and labels
    fig.legend(handles, labels, loc='lower center', ncol=len(data), fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.08)  # Adjust the top and bottom of the subplots so the suptitle and legend do not overlap
    plt.show()
