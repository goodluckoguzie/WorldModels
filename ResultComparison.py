import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
file_paths = ["RESULTS/RVO2results.csv",
              "RESULTS/duelingDQNresults.csv",
              "RESULTS/SFMresults.csv",
              "RESULTS/WMduelingDQNresults.csv",
              "RESULTS/MASPMduelingDQNAveragesresults.csv",
              "RESULTS/ADTSPHSMduelingDQNresults.csv"]

# Define method names
methods = ["RVO2", "duelingDQN", "SFM", "WMduelingDQN", "MASPMduelingDQNAverages", "ADTSPHSMduelingDQN"]

# Load each csv file into a pandas DataFrame
data = []
for method, file_path in zip(methods, file_paths):
    df = pd.read_csv(file_path)
    df["method"] = method
    data.append(df)

# Concatenate all DataFrames into one DataFrame
df_all = pd.concat(data)

# Define metrics
metrics = ['Discomfort Counts', 'Jerk Counts', 'Velocities', 'Path Lengths', 
           'Times', 'Out of Maps', 'Human Collisions', 
           'Max Steps', 'Ruccessive Run', 'Episode Reward', 
           'Idle Times', 'Personal Space Compliances']

for metric in metrics:
    # Create a figure for each metric
    plt.figure(figsize=(10, 6))

    # Plot histogram
    sns.histplot(data=df_all, x=metric, hue="method", element="step", stat="density", common_norm=False)

    # Set title
    plt.title(f'{metric} Comparison Across Methods', fontsize=15)

    # Adjust x-axis labels
    plt.xticks(rotation=45)  

    # Show the plot
    plt.show()
