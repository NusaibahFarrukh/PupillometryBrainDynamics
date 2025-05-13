# %% Load the libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the dataset base directory
base_dir = "F:\\Dataset\\"

# List of participants
participants = [f"sub-{str(i).zfill(3)}" for i in range(32, 99)]  # sub-032 to sub-099

# Event labels to filter
list_needed = [500105, 500109, 500113, 6001051, 6001050, 6001091, 6001090, 6001131, 6001130]

# Directory to save plots
output_dir = "D:\\QC_Plots"
os.makedirs(output_dir, exist_ok=True)

# Iterate over all participants
for participant in participants:
    try:
        print(f"Processing {participant}...")

        # Load pupil data
        pupil_file = os.path.join(base_dir, participant, "pupil", f"{participant}_task-memory_pupil.tsv")
        data = pd.read_csv(pupil_file, sep='\t')

        # Load event markers
        marker_file = os.path.join(base_dir, participant, "pupil", f"{participant}_task-memory_events.tsv")
        markers = pd.read_csv(marker_file, sep='\t')

        # Select only the required markers
        markers = markers[markers['label'].isin(list_needed)].reset_index()

        # Match with the closest timestamps in the pupil data
        nearest_stamps_list = []
        for marker_id in range(len(markers)):
            closest_timestamp = data.iloc[(data["pupil_timestamp"] - markers['timestamp'][marker_id]).abs().idxmin()]['pupil_timestamp']
            nearest_stamps_list.append(closest_timestamp)

        # Add new timestamps
        markers['Closest_timestamp'] = nearest_stamps_list

        # QC check
        markers["diff"] = abs(markers["Closest_timestamp"] - markers["timestamp"])

        print(f"{participant} - Total Expected Markers: {len(markers)}")
        print(f"{participant} - Markers Successfully Mapped: {markers['Closest_timestamp'].notna().sum()}")

        # Generate and save plot
        plt.figure(figsize=(8, 6))
        plt.plot(markers["timestamp"], markers["diff"], marker='o', linestyle='-', color='purple', alpha=0.5, linewidth=0.8)
        plt.xlabel("Original Event Timestamp")
        plt.ylabel("Difference with Closest Pupil Timestamp")
        plt.title(f"Difference Between Event and Mapped Timestamps\n{participant}")
        plt.axhline(y=0, color='r', linestyle='--', label="Ideal (No Difference)")
        plt.legend()
        
        # Save the figure
        plot_path = os.path.join(output_dir, f"{participant}_QC_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{participant} - QC plot saved successfully.")

    except Exception as e:
        print(f"Error processing {participant}: {e}")

print("Processing completed for all participants.")



