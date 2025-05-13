# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 12:51:55 2025

@author: Nusaibah Farrukh
"""
import pandas as pd
import mne
import numpy as np
import os
import pickle

# Define subject IDs
subject_ids = [f"sub-{i:03d}" for i in range(32, 34)]  
data_dir = "D:\\Dataset\\"  

# Define event dictionaries
event_dict_five = {
    "justlisten/five": 500105,
    "memory/correct/five": 6001051,
    "memory/incorrect/five": 6001050
}

event_dict_nine = {
    "justlisten/nine": 500109,
    "memory/correct/nine": 6001091,
    "memory/incorrect/nine": 6001090
}

event_dict_thirteen = {
    "justlisten/thirteen": 500113,
    "memory/correct/thirteen": 6001131,
    "memory/incorrect/thirteen": 6001130,
}

# Define sample rate
srate = 120

# Dictionary to store epochs for all subjects
all_epochs = {}

from collections import defaultdict

drop_log = defaultdict(lambda: {
    "five_left": {"total": 0, "dropped": 0},
    "nine_left": {"total": 0, "dropped": 0},
    "thirteen_left": {"total": 0, "dropped": 0},
    "five_right": {"total": 0, "dropped": 0},
    "nine_right": {"total": 0, "dropped": 0},
    "thirteen_right": {"total": 0, "dropped": 0}
})

#%%
# Loop through subjects
for subject_id in subject_ids:
    print(f"Processing {subject_id}...")
    subject_dir = os.path.join(data_dir, subject_id, "pupil")
    pupil_file = os.path.join(subject_dir, f"{subject_id}_task-memory_pupil.tsv")
    events_file = os.path.join(subject_dir, f"{subject_id}_task-memory_events.tsv")

    # Check if files exist
    if not os.path.exists(pupil_file) or not os.path.exists(events_file):
        print(f"Files not found for {subject_id}. Skipping.")
        continue

    try:
        data = pd.read_csv(pupil_file, sep='\t')
        markers = pd.read_csv(events_file, sep='\t')

        # Filter only available markers
        list_needed = [500105, 500109, 500113, 6001051, 6001050, 6001091, 6001090, 6001131, 6001130]
        markers = markers[markers['label'].isin(list_needed)]

        if markers.empty:
            print(f"No required markers found for {subject_id}. Skipping.")
            continue

        markers = markers.reset_index(drop=True)
        available_marker_ids = markers['label'].unique()

        event_dict_five = {k: v for k, v in event_dict_five.items() if v in available_marker_ids}
        event_dict_nine = {k: v for k, v in event_dict_nine.items() if v in available_marker_ids}
        event_dict_thirteen = {k: v for k, v in event_dict_thirteen.items() if v in available_marker_ids}

        # Match timestamps
        nearest_stamps_list = []
        for marker_id in range(len(markers)):
            closest_timestamp = data.iloc[(data["pupil_timestamp"] - markers['timestamp'][marker_id]).abs().idxmin()]['pupil_timestamp']
            nearest_stamps_list.append(closest_timestamp)

        markers['Closest_timestamp'] = nearest_stamps_list
        first_pupil_timestamp = data["pupil_timestamp"].min()
        markers["Relative_Time"] = markers["Closest_timestamp"] - first_pupil_timestamp
        markers["Sample_Index"] = (markers["Relative_Time"] * srate).astype(int)
        events = np.column_stack((markers["Sample_Index"], np.zeros_like(markers["Sample_Index"]), markers["label"]))


                # ===== THRESHOLD & FILTER PUPIL DATA =====
        confidence_threshold = 0.6

        columns_to_pick = ['eye_id', 'pupil_timestamp', 'blink', 'confidence', 'model_confidence', 'diameter', 'diameter_3d']
        df = data[columns_to_pick]

        # Step 1: Filter and clean left/right eye data
        df_left = df[(df["eye_id"] == 1) & (df["confidence"] >= confidence_threshold)].dropna(subset=["diameter"]).reset_index(drop=True)
        df_right = df[(df["eye_id"] == 0) & (df["confidence"] >= confidence_threshold)].dropna(subset=["diameter"]).reset_index(drop=True)
        
        # Step 2: Define helper to remap marker timestamps to filtered pupil data
        def map_events_to_filtered_data(filtered_df, markers_df, srate):
            first_ts = filtered_df["pupil_timestamp"].min()
            # Find closest timestamps in filtered_df
            closest_timestamps = []
            for ts in markers_df["timestamp"]:
                closest_ts = filtered_df.iloc[(filtered_df["pupil_timestamp"] - ts).abs().idxmin()]["pupil_timestamp"]
                closest_timestamps.append(closest_ts)
            
            # Compute sample index relative to new start
            relative_times = np.array(closest_timestamps) - first_ts
            sample_indices = (relative_times * srate).astype(int)
        
            # Create events array
            events = np.column_stack((sample_indices, np.zeros(len(sample_indices), dtype=int), markers_df["label"].to_numpy()))
            return events
        
        # Step 3: Map events for left and right eye separately
        events_left = map_events_to_filtered_data(df_left, markers, srate)
        events_right = map_events_to_filtered_data(df_right, markers, srate)
        
        # Step 4: Create MNE RawArrays
        info_left = mne.create_info(ch_names=['PupilDia_Left'], sfreq=srate, ch_types='eeg')
        info_right = mne.create_info(ch_names=['PupilDia_Right'], sfreq=srate, ch_types='eeg')
        
        eye_chan_left = mne.io.RawArray(df_left["diameter"].to_numpy().reshape(1, -1), info_left)
        eye_chan_right = mne.io.RawArray(df_right["diameter"].to_numpy().reshape(1, -1), info_right)
        

        subject_epochs = {}
        
        # ===== REJECTION FUNCTION =====
        def reject_low_conf_epochs(epochs, confidence_data, threshold=0.8, label="unknown", subject="unknown"):
            reject_idx = []
            for i, epoch in enumerate(epochs.get_data()):
                start_idx = epochs.events[i, 0]
                end_idx = start_idx + len(epoch[0])
                if end_idx > len(confidence_data):  # Prevent out-of-range indexing
                    reject_idx.append(i)
                    continue
                conf_values = confidence_data.iloc[start_idx:end_idx]
                if (conf_values < threshold).mean() > 0.5:
                    reject_idx.append(i)
                    
            total_epochs = len(epochs)
            dropped = len(reject_idx)
            
            # Log under the correct subject and label
            drop_log[subject][label]["total"] = total_epochs
            drop_log[subject][label]["dropped"] = dropped
            
            print(f"[{label}] Total: {len(epochs)} | Dropped due to confidence: {len(reject_idx)}")
            
            return epochs.drop(reject_idx, reason="low confidence")

        # ===== LEFT EYE EPOCHING =====
        if event_dict_five:
            try:
                epochs_five_left = mne.Epochs(eye_chan_left, events_left, tmin=-3, tmax=10, event_id=event_dict_five, 
                                              baseline=None, preload=True, reject=None, reject_by_annotation=False)
                epochs_five_left = reject_low_conf_epochs(epochs_five_left, df_left["confidence"], label="five_left", subject=subject_id)
                subject_epochs['epochs_five_left'] = epochs_five_left
            except Exception as e:
                print(f"five_left error: {e}")

        if event_dict_nine:
            try:
                epochs_nine_left = mne.Epochs(eye_chan_left, events_left, tmin=-3, tmax=18, event_id=event_dict_nine, 
                                              baseline=None, preload=True, reject=None, reject_by_annotation=False)
                epochs_nine_left = reject_low_conf_epochs(epochs_nine_left, df_left["confidence"], label="nine_left", subject=subject_id)
                subject_epochs['epochs_nine_left'] = epochs_nine_left
            except Exception as e:
                print(f"nine_left error: {e}")

        if event_dict_thirteen:
            try:
                epochs_thirteen_left = mne.Epochs(eye_chan_left, events_left, tmin=-3, tmax=26, event_id=event_dict_thirteen, 
                                                  baseline=None, preload=True, reject=None, reject_by_annotation=False)
                epochs_thirteen_left = reject_low_conf_epochs(epochs_thirteen_left, df_left["confidence"], label="thirteen_left", subject=subject_id)
                subject_epochs['epochs_thirteen_left'] = epochs_thirteen_left
            except Exception as e:
                print(f"thirteen_left error: {e}")

        # ===== RIGHT EYE EPOCHING =====
        if event_dict_five:
            try:
                epochs_five_right = mne.Epochs(eye_chan_right, events_right, tmin=-3, tmax=10, event_id=event_dict_five, 
                                               baseline=None, preload=True, reject=None, reject_by_annotation=False)
                epochs_five_right = reject_low_conf_epochs(epochs_five_right, df_right["confidence"], label="five_right", subject=subject_id)
                subject_epochs['epochs_five_right'] = epochs_five_right
            except Exception as e:
                print(f"five_right error: {e}")

        if event_dict_nine:
            try:
                epochs_nine_right = mne.Epochs(eye_chan_right, events_right, tmin=-3, tmax=18, event_id=event_dict_nine, 
                                               baseline=None, preload=True, reject=None, reject_by_annotation=False)
                epochs_nine_right = reject_low_conf_epochs(epochs_nine_right, df_right["confidence"], label="nine_right", subject=subject_id)
                subject_epochs['epochs_nine_right'] = epochs_nine_right
            except Exception as e:
                print(f"nine_right error: {e}")

        if event_dict_thirteen:
            try:
                epochs_thirteen_right = mne.Epochs(eye_chan_right, events_right, tmin=-3, tmax=26, event_id=event_dict_thirteen, 
                                                   baseline=None, preload=True, reject=None, reject_by_annotation=False)
                epochs_thirteen_right = reject_low_conf_epochs(epochs_thirteen_right, df_right["confidence"], label="thirteen_right", subject=subject_id)
                subject_epochs['epochs_thirteen_right'] = epochs_thirteen_right
            except Exception as e:
                print(f"thirteen_right error: {e}")

        # ===== SAVE EPOCHS FOR SUBJECT =====
        all_epochs[subject_id] = subject_epochs


    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        
#dropped epochs logger
log_rows = []

for subject, labels in drop_log.items():
    for label, stats in labels.items():
        log_rows.append({
            "subject": subject,
            "label": label,
            "total_epochs": stats["total"],
            "dropped_epochs": stats["dropped"]
        })

drop_df = pd.DataFrame(log_rows)
drop_df.to_csv(os.path.join(data_dir, "epoch_drop_log.csv"), index=False)
print("Saved structured drop log.")

import pandas as pd
df = pd.read_csv("D:\\Dataset\\epoch_drop_log.csv")
print(df.head())
#%% Storing data in a 5D Array and Pickling the data

subject_keys = list(all_epochs.keys())
n_subjects = len(subject_keys)
condition_labels = [
    'justlisten/five', 'memory/correct/five', 'memory/incorrect/five',
    'justlisten/nine', 'memory/correct/nine', 'memory/incorrect/nine',
    'justlisten/thirteen', 'memory/correct/thirteen', 'memory/incorrect/thirteen'
]
n_conditions = len(condition_labels)
eye_labels = ['left', 'right']
n_eyes = 2
n_channels = 1

# Step 1: Find max values to pad
max_epochs = 0
max_time_points = 0
for subj in subject_keys:
    for cond in condition_labels:
        for eye in eye_labels:
            key = f"epochs_{cond.split('/')[-1]}_{eye}"
            epochs = all_epochs[subj].get(key)
            if epochs:
                data = epochs.get_data()  # shape: (n_epochs, 1, time)
                max_epochs = max(max_epochs, data.shape[0])
                max_time_points = max(max_time_points, data.shape[-1])

# Step 2: Initialize 6D array (subjects, conditions, epochs, time, channel, eye)
final_data = np.full(
    (n_subjects, n_conditions, max_epochs, max_time_points, n_channels, n_eyes),
    np.nan
)

# Step 3: Fill the array
for subj_idx, subj in enumerate(subject_keys):
    for cond_idx, cond in enumerate(condition_labels):
        for eye_idx, eye in enumerate(eye_labels):
            key = f"epochs_{cond.split('/')[-1]}_{eye}"
            epochs = all_epochs[subj].get(key)
            if epochs:
                data = epochs.get_data()  # shape: (n_epochs, 1, time)
                n_epochs, _, time_len = data.shape
                final_data[subj_idx, cond_idx, :n_epochs, :time_len, :, eye_idx] = np.transpose(data, (0, 2, 1))  # to (n_epochs, time, channel)

# Step 4: Save to pickle
output_pickle = os.path.join(data_dir, "compiled_pupil_6D_array.pkl")

with open(output_pickle, 'wb') as f:
    pickle.dump({
        "data": final_data,
        "subjects": subject_keys,
        "conditions": condition_labels,
        "eyes": eye_labels,
        "srate": srate
    }, f)

print(f"\n✅ Saved 6D pupil array (all epochs) to {output_pickle}")
print(f"Shape: {final_data.shape} -> (subjects, conditions, epochs, time, channel, eye)")


#%% Visualizing pickle file data
import matplotlib.pyplot as plt

# Load data
with open("D:\\Dataset\\compiled_pupil_6D_array.pkl", "rb") as f:
    data_dict = pickle.load(f)

data = data_dict["data"]
subjects = data_dict["subjects"]
conditions = data_dict["conditions"]
eyes = data_dict["eyes"]
srate = data_dict["srate"]

# Select indices
sub_idx = 0
cond_idx = conditions.index("memory/incorrect/five")
epoch_idx = 0  # pick any valid epoch index (0 to 59)
eye_idx = eyes.index("left")

# Extract signal
signal = data[sub_idx, cond_idx, epoch_idx, :, 0, eye_idx]

# Handle NaNs (optional: mask or skip if needed)
if np.isnan(signal).all():
    print("All NaNs in selected signal.")
else:
    time = np.arange(len(signal)) / srate
    plt.plot(time, signal)
    plt.title(f"Subject {subjects[sub_idx]} | Condition: {conditions[cond_idx]} | Eye: {eyes[eye_idx]} | Epoch: {epoch_idx}")
    plt.xlabel("Time (s)")
    plt.ylabel("Pupil Diameter")
    plt.grid(True)
    plt.tight_layout()
    plt.show()












