# ✅ Imports
import os
import mne
import numpy as np
import pickle
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm

#%%

# ✅ Parameters
SAVE_PLOTS = False  # Set True to save EEG plots
dataset_path = "D:\\Dataset\\"
output_root = "C:\\Users\\student\\Desktop\\Nusaibah Farrukh\\output_eeg_data"
output_images_dir = "C:\\Users\\student\\Desktop\\Nusaibah Farrukh\\output_eeg_images"
os.makedirs(output_root, exist_ok=True)
if SAVE_PLOTS:
    os.makedirs(output_images_dir, exist_ok=True)

# Constants
num_conditions = 3
num_subconditions = 3
num_trials = 54
max_time = 26000  # same as ECG
sampling_rate = 1000

# ✅ Event dictionary
event_dicts = {
    "five": {
        "justlisten/five": 1,
        "memory/correct/five": 29,
        "memory/incorrect/five": 28
    },
    "nine": {
        "justlisten/nine": 2,
        "memory/correct/nine": 31,
        "memory/incorrect/nine": 30
    },
    "thirteen": {
        "justlisten/thirteen": 3,
        "memory/correct/thirteen": 33,
        "memory/incorrect/thirteen": 32
    }
}

#%%

# ✅ Duration and reject criteria
epoch_durations = {"five": 10, "nine": 18, "thirteen": 26}
reject_criteria = dict(eeg=200e-6)  # 200 µV

# ✅ Exclude problematic subjects
excluded_subjects = {
    "sub-013", "sub-014", "sub-015", "sub-016", "sub-017", "sub-018", "sub-019", "sub-020", 
    "sub-021", "sub-022", "sub-023", "sub-024", "sub-025", "sub-026", "sub-027", "sub-028", 
    "sub-029", "sub-030", "sub-031", "sub-037", "sub-066", "sub-094"
}
#%%

# ✅ Locate EEG .set files
def find_set_files(root_dir):
    set_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".set") and "task-memory_eeg" in file:  # for EEG now
                set_files.append(os.path.join(root, file))
    return set_files

set_files = find_set_files(dataset_path)

# ✅ Data storage
subject_list = []
data_array = []

#%%

# ✅ Loop through all EEG files
for file_path in tqdm(set_files, desc="Processing EEG subjects"):
    subject_id = os.path.basename(file_path).split("_")[0]
    if subject_id in excluded_subjects:
        continue

    try:
        print(f"\n🔍 Processing {subject_id}...")
        raw = mne.io.read_raw_eeglab(file_path, preload=True)

        # Select frontal channel(s), e.g., Fz (modify if needed)
        frontal_channel = "Fz"
        if frontal_channel not in raw.ch_names:
            print(f"⚠️ {subject_id}: {frontal_channel} not found, skipping.")
            continue

        raw.pick_channels([frontal_channel])
        events, _ = mne.events_from_annotations(raw)

        # Preallocate subject 4D array
        subject_data = np.full((num_conditions, num_subconditions, num_trials, max_time), np.nan, dtype=np.float32)

        for condition_idx, (condition, event_dict) in enumerate(event_dicts.items()):
            available_event_ids = set(events[:, 2])
            filtered_event_dict = {k: v for k, v in event_dict.items() if v in available_event_ids}
            if not filtered_event_dict:
                print(f"⚠️ Skipping condition '{condition}' for {subject_id}: No matching events.")
                continue

            try:
                epochs = mne.Epochs(
                    raw, events, tmin=-3, tmax=epoch_durations[condition],
                    event_id=filtered_event_dict, preload=True,
                    reject=reject_criteria, baseline=None
                )
            except Exception as e:
                print(f"❌ [{subject_id}] Epoching failed: {e}")
                continue

            for trial_idx in range(min(num_trials, len(epochs))):
                try:
                    selected_epoch = epochs[trial_idx]
                    eeg_signal = selected_epoch.get_data()[0, 0, :]

                    if np.isnan(eeg_signal).all() or len(eeg_signal) < 1000 or np.std(eeg_signal) < 1e-6:
                        continue

                    time_length = len(eeg_signal)
                    if time_length > max_time:
                        eeg_data_trimmed = eeg_signal[:max_time]
                    else:
                        eeg_data_trimmed = np.pad(eeg_signal, (0, max_time - time_length), constant_values=np.nan)

                    # Event label
                    event_code = epochs.events[trial_idx, 2]
                    if event_code in event_dict.values():
                        event_label = list(event_dict.keys())[list(event_dict.values()).index(event_code)]
                        subcondition_idx = 0 if "justlisten" in event_label else (1 if "memory/correct" in event_label else 2)
                    else:
                        continue

                    subject_data[condition_idx, subcondition_idx, trial_idx, :] = eeg_data_trimmed.astype(np.float32)

                    # Optional EEG plot
                    if SAVE_PLOTS:
                        plt.figure(figsize=(10, 2))
                        plt.plot(eeg_data_trimmed)
                        plt.title(f"{subject_id} - {condition} - Trial {trial_idx}")
                        plt.xlabel("Time (ms)")
                        plt.ylabel("EEG (µV)")
                        plot_path = os.path.join(output_images_dir, f"{subject_id}_{condition}_{trial_idx}.png")
                        plt.savefig(plot_path)
                        plt.close()

                except Exception as e:
                    print(f"⚠️ [{subject_id}] Trial {trial_idx} processing failed: {e}")
                    continue

        subject_list.append(subject_id)
        data_array.append(subject_data)
        del raw, epochs, subject_data
        gc.collect()

    except Exception as e:
        print(f"❌ Failed to process {subject_id}: {e}")
        continue


# ✅ Save as pickle
pickle_filename = os.path.join(output_root, "processed_eeg_data.pkl")
with open(pickle_filename, "wb") as f:
    pickle.dump({"subjects": subject_list, "data": np.array(data_array, dtype=np.float32)}, f)

print("\n EEG Processing Complete!")
print(f" Data shape: {np.array(data_array).shape}")
print(f" Subjects: {len(subject_list)} | Saved to: {pickle_filename}")

























# ✅ Loop through all EEG files
for file_path in tqdm(set_files, desc="Processing EEG subjects"):
    subject_id = os.path.basename(file_path).split("_")[0]
    if subject_id in excluded_subjects:
        continue

    try:
        print(f"\n🔍 Processing {subject_id}...")
        raw = mne.io.read_raw_eeglab(file_path, preload=True)

        # Select frontal channel(s), e.g., Fz (modify if needed)
        frontal_channel = "Fz"
        if frontal_channel not in raw.ch_names:
            print(f"⚠️ {subject_id}: {frontal_channel} not found, skipping.")
            continue

        raw.pick_channels([frontal_channel])
        events, _ = mne.events_from_annotations(raw)

        # Preallocate subject 4D array
        subject_data = np.full((num_conditions, num_subconditions, num_trials, max_time), np.nan, dtype=np.float32)

        for condition_idx, (condition, event_dict) in enumerate(event_dicts.items()):
            available_event_ids = set(events[:, 2])
            filtered_event_dict = {k: v for k, v in event_dict.items() if v in available_event_ids}
            if not filtered_event_dict:
                print(f"⚠️ Skipping condition '{condition}' for {subject_id}: No matching events.")
                continue

            try:
                epochs = mne.Epochs(
                    raw, events, tmin=-3, tmax=epoch_durations[condition],
                    event_id=filtered_event_dict, preload=True,
                    reject=reject_criteria, baseline=None
                )
            except Exception as e:
                print(f"❌ [{subject_id}] Epoching failed: {e}")
                continue

            for trial_idx in range(min(num_trials, len(epochs))):
                try:
                    selected_epoch = epochs[trial_idx]
                    eeg_signal = selected_epoch.get_data()[0, 0, :]

                    if np.isnan(eeg_signal).all() or len(eeg_signal) < 1000 or np.std(eeg_signal) < 1e-6:
                        continue

                    time_length = len(eeg_signal)
                    if time_length > max_time:
                        eeg_data_trimmed = eeg_signal[:max_time]
                    else:
                        eeg_data_trimmed = np.pad(eeg_signal, (0, max_time - time_length), constant_values=np.nan)

                    # Event label
                    event_code = epochs.events[trial_idx, 2]
                    if event_code in event_dict.values():
                        event_label = list(event_dict.keys())[list(event_dict.values()).index(event_code)]
                        subcondition_idx = 0 if "justlisten" in event_label else (1 if "memory/correct" in event_label else 2)
                    else:
                        continue

                    subject_data[condition_idx, subcondition_idx, trial_idx, :] = eeg_data_trimmed.astype(np.float32)

                    # Optional EEG plot
                    if SAVE_PLOTS:
                        plt.figure(figsize=(10, 2))
                        plt.plot(eeg_data_trimmed)
                        plt.title(f"{subject_id} - {condition} - Trial {trial_idx}")
                        plt.xlabel("Time (ms)")
                        plt.ylabel("EEG (µV)")
                        plot_path = os.path.join(output_images_dir, f"{subject_id}_{condition}_{trial_idx}.png")
                        plt.savefig(plot_path)
                        plt.close()

                except Exception as e:
                    print(f"⚠️ [{subject_id}] Trial {trial_idx} processing failed: {e}")
                    continue

        subject_list.append(subject_id)
        data_array.append(subject_data)
        del raw, epochs, subject_data
        gc.collect()

    except Exception as e:
        print(f"❌ Failed to process {subject_id}: {e}")
        continue

# ✅ Save as pickle
new_directory = "C:\\Users\\student\\Desktop\\Nusaibah Farrukh\\Pickle file" 
pickle_filename = os.path.join(new_directory, "processed_eeg_data.pkl")

with open(pickle_filename, "wb") as f:
    pickle.dump({"subjects": subject_list, "data": np.array(data_array, dtype=np.float32)}, f)

print("\n EEG Processing Complete!")
print(f" Data shape: {np.array(data_array).shape}")
print(f" Subjects: {len(subject_list)} | Saved to: {pickle_filename}")
