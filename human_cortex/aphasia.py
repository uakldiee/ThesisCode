#%%%
from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
#%%
def load_NWB_data(data_dir_base, date):
    data_path = Path(data_dir_base)
    folders = data_path.iterdir()
    matching_folders = [folder.name for folder in folders if date in folder.name]

    all_data = []

    for folder_name in matching_folders:
        for file_name in Path(data_dir_base, folder_name).iterdir():
            if file_name.suffix == '.nwb':
                print(f'Loading data from {file_name}...\n'
                      f'Please wait, this might take a while.')
                
                with NWBHDF5IO(file_name, "r") as io:
                    read_nwbfile = io.read()

                    # Extract the data from the nwb file
                    units_df = read_nwbfile.units.to_dataframe()
                    num_neurons = len(read_nwbfile.units.spike_times_index.data[:])
                    trials_df = read_nwbfile.trials.to_dataframe()

                    # Close the nwb file
                    io.close()

                data_dict = {
                    'file_name': file_name,
                    'units_df': units_df,
                    'num_neurons': num_neurons,
                    'trials_df': trials_df
                }

                all_data.append(data_dict)

                print(f'Data successfully loaded from {file_name}.')

    for i in range(len(all_data)):
        print(f'For the data in index {i}:')
        all_data[i]['task'] = check_task_type(all_data[i]['trials_df']['TASK_ID'][1])

    return all_data


def match_neurons_spikes_region(units_df):
    AG_neuron_IDs = []
    IFG_neuron_IDs = []
    MFG_neuron_IDs = []
    SMG_neuron_IDs = []

    for i in range(len(units_df['spike_times'])):
        # check if the spikes are recorded from more than one electrode
        if len(np.unique(units_df['labels'][i])) > 1:
            print(f'Neuron {i} is recorded from more than one electrode, it will not be used')
            continue
        # match the neurons to the regions
        if units_df['labels'][i][0].startswith('AG'):
            AG_neuron_IDs.append(i)
        elif units_df['labels'][i][0].startswith('IFG'):
            IFG_neuron_IDs.append(i)
        elif units_df['labels'][i][0].startswith('MFG'):
            MFG_neuron_IDs.append(i)
        elif units_df['labels'][i][0].startswith('SMG'):
            SMG_neuron_IDs.append(i)

    print('Matched neuron IDs to regions.')

    return AG_neuron_IDs, IFG_neuron_IDs, MFG_neuron_IDs, SMG_neuron_IDs

def match_spikes_experiments(spikes, times, buffer=False):
    """
    Function to match the spikes to the experiments
    """
    start_times = times['start']
    stop_times = times['end']
    fs = 30_000 # sampling frequency in Hz

    if buffer:
        t_pre = 0.5
        t_post = 0.5
    else:
        t_pre = 0
        t_post = 0

    # find the spikes that are within the start and stop times of the experiment
    spikes_in_exp = [[] for _ in range(len(spikes))]

    for neuron_ID, neuron_spikes in enumerate(spikes):
        for start_time, stop_time in zip(start_times, stop_times):
            # if there is no spike in the experiment, append a nan
            pot_spikes = neuron_spikes[np.logical_and(neuron_spikes >= start_time - (t_pre * fs), neuron_spikes <= stop_time + (t_post * fs))]
            if pot_spikes.size == 0:
                spikes_in_exp[neuron_ID].append(np.nan)
                continue

            spikes_in_exp[neuron_ID].append(pot_spikes)

    print(f'Found {len(spikes_in_exp[0])} successfull trials for {len(spikes_in_exp)} neurons.')
    return spikes_in_exp

def match_exp_spikes_regions(spikes_in_exp, AG_neurons, IFG_neurons, MFG_neurons, SMG_neurons):
    AG_spikes_in_exp = [[] for _ in range(len(AG_neurons))]
    IFG_spikes_in_exp = [[] for _ in range(len(IFG_neurons))]
    MFG_spikes_in_exp = [[] for _ in range(len(MFG_neurons))]
    SMG_spikes_in_exp = [[] for _ in range(len(SMG_neurons))]

    for neuron_ID, neuron_spikes in enumerate(spikes_in_exp):
        if neuron_ID in AG_neurons:
            AG_spikes_in_exp[AG_neurons.index(neuron_ID)] = neuron_spikes
        elif neuron_ID in IFG_neurons:
            IFG_spikes_in_exp[IFG_neurons.index(neuron_ID)] = neuron_spikes
        elif neuron_ID in MFG_neurons:
            MFG_spikes_in_exp[MFG_neurons.index(neuron_ID)] = neuron_spikes
        elif neuron_ID in SMG_neurons:
            SMG_spikes_in_exp[SMG_neurons.index(neuron_ID)] = neuron_spikes

    print('Matched spikes to regions.')
    print(f'There are {len(AG_spikes_in_exp)} neurons in AG, {len(IFG_spikes_in_exp)} neurons in IFG, {len(MFG_spikes_in_exp)} neurons in MFG, and {len(SMG_spikes_in_exp)} neurons in SMG.')
    return AG_spikes_in_exp, IFG_spikes_in_exp, MFG_spikes_in_exp, SMG_spikes_in_exp

def check_task_type(task_ID):
    if task_ID == 92:
        task = 'PN'
        print('The task is picture naming (PN)')
    elif task_ID == 95:
        task = 'REP'
        print('The task is word repetition (REP)')
    elif task_ID == 91:
        task = 'WPM'
        print('The task is word picture mapping (WPM)')
    return task

def create_df_success_trials(trials_df, exp_duration=8):
    """creates a dataframe with only the successful trials"""

    # if the trial is not successful or too long, drop it
    success_trials_df = trials_df[(trials_df['COMPLETE_TRIAL'] != 0) & (trials_df['stop_time'] - trials_df['start_time'] <= exp_duration * 30_000)]
    success_trials_df = success_trials_df.reset_index(drop=True)

    print(f'Dropped {len(trials_df) - len(success_trials_df)} trials that were not successful or too long.\n'
            f' {len(success_trials_df)} trials remaining.')

    times = {'start': success_trials_df['start_time'],
             'end': success_trials_df['stop_time'],
             'stim2': success_trials_df['START_STIMULUS_2']
             }
    return success_trials_df, times
#%%
date = '20221117'

data_dir_base = "/home/ge35ruz/NASgroup/projects/2023Aphasia/spike sorted data"
data_list = load_NWB_data(data_dir_base, date)

for i in range(len(data_list)):
    globals()['units_df_' + data_list[i]['task']] = data_list[i]['units_df']
    globals()['num_neurons_' + data_list[i]['task']] = data_list[i]['num_neurons']
    globals()['trials_df_' + data_list[i]['task']] = data_list[i]['trials_df']

# exp_ID = 'PN'
exp_ID = 'REP'

units_df = globals()['units_df_' + exp_ID]
num_neurons = globals()['num_neurons_' + exp_ID]
trials_df = globals()['trials_df_' + exp_ID]

# match the neurons to the regions
AG_neurons, IFG_neurons, MFG_neurons, SMG_neurons = match_neurons_spikes_region(units_df)

# select the successful trials
success_trials_df, times = create_df_success_trials(trials_df, exp_duration=8)

# math the spike times to individual trials
spiketimes_in_exp = match_spikes_experiments(units_df['spike_times'], times)

# match activities of neurons with regions
AG_spikes_in_exp, IFG_spikes_in_exp, MFG_spikes_in_exp, SMG_spikes_in_exp = match_exp_spikes_regions(spiketimes_in_exp, AG_neurons, IFG_neurons, MFG_neurons, SMG_neurons)

#%%
regions = pd.DataFrame({
    "region_0_AG" :[AG_neurons],
    "region_1_IFG" :[IFG_neurons],
    "region_2_MFG" :[MFG_neurons],
    "region_3_SMG" :[SMG_neurons]
})
regions = regions.T
regions.columns = ["units"]
regions['count'] = regions['units'].apply(len)
regions['spike_trains'] = [AG_spikes_in_exp, IFG_spikes_in_exp, MFG_spikes_in_exp, SMG_spikes_in_exp]

# %%
times = pd.DataFrame(times)
times['duration'] = (times['end'] - times['start'])/30000

# %%
spiketimes_in_exp = pd.DataFrame(spiketimes_in_exp)
melted_spikes_experiments = pd.melt(spiketimes_in_exp.reset_index(), id_vars=['index'],value_vars=spiketimes_in_exp.columns)
melted_spikes_experiments.columns = [ 'cell_id', 'experiment_id','spike_train']

# %%
# Count the number of neurons with no spike train
neurons_with_no_spike_train = melted_spikes_experiments['spike_train'].isnull().sum()
print(f"Number of neurons with no spike train: {neurons_with_no_spike_train}")

# Remove rows with NaN values in the spike_train column
melted_spikes_experiments_active = melted_spikes_experiments.dropna(subset=['spike_train'])

# %%
def get_spike_trains_by_regions_and_experiment(melted_spikes_experiments, exp_regions, region, experiment_num):
    exp = melted_spikes_experiments[melted_spikes_experiments["experiment_id"]==experiment_num]
    units = exp_regions.loc[region]["units"]
    region_exp = exp[exp['cell_id'].isin(units)]
    region_exp = region_exp.dropna(subset=['spike_train'])

    return region_exp["spike_train"].values
# %%
regions.index = ['AG', 'IFG', 'MFG', 'SMG']
#%%
experiment = 0
region_name = "MFG"
spike_trains = get_spike_trains_by_regions_and_experiment(melted_spikes_experiments,regions,region_name,experiment)#%%
start_time = times.iloc[experiment].start
spike_trains = spike_trains - start_time

spike_trains = spike_trains/30000 # to seconds

# Plot the raster plot using scatter plot
fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = [10, 5]
for i, spike_train in enumerate(spike_trains):
    ax.scatter(spike_train, [i] * len(spike_train), s=0.2, c='grey') # , marker='|'

ax.set_xlabel('Seconds')
ax.set_ylabel('Neuron')
ax.set_title(f'Raster Plot of Neurons in Region {region_name} During a Trial ')

plt.show()

# %%
spike_trains_df = pd.DataFrame(spike_trains)
# rename the column
spike_trains_df.columns = ['spike_trains']
#remove NaN values from the spike_trains column
spike_trains_df.spike_trains = spike_trains_df.spike_trains.apply(lambda x: x[~np.isnan(x)])
# count number of spikes in each spike train
spike_trains_df['spike_count'] = spike_trains_df.spike_trains.apply(len)
# calculate the firing rate
spike_trains_df['firing_rate'] = spike_trains_df.spike_count / times.iloc[experiment].duration
# calculate the mean firing rate
mean_firing_rate = spike_trains_df.firing_rate.mean()

#%%
neuron_count = len(spike_trains_df)  # Get the number of neurons

fig, axs = plt.subplots()
firing_rates = spike_trains_df.firing_rate.values
All_min = np.min(firing_rates)
All_max = np.max(firing_rates)
All_mean = np.mean(firing_rates)

bins = np.arange(All_min, All_max, 2)
bin_widths = np.diff(bins)
all_counts, _ = np.histogram(firing_rates, bins=bins)

# Plot the bar chart with neuron count on the y-axis
axs.bar(bins[:-1], all_counts, width=bin_widths, alpha=0.5, color="grey")
axs.axvline(x=All_mean, color='#39FF14', linestyle='--', label='Mean')
axs.set_xlabel('Firing rate')
axs.set_ylabel('Number of neurons')
axs.set_title('Histogram of firing rates')
axs.legend(loc="upper right")

plt.show()

#%%
def binning(spike_train):
    
    bin_size = 0.0001
    n_bins = int(times.iloc[experiment].duration / bin_size) + 1

    # Initialize an array to store the spike counts
    binned_spike_train = [0] * n_bins

    # Bin the spike train
    for spike_time in spike_train:
        bin_index = int(spike_time / bin_size)
        binned_spike_train[bin_index] += 1

    return np.array(binned_spike_train)

#%%
spike_trains_values = spike_trains_df.spike_trains.values
spike_trains_values_binned = []
for spike_train in spike_trains_values:
    # Convert the spike train into a binned spike train
    binned_spike_train = binning(spike_train)
    spike_trains_values_binned.append(binned_spike_train)
spike_trains_values_binned = np.array(spike_trains_values_binned)

#%%
# Check if the spike trains are binary if to adjust bin size
binary_flag = all(all(bin_val == 0 or bin_val == 1 for bin_val in train) for train in spike_trains_values_binned)
binary_flag
#%%
# get the most active spike train
most_active_spike_train = spike_trains_values_binned[np.argmax(spike_trains_df.spike_count.values)]
#%%
# apply convolution to smooth the spike train

gaus_kernel_size = 5000
frame_rate = 15  # Frame rate in Hz
kernel_size = 1
kernel_frames = int(kernel_size * frame_rate) 
spike_times = np.where(most_active_spike_train == 1)[0]

# Create the rectangular kernel with the specified width
kernel = np.ones(kernel_frames) / kernel_frames

smoothed_spike_train = np.convolve(most_active_spike_train, kernel, mode='same')

# Plot the original spike train and the smoothed spike train
time = np.arange(len(most_active_spike_train))  # Time axis
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(time, most_active_spike_train, 'k-', linewidth=0.5)
plt.title('Original Spike Train')
plt.xlabel('Bins (binsize=0.0001s)')
plt.ylabel('Spike')


plt.subplot(3, 1, 2)
plt.suptitle(" most active neuron", fontsize=12)

bin_width = 250  

# Calculate the number of bins and create an array of bin edges
num_bins = int(len(most_active_spike_train) // bin_width)+1
bin_edges = np.arange(0, len(most_active_spike_train) + bin_width, bin_width)

# Calculate the firing rate in each bin
firing_rate = []
for i in range(num_bins):
    bin_start = i * bin_width
    bin_end = (i + 1) * bin_width 
    spikes_in_bin = np.sum(most_active_spike_train[bin_start:bin_end])
    firing_rate.append(spikes_in_bin / bin_width)  # Firing rate = spikes / bin width

time_axis = np.arange(0, len(most_active_spike_train), bin_width)

# Plot the firing rate as a function of time
plt.plot(time_axis, firing_rate,  linestyle='-', color='black')
plt.xlabel('Bins (binsize=0.0001s)')
plt.ylabel('Firing Rate')
plt.title(f'Firing Rate(bin width: {bin_width*0.0001} s)')



# Plot the Gaussian kernel - still averaging - a weighted average, with less weigth at the edges
# Larger kernel sizes result in smoother curves - trade-off between capturing fine details and achieving smooth results
gaussian_kernel = np.exp(-0.5 * (np.linspace(-10, 10, gaus_kernel_size) / 2)**2)
gaussian_kernel /= gaussian_kernel.sum() # normalized to ensure that its values sum up to 1
convolved_spike_train_gaussian = np.convolve(most_active_spike_train, gaussian_kernel, mode='same')
plt.subplot(3, 1, 3) 
plt.plot(time, convolved_spike_train_gaussian, 'b-', linewidth=1.5)
plt.title(f'Firing Rate (Gaussian Kernel, {gaus_kernel_size} points)')
plt.xlabel('Bins (binsize=0.0001s)')
plt.ylabel('Firing Rate')

plt.tight_layout()
plt.show()

#%%
# convolve all the spike trains
convolved_spikes = []
gaus_kernel_size = 5000
gaussian_kernel = np.exp(-0.5 * (np.linspace(-10, 10, gaus_kernel_size) / 2)**2)
gaussian_kernel /= gaussian_kernel.sum() 
for spike_train in spike_trains_values_binned:
    convolved_spike_train = np.convolve(spike_train, gaussian_kernel, mode='same')
    convolved_spikes.append(convolved_spike_train)

#%%
def average_correlation(corr_matrix):
    # Get the upper triangular part of the correlation matrix (excluding diagonal)
    upper_triangular = np.triu(corr_matrix, k=1)

    # Calculate the average correlation
    average_correlation = np.mean(upper_triangular)

    print("Average Correlation:", average_correlation)
    return average_correlation

# %%
spike_trains_values = convolved_spikes
num_neurons = len(spike_trains_values)
corr_matrix = np.zeros((num_neurons, num_neurons))
for i in range(num_neurons):
    corr_matrix[i, i] = 1.0
    for j in range(i+1,num_neurons):
        corr, _ = pearsonr(spike_trains_values[i], spike_trains_values[j])
        corr_matrix[i, j] = corr
        corr_matrix[j, i] = corr

# plotting the correlation matrix
fig, axs = plt.subplots()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1)


plt.xlabel('Neuron ID')
plt.ylabel('Neuron ID')
plt.title(f'{region_name} Correlation Matrix')
plt.gca().set_aspect('equal')
plt.xticks(fontsize=5.5)
plt.yticks(fontsize=5.5)
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.15)

plt.yticks(rotation=0, ha='right')
plt.subplots_adjust(left=0.15)

plt.show()

#%%
# Calculate the average correlation
average_correlation(corr_matrix)


# %%

# calculating mean firing rate for all experiments
mean_FR_expr = []
region_name = "IFG"
for experiment in range(0,98):
    spike_trains = get_spike_trains_by_regions_and_experiment(melted_spikes_experiments,regions,region_name,experiment)
    start_time = times.iloc[experiment].start
    spike_trains = spike_trains - start_time

    spike_trains = spike_trains/30000 # to seconds

    spike_trains_df = pd.DataFrame(spike_trains)
    spike_trains_df.columns = ['spike_trains']



    spike_trains_df.spike_trains = spike_trains_df.spike_trains.apply(lambda x: x[~np.isnan(x)])

    spike_trains_df['spike_count'] = spike_trains_df.spike_trains.apply(len)

    spike_trains_df['firing_rate'] = spike_trains_df.spike_count / times.iloc[experiment].duration

    mean_firing_rate = spike_trains_df.firing_rate.mean()
    mean_FR_expr.append(mean_firing_rate)
    
np.mean(mean_FR_expr)