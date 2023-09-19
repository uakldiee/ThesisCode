# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import os
import re
import matplotlib.pyplot as plt
from pandas.plotting import table
import seaborn as sns
from scipy.stats import pearsonr
from pylab import *
import sys
import time
from modules.main_assemblies_detection import *
import h5py


# %%
thispath = os.path.abspath('..')
path_data = thispath + '/' + 'data'
pathdataread = path_data + '/' + '20230512_10Fishes'

# %%
def get_fish_names(folder):
    """Reads all fish names from the path"""
    regpattern = re.compile("^([a-zA-Z0-9]+).*\.mat$")
    allfiles = os.listdir(folder) 
    ret = []
    counter = 0
    for file in allfiles:
        if regpattern.match(file):
            counter += 1
            fishname = regpattern.match(file).group(1)
            ret.append(fishname)
    # unique only        
    fishes = list(set(ret))
    print(f"Found n = {counter} data files, and n = {len(fishes)} fishes")
    if counter == 0:
        raise ValueError("No .mat file found in folder")
    return fishes


fishes = get_fish_names(pathdataread)


# %%
def fullpath_file(datafolder,fishname,filetoread='GLUTAMATERGICS'):
    """
    Returns the actual file name and the date string.
    """
    file_types = ['GLUTAMATERGICS','CHOLINERGIC','CLUSTERS','ENTROPIES_EH','ALL_CELLS']
    # find all files in folder that match the fishname
    assert filetoread in file_types, \
        f"filetoread must be one of {file_types}, but you gave {filetoread}"
    allfiles = os.listdir(datafolder) 
    # if GLU or CHOL
    if filetoread in ['GLUTAMATERGICS','CHOLINERGIC']:
        regpattern = re.compile("^"+fishname+"_.*"+filetoread+"-(.*)\.mat$")
        for file in allfiles:
            if regpattern.match(file):
                print(f"Found file {file}")
                date_str = regpattern.match(file).group(1)
                return (datafolder + '/' + file)
    # other files, there is no date
    regpattern = re.compile("^"+fishname+"_.*"+filetoread+"\.mat$")
    for file in allfiles:
        if regpattern.match(file):
            print(f"Found file {file}")
            return (datafolder + '/' + file)
    raise ValueError(f"No file found for fish {fishname}\n in folder {datafolder}")

# %%
def get_assemblies_per_neurons(assembliesCells,n_cells):
  assem_neurons = pd.DataFrame(assembliesCells).T
  assemblies_per_neuron = []
  n_assemb = assem_neurons.shape[0]
  for neuron in range(0,n_cells):
    neurons_assemblies = []
    for assembly in range(0,n_assemb):
      if neuron in assem_neurons.iloc[assembly].to_numpy()[0]:
        neurons_assemblies.append(assembly)
    assemblies_per_neuron.append(neurons_assemblies)
  return assemblies_per_neuron


# %%
def read_one_fish(datafolder,fishname):
    """ 
    Reads ALL relevant data from one experiment folder,
    and returns a dataframe.
    """
    glutfile = fullpath_file(datafolder,fishname,'GLUTAMATERGICS')
    cholfile = fullpath_file(datafolder,fishname,'CHOLINERGIC')
    clufile = fullpath_file(datafolder,fishname,'CLUSTERS')
    
    rasterfile = datafolder + '/' + f"{fishname}_PART_1_stack_0_MultiPlane_RAW_RASTER.mat"
    # rasters for neurons
    rasters = loadmat(rasterfile)
    getcols = lambda x: [col for col in rasters[x].T]
    spike_events = getcols("spikes")
    spike_activity = getcols("raster")
    # glutamergic neurons
    glutamergic = loadmat(glutfile)
    cholinergic = loadmat(cholfile)
    
    isglut = glutamergic["glutNeurons"].flatten().astype(bool)
    ischol = cholinergic["ch_neurons"].flatten().astype(bool)
    n_cells = len(spike_events)
    cells = list(range(0,n_cells))
    # cell coordinates
    entrofile = fullpath_file(datafolder,fishname,'ENTROPIES_EH')
    entropy_all = loadmat(entrofile)
    cellxy = entropy_all["cell_xy"]
    cell_x = cellxy[:,0].tolist()
    cell_y = cellxy[:,1].tolist()
    # add the plane 
    cellsfile = fullpath_file(datafolder,fishname,'ALL_CELLS')
    cells_all = loadmat(cellsfile)
    theplanes = cells_all['fromPlane'].flatten().tolist()
    clusters_all = loadmat(clufile)
    n_cells = glutamergic["glutNeurons"].shape[1]
    assemblies_per_neuron = np.array(get_assemblies_per_neurons(clusters_all["assembliesCells"],n_cells), dtype="object").flatten()
    # create the dataframe and return it
    dfret = pd.DataFrame(\
      {'fish':fishname, 'cell':cells ,\
        'cell_x':cell_x, 'cell_y':cell_y,\
        'cell_plane':theplanes,\
        'assemblies': assemblies_per_neuron,\
        'spike_events':spike_events, 
        'spike_activity':spike_activity,\
        'isglut':isglut, 'ischol':ischol})
    
    return dfret

testdf = read_one_fish(pathdataread,'VG10')
testdf
# %%
# Dataframe of all activity combined
dfs = []
for fish in fishes:
    dfs.append(read_one_fish(pathdataread, fish))
spikes_df_all = pd.concat(dfs)
spikes_df_all.sort_values(by=['fish', 'cell'], inplace=True)

# %%
# assemblies

def read_assemblies_one(datafolder,fishname):
    clufile = fullpath_file(datafolder,fishname,'CLUSTERS')
    entrofile = fullpath_file(datafolder,fishname,'ENTROPIES_EH')
    clusters_all = loadmat(clufile)
    entropy_all = loadmat(entrofile)
    # assembly membership
    clusters = clusters_all["assembliesCells"][0]-1
    nclusters = len(clusters)
    clustercells = [ c[0] for c in clusters ]
    clusterncells = [ len(c[0]) for c in clusters ]
    # now, the entropy
    avg_dist = entropy_all["avg_dist"][0]
    std_dist = entropy_all["std_dist"][0]
    entropy = entropy_all["h"][0]
    
    dfret = pd.DataFrame(\
        {'fish':fishname,\
         'cluster_id':list(range(nclusters)),\
         'cluster_cells':clustercells ,\
         'cluster_ncells':clusterncells,\
         'avg_dist':avg_dist,\
         'std_dist':std_dist,\
          'entropy':entropy})     
    return dfret

# %%
dfs = []
for fish in fishes:
    dfs.append(read_assemblies_one(pathdataread, fish))
assemblies_df_all = pd.concat(dfs)
assemblies_df_all.sort_values(by=['fish', 'cluster_id'], inplace=True)
assemblies_df_all

# %%
testdf = assemblies_df_all
plt.scatter(testdf['avg_dist'], testdf['entropy'],color='black',s=0.7)
plt.xlabel('average distance between cells')
plt.ylabel('entropy')
plt.title('Entropy vs Average Distance')
plt.show()

# %%
# calculating averge excitatory neuron ratio in an assembly 
exploded_assembly_df = assemblies_df_all.explode('cluster_cells')

merged_df = exploded_assembly_df.merge(spikes_df_all, left_on='cluster_cells', right_on='cell')
merged_df['e_ratio_in_assembly'] = merged_df.groupby(['fish_x', 'cluster_id'])['isglut'].transform('mean')
average_ratios = merged_df.groupby(['fish_x', 'cluster_id'])['e_ratio_in_assembly'].mean().reset_index()


# Calculate the average e_ratio_in_assembly for each fish
average_ratios = average_ratios.groupby('fish_x')['e_ratio_in_assembly'].mean().reset_index()

# Rename the columns for clarity
average_ratios.columns = ['fish', 'e_ratio_in_assembly']
average_ratios.set_index('fish', inplace=True)
average_ratios

# %%
#  E/I composition within assemblies
fish_array=[]
cluster_array=[]
e_num_array=[]
i_num_array=[]
for fish in fishes:
    
    fish_df = merged_df[merged_df["fish_x"]==fish]
    clusters = np.unique(fish_df.cluster_id.values)
    mean_cluster_size  = np.mean(fish_df.drop_duplicates(subset='cluster_id', keep='first').cluster_ncells.values)
    for cluster in clusters:
        fish_dff = fish_df[fish_df["cluster_id"]==cluster]
        unique_rows = fish_dff.drop_duplicates(subset='cluster_cells', keep='first')
        e_num = unique_rows.isglut.values.sum() / mean_cluster_size
        i_num = (unique_rows.shape[0] - e_num) / mean_cluster_size
        fish_array.append(fish)
        cluster_array.append(cluster)
        e_num_array.append(e_num)
        i_num_array.append(i_num)
df_temp = pd.DataFrame(\
        {'fish':fish_array,\
         'cluster_id':cluster_array,\
         'e_num':e_num_array,\
         'i_num':i_num_array})     
df_temp


# %%
plt.hist(df_temp.e_num.values,alpha = 0.5,color ="grey")


# %%
def count_neurons(df_neurons):
  n_neus = df_neurons.shape[0]
  n_e = df_neurons.isglut.values.sum() 
  n_i = n_neus - n_e
 
  e_ratio = n_e / (n_e+n_i)
  n_neus_in_assembly = df_neurons[df_neurons['assemblies'].apply(len) > 0].shape[0]
  n_neus_no_assembly = df_neurons[df_neurons['assemblies'].apply(len) == 0].shape[0]
  neus_in_assembly_ratio = n_neus_in_assembly / n_neus

  return pd.Series({'n_neus':n_neus, 'n_e':n_e, 'n_i':n_i,'e_ratio':e_ratio, 'n_neus_in_assembly':n_neus_in_assembly,'n_neus_no_assembly':int(n_neus_no_assembly),'neus_in_assembly_ratio':neus_in_assembly_ratio})

# %%

df_counts = spikes_df_all.groupby('fish').apply(count_neurons)
for fish, row in df_counts.iterrows():
  print(f"""
  fish {fish} has {row.n_neus} neurons,
  of those, {row.n_e} are excitatory and {row.n_i} are inhibitory
  E neurons are {row.e_ratio*100:.1f}% of the total
        """)
df_counts


# %%
def assem_stats(df_assemblies):
    n_assemblies = df_assemblies.shape[0]
    avg_assembly_size = df_assemblies["cluster_ncells"].mean()
    
    return pd.Series({'n_assemblies':n_assemblies,'avg_assembly_size':avg_assembly_size})
# %%
assem_stats = assemblies_df_all.groupby('fish').apply(assem_stats)
for fish, row in assem_stats.iterrows():
  print(f"""
  fish {fish} has {row.n_assemblies} assemblies,
  and an average assembly size of {row.avg_assembly_size}
        """)
assem_stats

# %%
selected_cols = [df_counts[['n_neus',	'n_e',	'n_i',	'e_ratio',	'n_neus_in_assembly',	'n_neus_no_assembly',	'neus_in_assembly_ratio']]
,assem_stats[['n_assemblies',	'avg_assembly_size']],average_ratios[['e_ratio_in_assembly']]]

data_in_numbers = pd.concat(selected_cols, axis=1)
data_in_numbers


# %% [markdown]
# Firing rates histograms

# %%
def firing_rates(spike_trains):
  firing_rates = []
  for i in range(0,spike_trains.shape[0]):
    train = np.argwhere(spike_trains[i]==1)
    t = 3600 # seconds
    n = train.shape[0]
    f = n / t
    if f != 0:
      firing_rates.append(f)
  return firing_rates


# %%
def firing_rate_distributionsEverythingNormalised(path_data, fish):
    experiments = read_one_fish(path_data,fish)
    experiment_dataE = firing_rates(experiments.loc[(experiments.isglut == 1)].spike_activity.values)
    experiment_dataI = firing_rates(experiments.loc[(experiments.isglut == 0)].spike_activity.values)
    experiment_dataAll = firing_rates(experiments.spike_activity.values)

    E_min = np.min(experiment_dataE)
    I_min = np.min(experiment_dataI)
    All_min = np.min(experiment_dataAll)

    E_max = np.max(experiment_dataE)
    I_max = np.max(experiment_dataI)
    All_max = np.max(experiment_dataAll)

    E_mean = np.mean(experiment_dataE)
    I_mean = np.mean(experiment_dataI)
    All_mean = np.mean(experiment_dataAll)

    
    bins = np.arange(np.min([E_min,I_min,All_min]), 0.5, 0.02)


    excitatory_counts, _ = np.histogram(experiment_dataE, bins=bins)
    inhibitory_counts, _ = np.histogram(experiment_dataI, bins=bins)
    all_counts, _ = np.histogram(experiment_dataAll, bins=bins)


    fig, axs = plt.subplots(3, 1, figsize=(6, 11))

    # Normalize the histogram by firing rate
    bin_widths = np.diff(bins)

    excitatory_density = excitatory_counts / (np.sum(excitatory_counts) * bin_widths)
    inhibitory_density = inhibitory_counts / (np.sum(inhibitory_counts) * bin_widths)
    all_density = all_counts / (np.sum(all_counts) * bin_widths)

    fig.suptitle(f"{fish}: Histogram of firing rates", fontsize=12)
    plt.subplots_adjust(top=0.95)
    

    axs[0].bar(bins[:-1], inhibitory_density, width=bin_widths,alpha = 0.5, color ="red", label='Inhibitory')
    axs[0].axvline(x=E_mean, color='#39FF14', linestyle='--', label='Mean')
    axs[0].set_xlabel('Firing rate')                         
    axs[0].set_ylabel('Probability density')   
    axs[0].legend(loc="upper right")
    
    axs[1].bar(bins[:-1], excitatory_density, width=bin_widths,alpha = 0.5, color ="blue", label='Excitatory')
    axs[1].axvline(x=I_mean, color='#39FF14', linestyle='--', label='Mean')
    axs[1].set_xlabel('Firing rate')                         
    axs[1].set_ylabel('Probability density')   
    axs[1].legend(loc="upper right")
    
    axs[2].bar(bins[:-1], all_density, width=bin_widths,alpha = 0.5, color ="grey", label='All')
    axs[2].axvline(x=All_mean, color='#39FF14', linestyle='--', label='Mean')
    axs[2].set_xlabel('Firing rate')                         
    axs[2].set_ylabel('Probability density')   
    axs[2].legend(loc="upper right")
        

    
    print(f"Mean firing rate E: {np.mean(experiment_dataE)}")
    print(f"Mean firing rate I: {np.mean(experiment_dataI)}")
    
    
    plt.show()

# %%
firing_rate_distributionsEverythingNormalised(pathdataread, "CTR05")

# %%
fish = "CTR05"
df = assemblies_df_all[assemblies_df_all["fish"  ]==fish].cluster_ncells.values
mean_size = int(np.mean(df))
fig, axs = plt.subplots()
plt.hist(df,alpha = 0.5,color ="grey")
plt.grid(axis='y', alpha=0.75)
plt.axvline(x=mean_size, color='#39FF14', linestyle='--', label='Mean')
plt.legend(loc="upper right")
plt.xlabel('Assemby Size')
plt.ylabel('Count')
plt.title('Histogram of Assembly Sizes')
plt.show() 
#fig.savefig( thispath + '/' + 'zebrafish_tectum' + '/' + 'plots' + f"/{fish}_histogram_of_assembly_sizes.png")

# %% [markdown]
# ### Plot numbers of assemblies E/I neurons belong to

# %%
def neuron_assembly_distribution(fish,path_project):
    fish_name = fish

    assemblies_per_neurons = [[] for i in range(3)]
    assemblies_per_neurons[0] = spikes_df_all["assemblies"].values.tolist() #  assemblies from all neurons
    assemblies_per_neurons[1] = spikes_df_all[spikes_df_all.isglut == 1]["assemblies"].values.tolist() # assemblies from exitatory neurons
    assemblies_per_neurons[2] = spikes_df_all[spikes_df_all.isglut == 0]["assemblies"].values.tolist() # assemblies from inhibitory neurons
    

    # count the number of assemblies each neuron is part of for all neuron types
    n_assemblies_per_neuron = [[] for i in range(3)]
    for index, neuron_type in enumerate(assemblies_per_neurons):
        for assemblies in neuron_type:
            n_assemblies_per_neuron[index].append(len(assemblies))

    # count the number of occurences of each number of assemblies each neuron is part of
    # and sort by index 
    neuron_participations = [[] for i in range(3)]
    for i in range(3):
        neuron_participations[i] = pd.Series(n_assemblies_per_neuron[i]).value_counts().sort_index()

    assemblies_number = [[] for i in range(3)]
    for i in range(3):
        assemblies_number[i] = neuron_participations[i].index

    # bars = number of neurons 
    # representing the whole range from min to max number of assemblies
    x_achs = [[] for i in range(3)]
    for i in range(3):
        # assemblies_number[i].max()+1 could be set as max instead but used 16 to keep it consitent across experiments
        x_achs[i] = np.arange(assemblies_number[i].min(),16)


    # this is avoid skiping over a possible assembly number - representing the whole range
    # we also want to see that no neuron is participating in this many assemblies
    y_achs = [[] for i in range(3)]
    for i in range(3):
        for value in x_achs[i]:
            if value in assemblies_number[i].tolist():
                y_achs[i].append(neuron_participations[i][value])
            else:
                y_achs[i].append(0)

    # the plot for E/I neurons
    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_axes([0,0,1,1])

    ax.bar(x_achs[1] + 0.00, y_achs[1],alpha = 0.5,color = 'blue', width = 0.25)
    ax.bar(x_achs[2] + 0.25, y_achs[2],alpha = 0.5, color = 'red', width = 0.25)
    xticks = np.asarray(x_achs[1])
    ax.set_xticks(xticks)
    for i, v in enumerate(y_achs[1]):
        ax.text(i , v + 4, str(v),alpha = 0.5, color="blue", fontweight='bold')

    for i, v in enumerate(y_achs[2]):
        ax.text(i + 0.25 , v + 4, str(v), alpha = 0.5,color="red", fontweight='bold')

    ax.legend(labels=['Excitatory', 'Inhibitory'])

    plt.xlabel(f"Number of assemblies E/I neurons belongs to")
    plt.ylabel("Number of neurons")
    plt.title(f"{fish_name}: Numbers of assemblies E/I neurons belong to")

    #fig.savefig(path_project / "plots" / f"fish_{fish_name}_plots" / f"{fish_name}_EI_neurons_assembly_distribution.pdf")
    plt.show()

    #plot summary
    excitatory_mean = x_achs[1].mean()
    inhibitory_mean = x_achs[2].mean()
    excitatory_std = x_achs[1].std()
    inhibitory_std = x_achs[2].std()
    print(f"""
    fish {fish_name}: exitatory mean {excitatory_mean},excitatory standard devistion: {excitatory_std};
    inhibitory mean: {inhibitory_mean}, inhibitory standard deviation: {inhibitory_std}
        """)

    # the plot of all neurons
    fig2 = plt.figure(figsize = (10, 5))
    ax2 = fig2.add_axes([0,0,1,1])


    ax2.bar(x_achs[0], y_achs[0], alpha = 0.5,color = 'grey', width = 0.4)
    xticks = np.asarray(x_achs[0])
    ax2.set_xticks(xticks)

    # display the value of neuron numbers
    for i, v in enumerate(y_achs[0]):
        ax2.text(i - 0.1, v + 4, str(v),alpha = 0.5, color= 'grey', fontweight='bold')

    plt.xlabel(f"Number of a assemblies a neuron belongs to")
    plt.ylabel("Number of neurons")
    plt.title(f"{fish_name}: Numbers of assemblies neurons belong to")

    #fig.savefig(thispath + '/' + 'zebrafish_tectum' + '/' + 'plots' + f"/{fish}_neuron_assembly_distribution.png")
    plt.show()

# %%
neuron_assembly_distribution("CTR05",thispath)


# %% [markdown]
# ### Plot E/I in the tectum

# %%
def make_df_from_cell_per(cell_per_all,planes_all):
    df_temp_list = []
    for (k,pp) in enumerate(cell_per_all): # k - enumerates the neurons
        pp = pp[0] 
        theplane = planes_all[k][0] 
        dftemp = pd.DataFrame({"plane":theplane,
                    "neuron":k, 
                    "x_coord":pp[:,0],"y_coord":pp[:,1]})
        df_temp_list += [dftemp]            
    return pd.concat(df_temp_list)

# %%
def tectum_plot_centroids(fish, thispath):

    theplot,axs = plt.subplots(1,3,figsize=(15,int(6)))

    theplot.suptitle(f"{fish}: E/I Neurons in the Optic Tectum", fontsize=12)
    plt.subplots_adjust(top=0.95)

    file_all_cells = fullpath_file(pathdataread,fish,'ALL_CELLS')
    all_cells = loadmat(file_all_cells)
    avg_all = all_cells['avg']
    df_per = make_df_from_cell_per(all_cells['cell_per'],all_cells['fromPlane'])

    glut_file = fullpath_file(pathdataread,fish,'GLUTAMATERGICS')
    glutamergic = loadmat(glut_file)

    glut = glutamergic["glutNeurons"][0]
    df_glut_temp = pd.DataFrame({"neuron":range(len(glut)), "is_glut":(glut == 1)})
    df_per_glut = df_per.merge(df_glut_temp,on='neuron',how='left')

    color_glut = 'blue' 
    color_nonglut = 'red'

    for (plane,df_plane) in df_per_glut.groupby('plane'):
        myax = axs[plane-1]
        myax.set_aspect('equal')
        myax.axis('off')
        myax.set_title(f"Plane {plane}")
        
        myax.imshow(avg_all[:,:,plane-1],cmap="gray")
        for (neuron_and_glut,df_neuron) in df_plane.groupby(['neuron','is_glut']):
            _is_glut = neuron_and_glut[1]
            mycol = color_glut if _is_glut else color_nonglut
        
            xs = df_neuron['x_coord'].values
            ys = df_neuron['y_coord'].values
            xc = xs.mean()
            yc = ys.mean()
            
            myax.scatter(xc,yc,color=mycol,s=1)
    #theplot.savefig(thispath + '/' + 'zebrafish_tectum' + '/' + 'plots' + f"{fish}_tectum_plot_EI_neuron_centroids.pdf")


# %%
tectum_plot_centroids("CTR05",thispath)

#%%
spikes_df_all['spike_count'] = spikes_df_all['spike_activity'].apply(lambda x: np.sum(x))
act = spikes_df_all[spikes_df_all['spike_count'] > 0]

# %%
# calculate the average firing rates of neurons for each fish

mean_firing_rates = []
mean_firing_rates_non_0 = []
for fish in fishes:
    mean_firing_rate = np.mean(firing_rates(spikes_df_all[spikes_df_all["fish"] == fish].spike_events.values))
    mean_firing_rate_non_0 = np.mean(firing_rates(act[act["fish"] == fish].spike_events.values))
    mean_firing_rates.append(mean_firing_rate)
    mean_firing_rates_non_0.append(mean_firing_rate_non_0)
mean_FR_per_fish = pd.DataFrame(\
        {'fish':fishes,\
         'mean_firing_rate':mean_firing_rates,\
         'mean_firing_rate_non_0':mean_firing_rates_non_0})  
fish_higest_FR = mean_FR_per_fish[mean_FR_per_fish["mean_firing_rate"]==np.max(mean_FR_per_fish.mean_firing_rate)].fish.values[0]
print(fish_higest_FR)
mean_FR_per_fish['mean_firing_rate'].mean()




# %% [markdown]
# ### Plot a raster plot of 50 neruons with the highest firing rate 

# %%
def frames_to_seconds(spike_train):
    
    frame_rate=15
    time_duration_seconds = len(spike_train) / frame_rate

    # Convert spike train to seconds-based binary representation
    spike_train_seconds = np.zeros(int(time_duration_seconds), dtype=int)
    spike_indices = np.where(spike_train == 1)[0]
    spike_train_seconds[spike_indices // frame_rate] = 1
        
    spike_train_seconds = np.array(spike_train_seconds)
    return spike_train_seconds
spikes_df_all["spikes_over_seconds"] = spikes_df_all["spike_activity"].apply(frames_to_seconds)

# %%
# remove neurons that are not active
active_neurons_df = spikes_df_all[spikes_df_all["spikes_over_seconds"].apply(np.sum) > 0]

fish_higest_FR_df = active_neurons_df[active_neurons_df["fish"] == fish_higest_FR]

#create a new column with neuron firing rates
fish_higest_FR_df["firing_rate"] = fish_higest_FR_df["spikes_over_seconds"].apply(np.sum) / fish_higest_FR_df["spikes_over_seconds"].apply(len)

# order neurons by their firing rates
fish_higest_FR_df = fish_higest_FR_df.sort_values(by=['firing_rate'], ascending=False)

# select the top 50 neurons
fish_higest_FR_50_spikes = fish_higest_FR_df[:50]


# %%
def plot_50(fish_higest_FR,fish_higest_FR_50_spikes,thispath):
    fish_name = fish_higest_FR
    spikes = []
    EI_color_grading = True
    experiment_df = fish_higest_FR_50_spikes
    n_cells = experiment_df.shape[0]
    for i in range(n_cells):
        spikes.append([experiment_df.spike_events.values[i],experiment_df.isglut.values[i]])

    plotname = f"{fish_name}: Raster plot of 50 neurons with the highest firing rate "
    plt.rcParams["figure.figsize"] = [10, 5]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots()
    axs.set_ylabel("neurons",fontsize=12)
    axs.set_xlabel(f"seconds",fontsize=12)
    axs.set_title(plotname,fontsize=15)
    spikes = np.array(spikes,dtype=object)

    for (row,spk) in enumerate(spikes):
        spike_train = spk[0]
        time_train = np.arange(len(spike_train)) / 15
        spike_times = time_train[spike_train == 1]

        #xs = np.argwhere(spk[0]==1)
        nx = len(spike_times)
        ys = np.full(nx,row)
        if spk[1] == 0:
            color = "red"
        if spk[1] == 1:
            color = "blue"
        if EI_color_grading == False:
            color = "black"
        axs.scatter(spike_times,ys,c=color,s= 0.2)
    #fig.savefig(thispath + '/' + 'zebrafish_tectum' + '/' + 'plots' + f"{fish}_highest_FR_raster.pdf")
    plt.show()


# %%
plot_50(fish_higest_FR,fish_higest_FR_50_spikes,thispath)


# %% [markdown]
# ## Convolution Analysis

# %%
# convolve each spike train with the kernel
# Calculate the number of frames corresponding to the desired kernel width
def convolve_trains(spike_train, kernel_size, kernel_type):

    if kernel_type == "rectangle":
        frame_rate = 15  # Frame rate in Hz
        kernel_frames = int(kernel_size * frame_rate) # 0.5 second frame
        kernel = np.ones(kernel_frames) / kernel_frames

        smoothed_spike_train = np.convolve(spike_train, kernel, mode='same')
    if kernel_type == "gaussian":
        gaussian_kernel = np.exp(-0.5 * (np.linspace(-10, 10, kernel_size) / 2)**2)
        gaussian_kernel /= gaussian_kernel.sum() # normalized to ensure that its values sum up to 1
        smoothed_spike_train = np.convolve(spike_train, gaussian_kernel, mode='same')
    
    return np.array(smoothed_spike_train)


# %%
fish_df = spikes_df_all[spikes_df_all["fish"] == fish_higest_FR]
fish_df['spike_count'] = fish_df['spikes_over_seconds'].apply(lambda x: np.sum(x))

# 'active': True if the neuron is active, False if the spike count == 0
fish_df['active'] = fish_df['spike_count'] != 0

## STATISTICS
# number of neurons in the fish
print(f""" 
The number of neurons in the fish {fish_higest_FR} is {fish_df.shape[0]}. """)
# number of active nuruons in the fish
print(f""" 
Out of which {fish_df['active'].sum()} are active.
      """)
# mean spike count with non active neurons and without
mean_spike_count_active = fish_df[fish_df['active']]['spike_count'].mean()
mean_spike_count_all = fish_df['spike_count'].mean()
print("Mean spike count of active neurons:", mean_spike_count_active)
print("Mean spike count of all neurons:", mean_spike_count_all)

# spike count of the most active neuron and its Id
max_spike_count_index = fish_df['spike_count'].idxmax()
cell_id_highest_spike = fish_df.loc[max_spike_count_index, 'cell']
max_spike_count = fish_df.loc[max_spike_count_index, 'spike_count']
print(f"""
The neuron {cell_id_highest_spike} is the most active with the 
spike count of {max_spike_count}.
      """)

# %%
# looking at the most active neuron
most_active_neuron = fish_df.loc[max_spike_count_index]
most_active_spike_train = most_active_neuron['spikes_over_seconds']

# %%


def convolution_analysis(neuron_spike_train,gaus_kernel_size):
    # apply convolution to smooth the spike train
   
    frame_rate = 15  # Frame rate in Hz
    kernel_size = 1
    kernel_frames = int(kernel_size * frame_rate) # 0.5 second frame
    spike_times = np.where(most_active_spike_train == 1)[0]


    # Create the rectangular kernel with the specified width
    kernel = np.ones(kernel_frames) / kernel_frames

    smoothed_spike_train = np.convolve(most_active_spike_train, kernel, mode='same')

    # Plot the original spike train and the smoothed spike train
    time = np.arange(len(most_active_spike_train))  # Time axis
    plt.figure(figsize=(10, 10))
    plt.subplot(4, 1, 1)
    plt.plot(time, most_active_spike_train, 'k-', linewidth=0.5)
    plt.title('Original Spike Train')
    plt.xlabel('Time')
    plt.ylabel('Spikes (s)')
    plt.xlim(spike_times[0]-50, 3600)

    # second plot - take fixed bins and compute the firing rate as a funtion of time 
    plt.subplot(4, 1, 2)
    plt.suptitle(f"{fish}: most active neuron", fontsize=12)
    
    bin_width = 4  

    # Calculate the number of bins and create an array of bin edges
    num_bins = len(most_active_spike_train) // bin_width
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
    plt.xlabel('Time (s)')
    plt.ylabel('Firing Rate')
    plt.title(f'Firing Rate(bin width: {bin_width} s)')
    plt.xlim(spike_times[0]-50, 3600)


    plt.subplot(4, 1, 3)
    plt.plot(time, smoothed_spike_train, 'r-', linewidth=1.5, color='b')
    plt.title(f'Spike Train with Rectangular Kernel ({kernel_size} second frame)')
    plt.xlabel('Time (s)')
    plt.ylabel('Smoothed Spikes')
    plt.xlim(spike_times[0]-50, 3600)


    # Plot the Gaussian kernel - still averaging - a weighted average, with less weigth at the edges
    # Larger kernel sizes result in smoother curves - trade-off between capturing fine details and achieving smooth results
    gaussian_kernel = np.exp(-0.5 * (np.linspace(-10, 10, gaus_kernel_size) / 2)**2)
    gaussian_kernel /= gaussian_kernel.sum() # normalized to ensure that its values sum up to 1
    convolved_spike_train_gaussian = np.convolve(most_active_spike_train, gaussian_kernel, mode='same')
    plt.subplot(4, 1, 4)  # Third subplot
    plt.plot(time, convolved_spike_train_gaussian, 'b-', linewidth=1.5)
    plt.title(f'Spike Train with Gaussian Kernel ({gaus_kernel_size} points)')
    plt.xlabel('Time (s)')
    plt.ylabel('Smoothed Spikes')
    plt.xlim(spike_times[0]-50, 3600)

    plt.tight_layout()
    #plt.savefig(thispath + '/' + 'zebrafish_tectum' + '/' + 'plots' + f"/{fish}_most_active_neuon_convolutions.pdf")
    plt.show()
convolution_analysis(most_active_spike_train,100)

# %% [markdown]
# order the neurons in the raster plot above by assembly
# calculate a correlation matrix for all active neuons - get average correlation
# calculate the correlation matrix for only the 50 neurons that are most active

# %%
# all active neuons in the fish

fish_higest_FR_df = spikes_df_all[spikes_df_all["fish"] == fish_higest_FR]
fish_higest_FR_active_df = fish_higest_FR_df[fish_higest_FR_df["spikes_over_seconds"].apply(np.sum) > 0]

# remove cholinergic neurons
fish_higest_FR_no_chol_df = fish_higest_FR_active_df[fish_higest_FR_active_df['ischol'] == False]

#change args to change the convolution kernel
fish_higest_FR_no_chol_df["convolved_spikes"] = fish_higest_FR_no_chol_df["spikes_over_seconds"].apply(convolve_trains,args=(100, "gaussian"))

# remove all neurons that aren't part of a compact assembly
assemblies = read_assemblies_one(pathdataread,fish_higest_FR)

# calculate the median entropy and use it as a threshold to remove all the assemblies with higher entropy
median_entropy = assemblies['entropy'].median()
compact_assemblies = assemblies[assemblies['entropy'] <= median_entropy]
neu_ids_in_compact_assem = np.unique(np.concatenate(compact_assemblies["cluster_cells"].values))
neuorns_in_compact = fish_higest_FR_no_chol_df[fish_higest_FR_no_chol_df['cell'].isin(neu_ids_in_compact_assem)]

# order the assemblies by entropy
compact_assemblies_assend_entropy = compact_assemblies.sort_values(by='entropy', ascending=True)

# order the neurons in correlation matrix by assembly
order_by_assemblies = np.concatenate(compact_assemblies_assend_entropy["cluster_cells"].values)

order_df = pd.DataFrame({'cell': order_by_assemblies})

# Merge the order DataFrame with the original DataFrame to reorder the rows
ordered_by_assembly_df = order_df.merge(neuorns_in_compact, on='cell')
ordered_by_assembly_df



# %%
def corr_matrix_calc(df,corr_type,fish):
    spike_trains_assem = np.array(df["convolved_spikes"].values.tolist())
    num_neurons = len(spike_trains_assem)
    corr_matrix = np.zeros((num_neurons, num_neurons))
    for i in range(num_neurons):
        corr_matrix[i, i] = 1.0
        for j in range(i+1,num_neurons):
            corr, _ = pearsonr(spike_trains_assem[i], spike_trains_assem[j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    # plotting the correlation matrix
    fig, axs = plt.subplots()
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1)

    # Add labels and title
    plt.xlabel('Neuron ID')
    plt.ylabel('Neuron ID')
    plt.title(f'{fish} Correlation Matrix')
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=5.5)
    plt.yticks(fontsize=5.5)
    plt.xticks(rotation=90, ha='right')
    plt.subplots_adjust(bottom=0.15)

    plt.yticks(rotation=0, ha='right')
    plt.subplots_adjust(left=0.15)
    #plt.savefig(thispath + '/' + 'zebrafish_tectum' + '/' + 'plots' + f"/{fish}_correlation_matrix_{corr_type}.pdf")

    plt.show()
    return corr_matrix

#%%

act["spikes_over_seconds"] = act["spike_events"].apply(frames_to_seconds)
act_VG01 = act[act['fish'] == 'VG01']
act_VG01["convolved_spikes"] = act_VG01["spikes_over_seconds"].apply(convolve_trains,args=(100, "gaussian"))
act_VG01
#%%
corr_all_active = corr_matrix_calc(act_VG01,"all_filtered_neurons",fish_higest_FR)
# %%
# Calculate the correlation matrix
# not ordered neurons - neuorns_in_compact
corr_all = corr_matrix_calc(neuorns_in_compact,"all_filtered_neurons",fish_higest_FR)
# neurons ordered by assembly - ordered_by_assembly_df
#corr_ordered = corr_matrix_calc(ordered_by_assembly_df,"ordered_by_assembly",fish_higest_FR)

# %%
#create a new column with neuron firing rates
neuorns_in_compact["firing_rate"] = neuorns_in_compact["spikes_over_seconds"].apply(np.sum) / neuorns_in_compact["spikes_over_seconds"].apply(len)

# order neurons by their firing rates
neuorns_in_compact_ordered_by_FR = neuorns_in_compact.sort_values(by=['firing_rate'], ascending=False)

# select the top 50 neurons
neuorns_in_compact_ordered_by_FR_50 = neuorns_in_compact_ordered_by_FR[:50]

#%%
np.concatenate(neuorns_in_compact_ordered_by_FR_50.assemblies.values)
cluster_counts = neuorns_in_compact_ordered_by_FR_50['assemblies'].apply(pd.Series).stack().value_counts()
cluster_counts
# %%
plot_50(fish_higest_FR,neuorns_in_compact_ordered_by_FR_50,thispath)
coor_m_50 = corr_matrix_calc(neuorns_in_compact_ordered_by_FR_50,"50",fish_higest_FR)

#%%
fish_higest_FR_50_spikes["convolved_spikes"] = fish_higest_FR_50_spikes["spikes_over_seconds"].apply(convolve_trains,args=(100, "gaussian"))
#%%
coor_m_50_not_filtered = corr_matrix_calc(fish_higest_FR_50_spikes,"50_2",fish_higest_FR)

# %%
# the average corralation

def average_correlation(corr_matrix):
    # Get the upper triangular part of the correlation matrix (excluding diagonal)
    upper_triangular = np.triu(corr_matrix, k=1)

    # Calculate the average correlation
    average_correlation = np.mean(upper_triangular)

    print("Average Correlation:", average_correlation)
    return average_correlation



# %%
average_correlation(coor_m_50)

#%%
average_correlation(corr_all)

#%%
average_correlation(coor_m_50_not_filtered)
#%%
average_correlation(corr_all_active)

# %% [markdown]
# * Plot a raster plot with neuons ordered by assembly with E/I - only compact assemblies  
# * Choose by eye two assemblies that look visualy different                               
# * calculate a correlation matrix for these two asemblies, order the neuons by assembly    
# * calculate the average correaltion for both, and inbetween           

# %%
# plotting the distributiion of entropy values
# entropy here provides a measure of the variability or spread of distances within an assembly
# Higher entropy indicates greater variability in the distances - more dispersed or spread-out assembly 
# while lower entropy indicates more consistent or concentrated distances between neurons - more compact assembly


assemblies["entropy"].values
entropy_values = np.array(assemblies["entropy"].values)

# Compute the mean of the entropy values
mean_entropy = np.mean(entropy_values)
fig, axs = plt.subplots()
# Plotting the kernel density estimation of entropy values
sns.kdeplot(entropy_values, shade=True)
plt.xlabel('Entropy')
plt.ylabel('Density')
plt.axvline(mean_entropy, color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.title('VG01: Distribution of Entropy Values')
plt.legend()
plt.show()


# %%
# plot the topograpthy of the assembly with the highest entropy, lowest entropy and the mean entropy
# the topography is the spatial distribution of the neurons in the assembly
def assembly_topography_plot(assemblies,assembly_neurons_id,fishdf,fish):
    fish = fish
    all_cells_file = fullpath_file(pathdataread,fish,'ALL_CELLS')
    all_cells = loadmat(all_cells_file)
    avg_all = all_cells['avg']

    theplot,axs = plt.subplots(figsize=(3,5))

    axs.imshow(avg_all[:,:,1],cmap="gray")
    axs.axis('off')
    assembly_neurons = fishdf[fishdf["cell"].isin(assembly_neurons_id)]
    for index, neuron in assembly_neurons.iterrows():
        xs = neuron['cell_x']
        ys = neuron['cell_y']
        
        axs.scatter(xs,ys,color="yellow",s=5)


# %%
max_entropy = assemblies.loc[assemblies['entropy'].idxmax()]
min_entropy =  assemblies.loc[assemblies['entropy'].idxmin()]
diff = np.abs(assemblies['entropy'] - assemblies['entropy'].mean())
mean_entropy = assemblies.loc[diff.idxmin()]

assembly_topography_plot(assemblies,max_entropy['cluster_cells'],fish_df,fish_higest_FR)
assembly_topography_plot(assemblies,mean_entropy['cluster_cells'],fish_df,fish_higest_FR)
assembly_topography_plot(assemblies,min_entropy['cluster_cells'],fish_df,fish_higest_FR)

# %%
# read_one_experiment, matrix - spikes_df_all
# assemblies_one_experiment, assemblies_df - assemblies_df_all
matrix = spikes_df_all[spikes_df_all["fish"] == fish_higest_FR]

assemblies_df = assemblies_df_all[assemblies_df_all["fish"] == fish_higest_FR]
compact_assemblies = assemblies_df[assemblies_df['entropy'] <= median_entropy]
#compact_assemblies.reset_index(drop=True, inplace=True)
compact_assemblies.iloc[[1,56]]

# %%
def neural_activity_by_assembly_plot_EI(fish_higest_FR,assemblies_df,matrix,fig_x,fig_y,what_to_plot):
    color = "black"
    fish_name = fish_higest_FR
    spikes_per_neuron = np.array(matrix[what_to_plot])
    iglut = matrix["isglut"]
    assembly_neurons = assemblies_df["cluster_cells"]
    spikes_by_assembly = []

    assembly_devider = []
    assembly_size = 0
    for assembly in assembly_neurons:
        assembly_size += len(assembly)
        assembly_devider.append(assembly_size)
        for neuron in assembly:
            spikes_by_assembly.append([np.array(spikes_per_neuron[neuron]),iglut[neuron]])
        
    spikes_by_assembly = np.array(spikes_by_assembly,dtype=object)


    # the plot
    caption = "spikes orderd by assembly"
    plotname = f"{fish_name} E/I neural activity ordered by assembly"
    plt.rcParams["figure.figsize"] = [fig_x, fig_y]
    plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots()
    axs.set_ylabel("neurons")
    axs.set_xlabel(f"frame \n Note: {caption}")
    axs.set_title(plotname)
    # horizontal line dividing assemblies in the plot
    [axs.axhline(y=i, linewidth=0.3) for i in assembly_devider]

    for (row,spk) in enumerate(spikes_by_assembly):
        xs = np.argwhere(spk[0]==1)
        nx = len(xs)
        ys = np.full(nx,row)
        if spk[1] == 0:
            color = "red"
        if spk[1] == 1:
            color = "blue"
        axs.scatter(xs,ys,c=color,s=0.1)

    #fig.savefig(path_project + "/plots" + f"/fish_{fish_name}_plots" + f"/{fish_name}_EI_neural_activity_by_assembly.pdf")
    plt.show()

# %%
neural_activity_by_assembly_plot_EI(fish_higest_FR,compact_assemblies,matrix,21,42,"spikes_over_seconds")
# %%
neural_activity_by_assembly_plot_EI(fish_higest_FR,compact_assemblies.iloc[[1,56]],matrix,21,7,"spikes_over_seconds")


# %%
assem1_index = 1
assem2_index = 56
two_assemblies_cells_ids = np.concatenate(compact_assemblies.iloc[[assem1_index,assem2_index]].cluster_cells.values) 
fish_temp = spikes_df_all[spikes_df_all["fish"] == fish_higest_FR]
fish_temp["convolved_spikes"] = fish_temp["spikes_over_seconds"].apply(convolve_trains,args=(100, "gaussian"))
ordered_by_two_assem = fish_temp[fish_temp['cell'].isin(two_assemblies_cells_ids)]
ordered_by_two_assem = ordered_by_two_assem.set_index('cell').loc[two_assemblies_cells_ids].reset_index()
coor_m_two_assem = corr_matrix_calc(ordered_by_two_assem,"two_assem",fish_higest_FR)



# %%
avg_corr_assem_1 = average_correlation(coor_m_two_assem[:55,:55])
avg_corr_assem_2 = average_correlation(coor_m_two_assem[54:,54:])
#%%
average_correlation(coor_m_two_assem[54:,:55])

# %% [markdown]
# * try to get rid of noise in both
# * run CAD for both
# * try lowering p values threshold

# %%
assem1 = ordered_by_two_assem.iloc[:55]
assem2 = ordered_by_two_assem.iloc[54:]

# %%
matrix = spikes_df_all[spikes_df_all["fish"] == fish_higest_FR]
matrix["spike_count"] =  matrix['spikes_over_seconds'].apply(lambda x: np.sum(x))


# %%
# plot raster of one neuron to check if removing isolated neurons worked
spike_train = np.array(matrix[matrix["cell"] == 1762].spikes_over_seconds.values[0])


spike_indices = np.where(spike_train == 1)[0]
rcParams['figure.figsize']=(100,3)
plot(spike_indices, np.ones_like(spike_indices), '.')                                
xlabel('Time (s)')                          
yticks([])  

xticks(np.arange(0, len(spike_train), 100))
                                
show()

#%%
def delete_isolated_spikes_new(spike_train, window=100, spike_window=4):
    
    spike_train = np.array(spike_train)

    # Find the indices of the spikes
    spike_indices = np.where(spike_train == 1)[0]

    # Group consecutive indices together
    super_spikes = np.split(spike_indices, np.where(np.diff(spike_indices) > spike_window)[0] + 1) 
    if len(super_spikes[0]) > 0:
        # For each super spike, check if there's another super spike in the window
        for super_spike in super_spikes:
            # Get the start and end of the window
            start = max(0, super_spike[0] - window)
            end = min(len(spike_train), super_spike[-1] + window)

            # Check if there's another super spike in the window
            within_window = []
            for other_super_spike in super_spikes:
                if(other_super_spike is not super_spike and (start <= other_super_spike[0] <= end or start <= other_super_spike[-1] <= end)):
                    within_window.append(other_super_spike)
            if len(within_window) ==0:
                spike_train[super_spike] = 0

    return spike_train


# %%

spike_train = np.array(delete_isolated_spikes_new(matrix[matrix["cell"] == 1762].spikes_over_seconds.values[0], window=100,spike_window=4))

# Find the indices of the spikes
spike_indices = np.where(spike_train == 1)[0]
rcParams['figure.figsize']=(100,3)
plot(spike_indices, np.ones_like(spike_indices), '.')  

xlabel('Time (s)')                          
yticks([])                                  
show()

# %%
matrix["filtered_spikes"] = matrix["spikes_over_seconds"].apply(delete_isolated_spikes_new, window=10,spike_window=4)

matrix["filtered_spike_count"] =  matrix['filtered_spikes'].apply(lambda x: np.sum(x))
matrix

# %%
neural_activity_by_assembly_plot_EI(fish_higest_FR,compact_assemblies.iloc[[1,56]],matrix,21,7,"filtered_spikes")

# %%
neural_activity_by_assembly_plot_EI(fish_higest_FR,compact_assemblies.iloc[[1,56]],matrix,21,7,"spikes_over_seconds")



# %%
convolution_analysis(matrix.iloc[matrix['filtered_spike_count'].idxmax()].filtered_spikes,100)

# %%
# get only active neurons
matrix["convolved_spikes"] = matrix["filtered_spikes"].apply(convolve_trains,args=(100, "gaussian"))
active_neurons = matrix[matrix["filtered_spike_count"] > 0]

active_neurons

# %%
corr_filtered = corr_matrix_calc(active_neurons,"blabla","VG01")

# %%
average_correlation(corr_filtered)



# %%
two_assemblies_cells_ids[:55]
assem1["filtered_spikes"] = assem1["spikes_over_seconds"].apply(delete_isolated_spikes_new, window=10,spike_window=4)


# %%
assem1_for_CAD = assem1[["fish","spikes_over_seconds"]]
assem1_for_CAD = assem1_for_CAD.spikes_over_seconds.values
#%%
assem2_for_CAD = assem2[["fish","spikes_over_seconds"]]
assem2_for_CAD = assem2_for_CAD.spikes_over_seconds.values

# %%
spM_assem1 = [[]]*len(assem1_for_CAD)
for i in range(len(assem1_for_CAD)):
    
    spike_train = assem1_for_CAD[i]
    time_train = np.arange(len(spike_train)) 
    spike_times = time_train[spike_train == 1]
    spM_assem1[i] = spike_times

#%%
spM_assem2 = [[]]*len(assem2_for_CAD)
for i in range(len(assem2_for_CAD)):
    
    spike_train = assem2_for_CAD[i]
    time_train = np.arange(len(spike_train)) 
    spike_times = time_train[spike_train == 1]
    spM_assem2[i] = spike_times

# %%
max_len = max(len(arr) for arr in spM_assem1)
# Pad each array with NaN values up to the maximum length because that how the data looks in CAD Russo example
spM_assem1 = [np.pad(np.array(arr, dtype=float), (0, max_len - len(arr)), mode='constant', constant_values=np.nan) for arr in spM_assem1]
spM_assem1 = np.array(spM_assem1)
spM_assem1.shape
spM_assem1
#%%
max_len = max(len(arr) for arr in spM_assem2)
# Pad each array with NaN values up to the maximum length because that how the data looks in CAD Russo example
spM_assem2 = [np.pad(np.array(arr, dtype=float), (0, max_len - len(arr)), mode='constant', constant_values=np.nan) for arr in spM_assem2]
spM_assem2 = np.array(spM_assem2)
spM_assem2.shape
spM_assem2



# %%
