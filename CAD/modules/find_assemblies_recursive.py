
import numpy as np
from modules.test_pair import test_pair
import scipy.io as scio
import sys
from pathlib import Path
import os
import math
from multiprocessing import  Pool
from itertools import repeat
import multiprocessing

def test_pairs(pair,spiketrain_binned,maxlag, alpha,Dc,min_occurences,reference_lag,assembly_in,assembly_out,significant_pairs,nns ):
    w1, w2 = pair
    spikeTrain2 = spiketrain_binned[w2,:].T
    
    # coverting to numpy arrays for everything to work within the test_pair function
    assemD = test_pair(assembly_in[w1], np.array(spikeTrain2), w2, maxlag, Dc, reference_lag)
    if assemD['pvalue'][-1] < alpha and assemD['n_occurences'][-1] > min_occurences:
        #assembly_out[nns - 1] = assemD
        #significant_pairs[w1, w2] = 1
        nns += 1
        return (w1,w2), assemD

def find_assemblies_recursive(spiketrain_binned,maxlag,alph,gg,Dc,min_occurences,O_th,bytelimit,reference_lag, n_workers = 20):
    assembly_output = []
    """The function agglomerate pairs of units (or a unit and a preexisting assembly), 
    tests their significance and stop when the detected assemblies reach their maximal dimention."""

    # renamed No_th to min_occurences = Minimal number of occurrences required for an assembly (all assemblies, even if significant, with fewer occurrences
        #than min_occurrences are discarded). Default: 0  
    # renamed binM to spiketrain_binned
        # spiketrain_binned - binary matrix, dimensions "nu x m", where "nu" is the number of elements(neurons) and "m" is the number of occurrences(spike count series).
        # spiketrain_binned - the recoreded spike trains binned - binning in the main_assemblies_detection function - Binned spike trains containing data to be analyzed
    # renamed nu to n_neurons for clarity
    # renamed ANfo to significant_pairs
    n_neurons = spiketrain_binned.shape[0] 

    # loop over each row of binM and store some info about the row in assembly_in
    assembly_in = [{'elements': None,
                    'lag': None,
                    'pvalue': None,
                    'times': None,
                    'n_occurences': None} for _ in range(n_neurons)]
    

    '''
    ## initialize empty assembly - in other implementation (OI)
    assembly_in = [{'neurons': None,
                    'lags': None,
                    'pvalue': None,
                    'times': None,
                    'signature': None} for _ in range(n_neurons)]'''

    for w1 in range(n_neurons):
            assembly_in[w1]["elements"] = w1 # contains the index of the current row - the neuron
            assembly_in[w1]["lag"] = []
            assembly_in[w1]["pvalue"] = []
            assembly_in[w1]["times"] = np.array(spiketrain_binned[w1, :].T) # contains the values in the current row of the "binM" matrix #spike time series
            assembly_in[w1]["n_occurences"] = spiketrain_binned[w1, :].sum() # called 'signature'in the OI

    # ANin = np.ones((nu,nu)) # what is this for? - not used anywhere

    #del assembly_out, Anfo

    # Significance levels α at each step of the agglomeration scheme are strictly Bonferroni-corrected as α¯i=α/R_i
    # R_i = total number of tests performed
    # here for R_1; nu = total number of single units(individual neurons) -> correcting for the total number of different pairs

    ''' first order = test over pairs'''

    # denominator of the Bonferroni correction
    # divide alpha by the number of tests performed in the first pairwise testing loop

    alpha = alph * 2 / (n_neurons * (n_neurons-1) * (2 * maxlag + 1))
    #n_as = ANin.shape[0] 
    #nu = ANin.shape[1] 

    #n_as, nu = spiketrain_binned.shape - wrong n_as and nu should be the same number

    # prANout = np.ones((nu,nu))  - not really used 

    # significant_pairs - matrix with entry of 1 for the significant pairs
    significant_pairs = np.zeros((n_neurons,n_neurons))
    
    assembly_out = [] # [[]] * (n_neurons * n_neurons) 
   

    assembly_out_dim = (n_neurons * n_neurons)

    # nns: count of the existing assemblies
    nns = 1 
    
    pairs = [(w1, w2) for w1 in range(n_neurons-1) for w2 in range(w1+1, n_neurons)]


    allargs = list(zip(pairs,repeat(spiketrain_binned),repeat(maxlag),repeat(alpha),repeat(Dc),repeat(min_occurences),repeat(reference_lag),repeat(assembly_in),repeat(assembly_out),repeat(significant_pairs), repeat(nns) ))
    if multiprocessing.get_start_method() is None:
        multiprocessing.set_start_method('spawn') 
    thepool = Pool(n_workers)
    results = thepool.starmap(test_pairs,allargs)

    valid_results = [result for result in results if result is not None]

    if valid_results:
        pair, added_assembly = zip(*valid_results)

    #pair, added_assembly = zip(*[result for result in results if result is not None])

        for pair, added_assembly in zip(pair, added_assembly):
            w1, w2 = pair
            significant_pairs[w1, w2] = 1
            assembly_out.append(added_assembly)
    else: 
        assembly_out =  [[]] * (n_neurons * n_neurons) 
        assembly_out[nns-1:]  = []

    #if len(assembly_out) < assembly_out_dim:
    #    assembly_out.extend([[]]* (assembly_out_dim - len(assembly_out)))

    #assembly_out[nns-1:]  = []
    #del assembly_in, assemD
    # del assemS1

    # making significant_pairs symmetric
    significant_pairs = significant_pairs + significant_pairs.T
    significant_pairs[significant_pairs==2] = 1
    

    assembly = assembly_out
    del assembly_out
    if not assembly:
        assembly_output = []

    #save the assembly to a .mat file

    thispath = Path(__file__).parent.resolve()
    path_project = (thispath/ '..'/ '..'/'..' ).resolve()  

    fname = f'Assembly_0{1}_b{gg}.mat'
    folder_path = path_project / 'thesis_code' / 'CAD' / 'output'
    file_path = os.path.join(folder_path, fname)
    scio.savemat(file_path, {'assembly': assembly}, format = '5')

    # second order and more: increase the assembly size by adding a new unit
    """__________________________increase the assembly size by adding a new unit_____________________________"""
    agglomeration_size = 1 
    element_added = True #  are new elememts added? if not stop while loop

    # while new units are added (Oincrement != 0) and the current size of assemblies is smaller than the maximal assembly order (O_th)
    while element_added and agglomeration_size < (O_th):

        element_added = False
        n_assem = len(assembly) # number of groups previously found
        assembly_out = [[]]*(n_assem*n_neurons)

        nns = 1
        for w1 in range(n_assem): # runs over existing assemblies
            w1_dict = assembly[w1]
            w1_elements = dict(w1_dict).get("elements")
            # Add only neurons that have significant first order cooccurrences with members of the assembly
            _, w2_to_test = np.where(significant_pairs[w1_elements, :] == 1) # discard the row indices by assigning them to _ (underscore), which is a conventional symbol in Python used for ignoring values that are not of interest.
            w2_to_test = w2_to_test[np.logical_not(np.isin(w2_to_test, w1_elements))]
            w2_to_test = np.unique(w2_to_test)

            # check that there are candidate neurons for agglomeration
            if len(w2_to_test) == 0:
                alpha = float('inf')
            else:
                # bonferroni correction only for the tests actually performed
                alpha = alph / (len(w2_to_test) * n_assem * (2 * maxlag + 1))  ## bonferroni correction only for the test that I actually perform

            for ww2 in range(len(w2_to_test)):
                w2 = w2_to_test[ww2]
                spikeTrain2 = spiketrain_binned[w2, :].T

                assemD = test_pair(dict(assembly[w1]), spikeTrain2,w2,maxlag,Dc, reference_lag)

                # if the assembly given in output is significant and the number of occurrences is higher than the minimum requested number
                if assemD['pvalue'][-1] < alpha and assemD['n_occurences'][-1] > min_occurences:
                    #assembly_out.append(assemD)
                    assembly_out[nns-1] = assemD
                    count_w1 = w1
                    while count_w1 >= n_neurons:
                        significant_pairs = increase_matrix_size(significant_pairs)
                        count_w1 = count_w1 + 1
                    significant_pairs[w1,w2] = 1
                    
                    element_added = True
                    nns += 1

                
                    
                
                #del spikeTrain2, assemD

        assembly_out[nns-1:] = []

        # finalizing the updated assemblies by selecting the most significant ones and discarding redundant assemblies
        if nns > 1: # checks if there is more than one updated assembly
            agglomeration_size = agglomeration_size + 1 # assembly order increses
            assembly = assembly_out
            del assembly_out

            na = len(assembly) # number of assemblies
            nelement = agglomeration_size + 1  # number of elements for assembly
            selection = np.full((na, nelement+1+1), np.nan)
            assembly_final = [[]]*na # max possible dimensions
            nns = 1

            for i in range(na):
                elem = np.sort(assembly[i]["elements"]) # retrieves the sorted indices of the neurons present in the current assembly
                indx, ism = np.where(np.isin(selection[:, 0:nelement], elem).all(axis=1, keepdims=True)) # checks if there is an existing assembly with the same set of neurons as the current assembly, "indx" stores the row indices where the condition is satisfied, and ism stores the column index
                if len(ism) > 0 and len(indx) > 0:
                    ism = ism.astype(int)[0]
                    indx = indx.astype(int)[0]
                else:
                    ism = -1
                    indx = -1
                if ism==-1:
                    # no matching assembly found
                    assembly_final[nns-1] = assembly[i] # current asseembly added to the final assembly
                    selection[nns-1,0:nelement] = elem # The neurons in the assembly are added to selection at the corresponding row
                    selection[nns-1,nelement] = assembly[i]['pvalue'][-1] # p-value of the assembly is stored in selection
                    selection[nns-1,nelement+1] = i # The index of the assembly in the assembly list is stored in selection
                    nns = nns+1
                else:
                    # If the p-value of the current assembly is smaller (more significant) than the existing matching assembly, it replaces the existing assembly in assembly_final and updates the corresponding significance value and index in selection.
                    if selection[indx,nelement] > assembly[i]['pvalue'][-1]: 
                        assembly_final[indx] = assembly[i]
                        selection[indx, nelement] = assembly[i]['pvalue'][-1]
                        selection[indx, nelement+1] = i
            assembly_final[nns-1:] = []
            assembly = assembly_final
            del assembly_final
        
        #del assemS2, assemS1
        
        fname =  'Assembly_0{}_b{}.mat'.format(agglomeration_size,gg)
        folder_path = path_project / 'thesis_code' / 'CAD' / 'output'
        file_path = os.path.join(folder_path, fname)
        scio.savemat(file_path, {'assembly': assembly}, format='5')

        bytesize = sys.getsizeof(assembly)
        if bytesize > bytelimit:
            print('The algorithm has been interrupted because assembly structures reached a global size of {} bytes, this limit can be changed in size or removed with the "bytelimit" option\n'.format(bytelimit))
            agglomeration_size = O_th
    
    maxOrder = agglomeration_size

    """_________________________pruning step 1____________________________"""
    # I remove assemblies whose elements are already ALL included in a bigger assembly

    nns = 1
    
    for o in range(0, maxOrder):
        fname = 'Assembly_0{}_b{}.mat'.format(o+1, gg)
        folder_path = path_project / 'thesis_code' / 'CAD' / 'output'
        file_path = os.path.join(folder_path, fname)
        assembly = scio.loadmat(file_path)['assembly']
        minor = assembly.copy()
        del assembly

        no = minor.shape[1]                      # number assemblies
        selection_o = np.ones(no, dtype=bool)
            
        for O in range(maxOrder, o+1, -1):
            fname = 'Assembly_0{}_b{}.mat'.format(O, gg)
            folder_path = path_project / 'thesis_code' / 'CAD' / 'output'
            file_path = os.path.join(folder_path, fname)
            assembly = scio.loadmat(file_path)['assembly']
            major = assembly.copy()
            del assembly
            
            nO = major.shape[1]                      # number assemblies
            
            index_elemo = np.where(selection_o == 1)[0]
            for i in range(sum(selection_o)):
                elemo = minor[0][index_elemo[i]][0][0]['elements']

                for j in range(nO):
                    elemO = major[0][j][0][0]['elements']
                    if np.isin(elemo, elemO).all():
                        selection_o[index_elemo[i]] = False
                        j = nO
                
                    
            if not np.any(selection_o):
                O = 0 
                
        index_elemo = np.where(selection_o == 1)[0]
        
        for i in range(sum(selection_o)):
            assembly_output.insert(nns-1,(minor[0][index_elemo[i]][0][0]))
            nns += 1

        # Turn off recycling of deleted files
        os.environ['RUBBISH_DISABLED'] = '1'

        # Define file name
        fname = f'Assembly_0{o}_b{gg}.mat'

        # Delete file if it exists
        if os.path.exists(fname):
                os.remove(fname)

        # Turn on recycling of deleted files
        os.environ['RUBBISH_DISABLED'] = '0'







    

    return assembly_output

def increase_matrix_size(matrix):
        num_rows = len(matrix)
        num_cols = len(matrix[0]) if num_rows > 0 else 0

        # Create a new matrix with an additional row
        new_matrix = [[0] * num_cols for _ in range(num_rows + 1)]

        # Copy the elements from the previous matrix
        for i in range(num_rows):
            for j in range(num_cols):
                new_matrix[i][j] = matrix[i][j]

        return np.array(new_matrix)