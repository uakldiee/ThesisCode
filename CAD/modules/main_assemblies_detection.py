
import numpy as np
import pandas as pd
import multiprocessing as mp
#from find_assemblies_recursive_optimized import find_assemblies_recursive
from modules.find_assemblies_recursive import find_assemblies_recursive

def main_assemblies_detection(spM,MaxLags,BinSizes,ref_lag = 2,alph = 0.05,No_th = 0,O_th = float('inf'),bytelimit = float('inf')):
    assembly= 0 # of course not
    """this function returns cell assemblies detected in spM spike matrix binned at a temporal 
    resolution specified in 'BinSizes' vector and testing for all lags between '-MaxLags(i)' 
    and 'MaxLags(i)'"""

    """ARGUMENTS:
        spM                  := matrix with population spike trains; each row is the spike train (time stamps, not binned) relative to a unit. 
        BinSizes             := vector of bin sizes to be tested;
        MaxLags              := vector of maximal lags to be tested. For a binning dimension of BinSizes(i) the program will test all pairs configurations with a time shift between -MaxLags(i) and MaxLags(i);
        (optional) ref_lag   := reference lag. Default value 2
        (optional) alph      := alpha level. Default value 0.05
        (optional) No_th     := minimal number of occurrences required for an assembly (all assemblies, even if significant, with fewer occurrences than No_th are discarded). Default value 0.
        (optional) O_th      := maximal assembly order (the algorithm will return assemblies of composed by maximum O_th elements).
        (optional) bytelimit := maximal size (in bytes) allocated for all assembly structures detected with a bin dimension. When the size limit is reached the algorithm stops adding new units."""
    
    
    nneu = spM.shape[0] # number of units
    assemblybin = [[]]*len(BinSizes)
    Dc=100 #length (in # bins) of the segments in which the spike train is divided to compute #abba variance (parameter k).

    for gg in range(len(BinSizes)):
        binSize = BinSizes[gg]
        maxlag = MaxLags[gg]
        print(f'{gg} - testing: bin size={binSize:.3f} sec; max tested lag={maxlag}')

        # binning
        tb = np.arange(np.nanmin(spM), np.nanmax(spM),binSize)
        binM = np.zeros((nneu, tb.shape[0] - 1), dtype=np.uint8)
        for n in range(nneu):
                binM[n,:],_ = np.histogram(spM[n,:], bins = tb)



        if binM.shape[1] - MaxLags[gg] < 100:
                print(f"Warning: testing bin size={binSize:.3f}%. The time series is too short, consider taking a longer portion of spike train or diminish the bin size to be tested")
        else:
                # Analysis
                assemblybin[gg] = {}
                assemblybin[gg]["n"] = find_assemblies_recursive(binM, maxlag, alph, gg, Dc, No_th, O_th, bytelimit, ref_lag)
                if assemblybin[gg]["n"]:
                       assemblybin[gg]['bin_edges'] = tb
                        
                print(f"{gg} - testing done")
                fname = f"assembly{gg}.mat"
                #parsave(fname, assemblybin[gg])



    assembly = {}
    assembly['bin'] = assemblybin
    assembly['parameters'] = {'alph': alph, 'Dc': Dc, 'No_th': No_th, 'O_th': O_th, 'bytelimit': bytelimit}


    #def parsave(fname, aus):
    #    np.save(fname, aus)






    return assembly