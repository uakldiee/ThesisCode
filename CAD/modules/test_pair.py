import numpy as np
import math
from scipy.stats import f

def test_pair(ensemble,spikeTrain2,n2,maxlag,Dc,reference_lag):
    assemD = 0 # of course not
    """this function tests if the two spike trains have repetitive patterns occurring more frequently than chance."""
    """ ensemble := structure with the previously formed assembly and its spike train
        spikeTrain2 := spike train of the new unit to be tested for significance (candidate to be a new assembly member)
        n2 := new unit tested
        maxlag := maximum lag to be tested
        Dc := size (in bins) of chunks in which I divide the spike trains to compute the variance (to reduce non stationarity effects on variance estimation)
        reference_lag := lag of reference; if zero or negative reference lag=-l
    """
    # spike train pair I am going to test
    # minimum value is subtracted from each element - used to shift the time axis so that the first event occurs at time 0 ??
    # check the dimensions and type of ensemble.Time!!!!
    # couple = np.concatenate((ensemble.Time - np.min(ensemble.Time), spikeTrain2 - np.min(spikeTrain2)), axis=1) 
    ensamble_time = ensemble['times'] - np.min(ensemble['times'])
    spike_train2 = spikeTrain2 - np.min(spikeTrain2) # move outside
    #if np.min(spikeTrain2) != 0:
        #print(np.min(spikeTrain2))
    couple = np.hstack((ensamble_time.reshape(-1,1),spike_train2.reshape(-1,1)))
    nu = 2
    ntp = couple.shape[0] # trial length

    """Split the spike count series into a set of binary processes"""
    maxrate = np.max(couple)

    # creates a list of maxrate NumPy arrays, where each array has the same shape as couple and is initialized with all zeros.
    # creation of the parallel processes, one for each rate up to maxrate
    # and computation of the coincidence count for both neurons
    Zaa = [np.zeros_like(couple, dtype=np.uint8) for i in range(maxrate)]
    ExpABi = np.zeros(maxrate) # what is it for?? added correction for continuity approximation (ExpAB limit and Yates correction) ???
    
    for i in range(1, maxrate+1):
        Zaa[i-1][couple >= i] = 1 
        # rest is used to later correct for continuity approximation
        col_sums = np.sum(Zaa[i-1], axis=0) # essentialy the sum of the '1's in the binary subprocess
        ExpABi[i-1] = np.prod(col_sums) / couple.shape[0] # product of the sum of the '1's in the binary subprocess rate i-1 of the couple to be tested devided by the number of bins??


    """Get the best lag - the lag with the most coincidences"""
    """from the range of all considered lags we choose the one which corresponds to the highest
       count #AB,l_"""
    # structure with the coincidence counts for each lag
    ctAB = np.empty(maxlag + 1) # counts for positive lags
    ctAB.fill(np.nan)
    ctAB_ = np.empty(maxlag + 1) # counts for negative lags
    ctAB_.fill(np.nan)

    # # Loop over lags from 0 to maxlag
    for lag in range(0,maxlag+1):
        # horizontal concatenation
        trAB = np.vstack((couple[:couple.shape[0]-(maxlag),0], couple[lag:couple.shape[0]-(maxlag)+lag,1])).T
        trBA = np.vstack((couple[lag:couple.shape[0]-(maxlag)+lag,0],couple[:couple.shape[0]-(maxlag),1])).T
        # MATLAB uses the apostrophe operator (') to perform a complex conjugate transpose
        ctAB[lag] = np.nansum(np.nanmin(trAB,axis=1))
        ctAB_[lag] = np.nansum(np.nanmin(trBA,axis=1))
    
    if reference_lag <= 0:
        # vertical concatenation
        aus = np.vstack((ctAB, ctAB_))
        flattened_aus = aus.flatten('F')
        a = np.max(flattened_aus) # a contains maximum value in the flattened array aus
        b = np.argmax(flattened_aus) # b contains the flattened index of the maximum value in the array aus
        m, n = aus.shape
        I, J = np.unravel_index(b,(m, n),'F') # converts the linear index b to the corresponding row-column indices I and J in the matrix aus
        l_ =  ((I==0) * ((J-1)+1) ) - ((I==1) * ((J-1)+1)) # plus one added so that the value is the same as in the matlab code
    else:   
        Hab_l = np.hstack((ctAB_[1:][::-1], ctAB))
        a = np.max(Hab_l)
        b = np.argmax(Hab_l)
        lags = np.arange(-maxlag, maxlag+1)
        l_ = lags[b]
        Hab = Hab_l[b]

        if l_ < 0:
            l_ref = l_ + reference_lag
            Hab_ref = Hab_l[np.where(lags==l_ref)[0][0]]
        else:
            l_ref = l_ - 2
            Hab_ref = Hab_l[np.where(lags==l_ref)[0][0]]
        
    ExpAB = sum(ExpABi)
    if a==0 or ExpAB <=5 or ExpAB >=(min(sum(couple[:,0]),sum(couple[:,1]))-5):
        assemD = {
                'elements': np.hstack((ensemble['elements'], n2)),
                'lag': np.hstack((ensemble['lag'], 99)),
                'pvalue': np.hstack((ensemble['pvalue'], 1)),
                'times': [],
                'n_occurences': np.hstack((ensemble['n_occurences'], 0))
        }
    else:
        len = couple.shape[0] # trial length
        time = np.concatenate(np.zeros((len,1), dtype=np.uint8), axis= 0)
        if reference_lag <= 0:
            if l_ == 0:
                for i in range(maxrate):
                    sA = Zaa[i][:,0]
                    sB = Zaa[i][:,1]
                    time[0:len] = time[0:len] + sA[0:len]*sB[0:len]
                TPrMTot = np.array([[0, ctAB[0]], [ctAB_[2], 0]])
            elif l_ > 0:
                for i in range(maxrate):
                    sA = Zaa[i][:,0]
                    sB = Zaa[i][:,1]
                    time[0:len-l_] = time[0:len-l_] + sA[0:len-l_]*sB[l_:len]
                TPrMTot = np.array([[0, ctAB[J]], [ctAB_[J], 0]])
            else:
                for i in range(maxrate):
                    sA = Zaa[i][:,0]
                    sB = Zaa[i][:,1]
                    time[-l_:] = time[-l_:] + sA[-l_:]*sB[0:len+l_]
                TPrMTot = np.array([[0, ctAB[J]], [ctAB_[J], 0]])
        else:
            if l_ == 0:
                for i in range(maxrate):
                    sA = Zaa[i][:,0]
                    sB = Zaa[i][:,1]
                    time[0:len] = time[0:len] + sA[0:len]*sB[0:len]
            elif l_ > 0:
                for i in range(maxrate):
                    sA = Zaa[i][:,0]
                    sB = Zaa[i][:,1]
                    time[0:len-l_] = time[0:len-l_] + sA[0:len-l_]*sB[l_:len]
            else:
                for i in range(maxrate):
                    sA = Zaa[i][:,0]
                    sB = Zaa[i][:,1]
                    time[-l_:len] = time[-l_:len]+sA[-l_:len]*sB[0:len+l_]
            TPrMTot = np.array([[0, Hab], [Hab_ref, 0]])

        

        """cut the spike train in stationary segments"""
        nch = math.ceil((couple.shape[0] - maxlag) / Dc)
        Dc = math.floor((couple.shape[0] - maxlag) / nch) # new chunk size, this is to have all chunks of rougly the same size
        chunked = [[]] * nch
        couple_cut = np.full(((couple.shape[0] - maxlag), 2),np.nan)
        if l_ == 0:
            couple_cut = couple[0:len-maxlag, :]
        elif l_ > 0:
            couple_cut[:, 0] = couple[0:len-maxlag, 0]
            couple_cut[:, 1] = couple[l_:len-(maxlag)+l_, 1]
        else:
            couple_cut[:, 0] = couple[-l_:len-maxlag-l_, 0]
            couple_cut[:, 1] = couple[0:len-maxlag, 1]

        for iii in range(0,nch):
            chunked[iii] = couple_cut[(1+Dc*(iii)-1):Dc*(iii+1), :]
        chunked[nch-1] = couple_cut[(Dc*(nch-1)):, :] # last chunk can be of slightly different size

        
        """___________________________________________"""

        MargPr_t = [[[] for _ in range(maxrate)] for _ in range(nch)]
        maxrate_t = np.empty(nch)
        maxrate_t.fill(np.nan)
        ch_nn = np.empty(nch) 
        ch_nn.fill(np.nan)
        
        

        for iii in range(nch):
            couple_t = chunked[iii]
            maxrate_t[iii] = np.max(couple_t)
            ch_nn[iii] = couple_t.shape[0]
            Zaa_t = [[]]*int(maxrate_t[iii])
        
            for i in range(0, int(maxrate_t[iii])):
                    Zaa_t[i] = np.zeros_like(couple_t, dtype=np.uint8)
                    Zaa_t[i][couple_t >= i+1] = 1

            ma = int(maxrate_t[iii])
            for i in range(1, ma+1):
                    sA = Zaa_t[i-1][:, 0]
                    sB = Zaa_t[i-1][:, 1]
                    MargPr_t[iii][i-1] = [np.sum(sA), np.sum(sB)]
        
        """______________chunks_________________"""
        

        n = ntp - maxlag
        Mx0 = [[[]]*maxrate for _ in range(nch)]
        covABAB = [[]]*nch
        covABBA = [[]]*nch
        varT = [[]]*nch
        covX = [[]]*nch
        varX = [[]]*nch
        varXtot = np.zeros((2,2))

        for iii in range(0,nch):
            maxrate_t = int(np.max(chunked[iii]))
            ch_n = ch_nn[iii]
            # evaluation of #AB
            for i in range(0,maxrate_t):
            # checks if MargPr_t[iii][i] is not empty
                    if  MargPr_t[iii][i]:
                            temp = MargPr_t[iii][i]*np.ones((2,1))
                            Mx0[iii][i] = temp.T

            varT[iii] = np.zeros((nu,nu))
            covABAB[iii]= np.empty((maxrate_t, maxrate_t), dtype=object)
            for i in range(0,maxrate_t):
                    temp = MargPr_t[iii][i]*np.ones((2,1))
                    Mx0[iii][i] = temp.T
                    covABAB[iii][i][i] = ((Mx0[iii][i]*Mx0[iii][i].T/ch_n)*(ch_n - Mx0[iii][i])*
                                            (ch_n - Mx0[iii][i]).T / (ch_n*(ch_n - 1)))
                    varT[iii] = varT[iii]+covABAB[iii][i][i]
                    for j in range(i+1,maxrate_t):
                            covABAB[iii][i][j] = (2*(Mx0[iii][j]*Mx0[iii][j].T/ch_n)*
                                            (ch_n - Mx0[iii][i])*
                                            (ch_n - Mx0[iii][i].T)/(ch_n*(ch_n-1)))
                            varT[iii] = varT[iii]+covABAB[iii][i][j]
            
            # evaluation of X = #AB - #BA
            covX[iii] = np.zeros((nu,nu))
            covABBA[iii] = np.empty((maxrate_t, maxrate_t), dtype=object)
            for i in range(0, maxrate_t):
                    covABBA[iii][i][i] = ((Mx0[iii][i]*Mx0[iii][i].T/ch_n)*
                                            (ch_n - Mx0[iii][i])*
                                            (ch_n - Mx0[iii][i]).T / (ch_n*(ch_n-1)**2))
                    covX[iii] = covX[iii]+covABBA[iii][i][i]
                    for j in range(i+1,maxrate_t):
                            covABBA[iii][i][j]=(2*(Mx0[iii][j]*Mx0[iii][j].T/ch_n)*
                                            (ch_n-Mx0[iii][i])*
                                            (ch_n-Mx0[iii][i]).T / (ch_n*(ch_n-1)**2))
                            covX[iii] = covX[iii] + covABBA[iii][i][j]
            varX[iii] = varT[iii] + varT[iii].T - covX[iii]-covX[iii].T
            varXtot = varXtot+varX[iii]

        # everything before here revised

        """____________________________________________"""
        X = TPrMTot-TPrMTot.T
        if np.abs(X[0,1]) > 0:
            X = np.abs(TPrMTot-TPrMTot.T) - 0.5 # yates correction
        
        if varXtot[0,1] == 0:
            prF = 1
        else:
            F = X**2 /varXtot
            prF = f.sf(F[0,1],1,n)

        """______________________________________________"""
        # All information about the assembly and test are returned
        assemD = {
                'elements': np.hstack((ensemble['elements'], n2)),
                'lag': np.hstack((ensemble['lag'], [l_])),
                'pvalue': np.hstack((ensemble['pvalue'], [prF])),
                'times': time,
                'n_occurences': np.hstack((ensemble['n_occurences'], np.sum(time)))
        }
    return assemD
