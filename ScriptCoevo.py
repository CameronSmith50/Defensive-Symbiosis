#%% Packages

from Virulence_Coevo import *
import os

#%% Filepaths

dataPath = './Datasets/'
figPath = './Figures/'

#%% 1. Introducing a mutualist results in the parasite becoming
#      more virulent, regardless of its protection level. 
dataPathRes1 = dataPath + 'Result1/'
figPathRes1 = figPath + 'Result1/'

# Result 1: Evolved virulence over y
# print('1')
# singleMutSweep(ny=51, c1Vec=[0.2, 0.5, 0.8], delta=0, singleColumn=False)

#%% 2. The mutualist can cause diversification in the parasite 
#      population if it is fully mutualistic.
dataPathRes2 = dataPath + 'Result2/'
figPathRes2 = figPath + 'Result2/'  

# Function for result 2
def Res2(c1Vals = [0.45, 0.5, 0.55], delta=0):
    '''
    Code to generate the second figure of the manuscript.
    Takes in the c1Vals that should be simulated over.
    Creates an evoSim and PIP for each yTrait value, and outputs a combined figure in the Figures directory.
    Also generates data, which must be cleared between successive runs.
    '''

    # Number of c1Vals
    nc1 = len(c1Vals)

    # Set up a figure
    paperwidth = 12
    paperheight = 15
    fig = plt.figure(figsize=(paperwidth, paperheight))
    gs = fig.add_gridspec(2,nc1)
    axs = [None] * (2*nc1)
    matplotlib.rcParams.update({'font.size': 18})
    
    # Create inset figure size and vector
    loc = [0.6, 0.12, 0.35, 0.2]
    axsI = [None] * (nc1)

    # Loop through the length of nTraits
    for ii in range(nc1):

        print('evoSim for c1 = %s...' % c1Vals[ii])

        # Create a simulation
        sim = Simulation(nTraity=1, ytraitMin=1, ytraitMax=1, nTraitb2=101, b2traitMax=20, c1=c1Vals[ii], ntEvo=1500, delta=delta)
        
        # Datapath
        dp = dataPathRes2 + 'Res2_' + str(ii) + '/'

        # EvoSim
        sim.evoSim(dp, saveData=True)

        # Plot evoSim
        createdFile = [f for f in os.listdir(dp) if os.path.isfile(os.path.join(dp, f))][-1][:-4]
        [_, IIb2] = plotOverParams(dp, figPathRes1)
        ax = fig.add_subplot(gs[0, ii])
        ax.pcolormesh(sim.b2Traits, np.arange(sim.ntEvo+1), np.log(IIb2), cmap='Greys')

        # Remove all axis labels and ticks
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([0, sim.b2traitMax/2, sim.b2traitMax])
        ax.set_yticks([0, int(sim.ntEvo/2), sim.ntEvo])
        ax.set_xticklabels(['','',''])
        ax.set_yticklabels(['','',''])

        print('DONE')
        print('Generating PIPs for c1 = %s...' % c1Vals[ii])

        # Save the axis to the list
        axs[2*ii] = ax

        # Generate the PIPs
        n = 101
        [PIPMut, PIPPar] = sim.plotPIPs(yRes = 1, b2Res=None, n=n)
        bb = np.linspace(sim.b2traitMin, sim.b2traitMax, n)
        axI = ax.inset_axes(loc)
        axI.pcolormesh(bb, bb, PIPPar > 1e-6, cmap='Greys')

        # Remove all axis labels and ticks
        axI.set_xlabel(r'$\beta_P^r$', fontsize=14)
        axI.set_ylabel(r'$\beta_P^m$', fontsize=14)
        axI.set_xticks([0, sim.b2traitMax/2, sim.b2traitMax])
        axI.set_yticks([0, sim.b2traitMax/2, sim.b2traitMax])
        axI.tick_params(axis='both', which='major', labelsize=14)

        # Save the plot to the axis list
        axsI[ii] = axI

        print('DONE')
        print('Calculating proportion of coinfections for c1 = %s...' % c1Vals[ii])

        # Adding the final plot
        ax = fig.add_subplot(gs[-1,ii])

        # Extract necessary values
        file = open(dp + createdFile + '.pkl', 'rb')
        pdict = pickle.load(file)
        file.close()
        sim = pdict['self']
        b2Traits = pdict['b2TraitListMat']
        store = pdict['store']

        # Preallocate arrays
        I2VecLower = np.zeros(sim.ntEvo)
        I2VecUpper = np.zeros(sim.ntEvo)
        I12VecLower = np.zeros(sim.ntEvo)
        I12VecUpper = np.zeros(sim.ntEvo)

        # Calculate the proportion of coinfections amongst all parasite infections
        # Loop through time
        for nn in range(0,sim.ntEvo):

            # Identify the traits that exist at this time
            BB = b2Traits[nn,]

            # Check if there is a branch. This will be defined as there being a gap in the trait index vector
            dBB = np.diff(BB)
            nBranch = int(np.sum(np.abs(dBB))/2*(np.sum(np.abs(dBB)) % 2 == 0) + (np.sum(np.abs(dBB))+1)/2*(np.sum(np.abs(dBB)) % 2 == 1))

            # Final check, if nBranch is calculated as 1, there could still be 2 branches if they are at each end of the trait space
            if BB[0] == 1 and BB[-1] == 1 and nBranch == 1:
                nBranch = 2
            
            # Calculate the indices for each of the branches
            if nBranch == 1:
                b2Lower = BB*(np.arange(0,sim.nTraitb2)+1)
                b2Lower = (b2Lower[b2Lower > 0] - 1).astype('int')
                b2Upper = b2Lower.copy()
            elif nBranch == 2:
                indLower = dBB*(np.arange(0,sim.nTraitb2-1)+1)
                indLower = indLower[indLower>0]-1
                if len(indLower) == 1:
                    indLower = np.concatenate((np.array([0]), indLower))
                indUpper = dBB*(np.arange(0,sim.nTraitb2-1)+1)
                indUpper = -indUpper[indUpper<0]-1
                if len(indUpper) == 1:
                    indUpper = np.concatenate((indUpper, np.array([sim.nTraitb2-1])))
                b2Lower = np.arange(indLower[0], indUpper[0]+1).astype('int')
                b2Upper = np.arange(indLower[1], indUpper[1]+1).astype('int')

            # Calculate the numbers in each branch
            I2 = store[nn, (1+sim.nTraity):(1+sim.nTraity+sim.nTraitb2)]
            I12 = np.sum(np.reshape(store[nn, (1+sim.nTraity+sim.nTraitb2):], (sim.nTraity, sim.nTraitb2)), axis=0)
            I2VecLower[nn] = np.sum(I2[b2Lower])
            I2VecUpper[nn] = np.sum(I2[b2Upper])
            I12VecLower[nn] = np.sum(I12[b2Lower])
            I12VecUpper[nn] = np.sum(I12[b2Upper])
            
        # Plot
        ax.plot(I12VecUpper[1:]/(I2VecUpper[1:]+I12VecUpper[1:]), np.arange(1, sim.ntEvo), 'r', lw=2)
        ax.plot(I12VecLower[1:]/(I2VecLower[1:]+I12VecLower[1:]), np.arange(1, sim.ntEvo), 'k', lw=2)

        # Remove all axis labels and ticks
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticks([0, int(sim.ntEvo/2), sim.ntEvo])
        ax.set_yticklabels(['','',''])
        ax.set_ylim([0, sim.ntEvo])

        # Save the plot to the axis list
        axs[2*ii+1] = ax


    print('Resolving axis labels...')

    # xLabels
    for jj in range(0,2*nc1,2):
        axs[jj].set_xticklabels([0, int(sim.b2traitMax/2), sim.b2traitMax])
    axs[2].set_xlabel(r'Parasite transmission, $\beta_P$')
    axs[3].set_xlabel('Proportion of parasitised hosts that posess defensive symbionts')

    # yLabels
    axs[0].set_yticklabels([0, int(sim.ntEvo/2), sim.ntEvo])
    axs[1].set_yticklabels([0, int(sim.ntEvo/2), sim.ntEvo])
    axs[0].set_ylabel('Evo. time')
    axs[1].set_ylabel('Evo. time')

    # Titles
    for kk in range(0, 2*nc1, 2):
        axs[kk].set_title(r'$c_1=%s$' % str(c1Vals[int(kk/2)]))

    print('DONE')

    # Save the figure 
    fig.savefig(figPathRes2 + 'Result2.pdf')    

# # Example 2:
print('2')
Res2(c1Vals=[0.4,0.5,0.6], delta=0)

#%% 3. Mutualists can evolve to not help the host, leading to 
#      a higher virulence parasite and a mutualist with a (small) 
#      added virulence. In some cases, this leads to the mutualist 
#      being a detriment to the host population.
dataPathRes3 = dataPath + 'Result3/'
figPathRes3 = figPath + 'Result3/'

def Res3b(dataDir, saveDir, saveName):
    '''
    Saves data that has been created by the function newEvoSimGif.
    Requires the loaction of the data and a location to save the combined figure.
    '''

    # Look in the data directory, list the files and find the number of them
    filesInDir = [f for f in os.listdir(dataDir) if os.path.isfile(os.path.join(dataDir, f))]
    nFiles = len(filesInDir)

    # Create lists to contain each of the datasets
    deathList = [None] * nFiles
    PopList = [None] * nFiles
    yLowList = [None] * nFiles
    yUpList = [None] * nFiles
    bLowList = [None] * nFiles
    bUpList = [None] * nFiles

    # Loop through each of the datasets
    for ii in range(nFiles):

        # Load the directory
        file = open(dataDir + filesInDir[ii], 'rb')
        pdict = pickle.load(file)
        file.close()
        deathList[ii] = pdict['deathRateMat']
        PopList[ii] = pdict['NMat']
        yLowList[ii] = pdict['yL']
        yUpList[ii] = pdict['yU']
        bLowList[ii] = pdict['bL']
        bUpList[ii] = pdict['bU']
        yT = pdict['yT']
        bT = pdict['bT']
        cmap = pdict['cmap']
        cmapalt = pdict['cmapalt']

    # Figure 1: Same colourmap for each
    plotWidth = 11.69
    plotHeight = 8.27
    fsize = 10
    fig1 = plt.figure(figsize=(plotWidth, plotHeight))
    ax11 = fig1.add_subplot(241)
    ax12 = fig1.add_subplot(242)
    ax13 = fig1.add_subplot(243)
    ax14 = fig1.add_subplot(244)
    ax15 = fig1.add_subplot(245)
    ax16 = fig1.add_subplot(246)
    ax17 = fig1.add_subplot(247)
    ax18 = fig1.add_subplot(248)
    axs = [ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18]

    # Loop through the axes
    for ii in range(8):
        if ii % 2 == 0:
            jj = int(np.round(ii/2))
            axs[ii].contourf(yT, bT, 100*PopList[jj], levels=[-50,-25,0,25,50], extend='both', cmap=cmap, alpha=0.3)
            contour = axs[ii].contour(yT, bT, 100*PopList[jj], levels=[-50,-25,0,25,50], colors='dimgray')
            axs[ii].clabel(contour, inline=True, fontsize=fsize)
            axs[ii].plot([0,1], [bLowList[jj][0,0], bLowList[jj][0,0]], 'k--')
            # axs[ii].set_title('Population size', fontsize=fsize)
            for n in range(3):
                if not ((ii == 6 or ii == 7) and n == 2):
                    axs[ii].plot(yLowList[jj][:,n], bLowList[jj][:,n], 'k', lw=2)
                    axs[ii].plot(yUpList[jj][:,n], bUpList[jj][:,n], 'k', lw=2)
                if ii == 0 or ii == 1:
                    axs[ii].plot(yLowList[jj][0,n], bLowList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yLowList[jj][-1,0], bLowList[jj][-1,0], 'r.', ms=18, clip_on=False)
                    axs[ii].plot(yUpList[jj][0,n], bUpList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yUpList[jj][-1,0], bUpList[jj][-1,0], 'r.', ms=18, clip_on=False)
                elif ii == 4 or ii == 5:
                    axs[ii].plot(yLowList[jj][0,n], bLowList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yLowList[jj][-1,2*n*(n<=1)+n*(n==2)], bLowList[jj][-1,2*n*(n<=1)+n*(n==2)], 'r.', ms=18, clip_on=False)
                    axs[ii].plot(yUpList[jj][0,n], bUpList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yUpList[jj][-1,2*n*(n<=1)+n*(n==2)], bUpList[jj][-1,2*n*(n<=1)+n*(n==2)], 'r.', ms=18, clip_on=False)
                elif (ii == 6 or ii == 7) and n == 2:
                    axs[ii].plot(yLowList[jj][0,n], bLowList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yLowList[jj][0,n], bLowList[jj][0,n], 'rx', ms=12, lw=2, clip_on=False)
                else:
                    axs[ii].plot(yLowList[jj][0,n], bLowList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yLowList[jj][-1,n], bLowList[jj][-1,n], 'r.', ms=18, clip_on=False)
                    axs[ii].plot(yUpList[jj][0,n], bUpList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yUpList[jj][-1,n], bUpList[jj][-1,n], 'r.', ms=18, clip_on=False)
            axs[ii].set_xlabel(r'$y$ (mutualism strength)', fontsize=fsize)
            axs[ii].set_ylabel(r'$\beta_P$ (parasite transmission)', fontsize=fsize)
            axs[ii].tick_params(axis='x', labelsize=fsize)
            axs[ii].tick_params(axis='y', labelsize=fsize)
        else:
            jj = int(np.round((ii-1)/2))
            axs[ii].contourf(yT, bT, gaussian_filter(100*deathList[jj],1.5), levels=[-20,-10,0,10,20], extend='both', cmap=cmap, alpha=0.3)
            contour = axs[ii].contour(yT, bT, gaussian_filter(100*deathList[jj],1.5), levels=[-20,-10,0,10,20], colors='dimgray')
            axs[ii].clabel(contour, inline=True, fontsize=fsize)
            axs[ii].plot([0,1], [bLowList[jj][0,0], bLowList[jj][0,0]], 'k--')
            # axs[ii].set_title('Death rate', fontsize=fsize)
            for n in range(3):
                if not ((ii == 6 or ii == 7) and n == 2):
                    axs[ii].plot(yLowList[jj][:,n], bLowList[jj][:,n], 'k', lw=2)
                    axs[ii].plot(yUpList[jj][:,n], bUpList[jj][:,n], 'k', lw=2)
                if ii == 0 or ii == 1:
                    axs[ii].plot(yLowList[jj][0,n], bLowList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yLowList[jj][-1,0], bLowList[jj][-1,0], 'r.', ms=18, clip_on=False)
                    axs[ii].plot(yUpList[jj][0,n], bUpList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yUpList[jj][-1,0], bUpList[jj][-1,0], 'r.', ms=18, clip_on=False)
                elif ii == 4 or ii == 5:
                    axs[ii].plot(yLowList[jj][0,n], bLowList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yLowList[jj][-1,2*n*(n<=1)+n*(n==2)], bLowList[jj][-1,2*n*(n<=1)+n*(n==2)], 'r.', ms=18, clip_on=False)
                    axs[ii].plot(yUpList[jj][0,n], bUpList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yUpList[jj][-1,2*n*(n<=1)+n*(n==2)], bUpList[jj][-1,2*n*(n<=1)+n*(n==2)], 'r.', ms=18, clip_on=False)
                elif (ii == 6 or ii == 7) and n == 2:
                    axs[ii].plot(yLowList[jj][0,n], bLowList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yLowList[jj][0,n], bLowList[jj][0,n], 'rx', ms=12, lw=2, clip_on=False)
                else:
                    axs[ii].plot(yLowList[jj][0,n], bLowList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yLowList[jj][-1,n], bLowList[jj][-1,n], 'r.', ms=18, clip_on=False)
                    axs[ii].plot(yUpList[jj][0,n], bUpList[jj][0,n], 'g.', ms=18, clip_on=False)
                    axs[ii].plot(yUpList[jj][-1,n], bUpList[jj][-1,n], 'r.', ms=18, clip_on=False)
            axs[ii].set_xlabel(r'$y$ (mutualism strength)', fontsize=fsize)
            axs[ii].set_ylabel(r'$\beta_P$ (parasite transmission)', fontsize=fsize)
            axs[ii].tick_params(axis='x', labelsize=fsize)
            axs[ii].tick_params(axis='y', labelsize=fsize)

    plt.tight_layout()
    plt.savefig(saveDir + saveName + '_allSameCmap.pdf')

    plt.show()

# # Result 3a: Heatmap
# n = (10*2**np.arange(6)+1).astype('int')  # Number of elements at each level
# delta = 0  # Tolerance or resistance
# print('3a')

# Initialisation and refinement of classification
# heatmapData(dataPathRes3 + 'Ex3a/y0l0/', yInit=0, n=n[0], delta=delta)  # Initial level for yInit=0
# heatmapData(dataPathRes3 + 'Ex3a/y1l0/', yInit=1, n=n[0], delta=delta)  # Initial level for yInit=1
# initialDataframeClassify(dataPathRes3 + 'Ex3a/y0l0/', dataPathRes3 + 'Ex3a/y1l0/', dataPathRes3 + 'Ex3a/', 'Ex3aData.csv')  # Initial dataframe
# refinementStep(dataPathRes3 + 'Ex3a/Ex3aData.csv', 1, np.linspace(0, 1, n[0]), np.linspace(-5, 5, n[0]), dataPathRes3 + 'Ex3a/y0l1/', dataPathRes3 + 'Ex3a/y1l1/', delta=delta)  # Refinement level 1
# refinementStep(dataPathRes3 + 'Ex3a/Ex3aData.csv', 2, np.linspace(0, 1, n[1]), np.linspace(-5, 5, n[1]), dataPathRes3 + 'Ex3a/y0l2/', dataPathRes3 + 'Ex3a/y1l2/')  # Refinement level 2
# refinementStep(dataPathRes3 + 'Ex3a/Ex3aData.csv', 3, np.linspace(0, 1, n[2]), np.linspace(-5, 5, n[2]), dataPathRes3 + 'Ex3a/y0l3/', dataPathRes3 + 'Ex3a/y1l3/')  # Refinement level 3
# refinementStep(dataPathRes3 + 'Ex3a/Ex3aData.csv', 4, np.linspace(0, 1, n[3]), np.linspace(-5, 5, n[3]), dataPathRes3 + 'Ex3a/y0l4/', dataPathRes3 + 'Ex3a/y1l4/')  # Refinement level 4
# refinementStep(dataPathRes3 + 'Ex3a/Ex3aData.csv', 5, np.linspace(0, 1, n[4]), np.linspace(-5, 5, n[4]), dataPathRes3 + 'Ex3a/y0l5/', dataPathRes3 + 'Ex3a/y1l5/')  # Refinement level 5

# Plotting
# plotClassify(dataPathRes3 + 'Ex3a/Ex3aData.csv')

# Example 3b: Heatmaps

# delta = 0  # Tolerance or resistance
# print('3b')

# # c1 = 0.4, c2 = 2
# sim = Simulation(c1 = 0.4, c2 = 2, yInit = 0.1, delta=delta)
# sim.evoSim(dataPathRes3 + 'Ex3b/rawData11New/', saveData = True)
# sim = Simulation(c1 = 0.4, c2 = 2, yInit = 0.5, delta=delta)
# sim.evoSim(dataPathRes3 + 'Ex3b/rawData11New/', saveData = True)
# sim = Simulation(c1 = 0.4, c2 = 2, yInit = 0.9, delta=delta)
# sim.evoSim(dataPathRes3 + 'Ex3b/rawData11New/', saveData = True)

# # c1 = 0.4, c2 = -2
# sim = Simulation(c1 = 0.4, c2 = -2, yInit = 0.1, delta=delta)
# sim.evoSim(dataPathRes3 + 'Ex3b/rawData21New/', saveData = True)
# sim = Simulation(c1 = 0.4, c2 = -2, yInit = 0.5, delta=delta)
# sim.evoSim(dataPathRes3 + 'Ex3b/rawData21New/', saveData = True)
# sim = Simulation(c1 = 0.4, c2 = -2, yInit = 0.9, delta=delta)
# sim.evoSim(dataPathRes3 + 'Ex3b/rawData21New/', saveData = True)

# # c1 = 0.9, c2 = 2
# sim = Simulation(c1 = 0.9, c2 = 2, yInit = 0.1, delta=delta)
# sim.evoSim(dataPathRes3 + 'Ex3b/rawData12New/', saveData = True)
# sim = Simulation(c1 = 0.9, c2 = 2, yInit = 0.5, delta=delta)
# sim.evoSim(dataPathRes3 + 'Ex3b/rawData12New/', saveData = True)
# sim = Simulation(c1 = 0.9, c2 = 2, yInit = 0.9, delta=delta)
# sim.evoSim(dataPathRes3 + 'Ex3b/rawData12New/', saveData = True)

# # c1 = 0.9, c2 = -2
# sim = Simulation(c1 = 0.9, c2 = -2, yInit = 0.1, delta=delta)
# sim.evoSim(dataPathRes3 + 'Ex3b/rawData22New/', saveData = True)
# sim = Simulation(c1 = 0.9, c2 = -2, yInit = 0.5, delta=delta)
# sim.evoSim(dataPathRes3 + 'Ex3b/rawData22New/', saveData = True)
# sim = Simulation(c1 = 0.9, c2 = -2, yInit = 0.9, delta=delta)
# sim.evoSim(dataPathRes3 + 'Ex3b/rawData22New/', saveData = True)

# # Create individual plots and save the required data
# newEvoSimGif(dataPathRes3 + 'Ex3b/rawData11New/', dataPathRes3 + 'Ex3b/IndPlots/', 0)
# newEvoSimGif(dataPathRes3 + 'Ex3b/rawData12New/', dataPathRes3 + 'Ex3b/IndPlots/', 1)
# newEvoSimGif(dataPathRes3 + 'Ex3b/rawData21New/', dataPathRes3 + 'Ex3b/IndPlots/', 2)
# newEvoSimGif(dataPathRes3 + 'Ex3b/rawData22New/', dataPathRes3 + 'Ex3b/IndPlots/', 3)

# # Plotting
# Res3b(dataPathRes3 + 'Ex3b/IndPlots/', figPathRes3 + 'Ex3b/processedPlots/', '')