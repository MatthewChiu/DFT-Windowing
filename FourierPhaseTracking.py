# Windowing DFT Code with log-weight for specific pieces
# Created by Matt G. Chiu — Eastman School of Music
# AND
# Jenn Harding
# mchiu9@u.rochester.edu / matthewgychiu@gmail.com (860)682–3832

from music21 import *
import numpy as np
from scipy.fftpack import fft
import math
import pandas as pd


def get_Data(score):
    score = score.stripTies(retainContainers=False)
    fullData = []
    for i in score.flat:
        check1 = isinstance(i, chord.Chord)
        check2 = isinstance(i, note.Note)
        check3 = isinstance(i, note.Rest)
        if check1 == True:
            fullData.append([i.pitchClasses, i.offset])
        elif check2 == True:
            fullData.append([i.pitch.pitchClass, i.offset])
    return fullData

def window(data, size):
    data_len = len(data)
    outputData = []
    newData = []
    for i in range(0, data_len):
        window_end = i + size
        index_end = 0
        for c, item in enumerate (data, index_end):
            if i <= item[1] <= window_end:
                x = item[0]
                if isinstance(x, (float, int)):
                    newData.append(x)
                else:
                    newData.extend(x)
            elif item[1] > window_end:
                outputData.append(newData)
                index_end = c-1
                newData = []
                break
        i = i+1
    return outputData


def storage(window_list):
    collection = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    PCStorage = []
    index = -1
    for w in window_list:
        index += 1
        PCStorage.append(collection)
        for x in w:
            PCStorage[index][x] += 1
        collection = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return PCStorage


# Takes the storage information and weights it by Log2 (weighting taken from Matt Chiu).
def log_weight(stored_pcdata):
    for counter, array in enumerate(stored_pcdata):
        for counter2, element in enumerate(array):
            array[counter2] = math.log(element+1, 2)
    return stored_pcdata


# Takes the weighted information, performs the DFT, then grabs the magnitude and phase of each array. Returns a tuple containing lists of lists (magnitues and phases).
def make_arrays(weighted_data):
    DFTArray = []
    magnitudeArray = []
    phaseArray = []
    for x in weighted_data:
        DFTArray.append(np.fft.fft(x)) 
    for eachArray in DFTArray:
        magnitudeArrayGrabber = np.absolute(eachArray)
        magnitudeArray.append(magnitudeArrayGrabber)
        phaseArrayGrabber = np.angle(eachArray)
        phaseArray.append(phaseArrayGrabber)
    return (magnitudeArray, phaseArray, DFTArray)


# USE THIS FUNCTION TO RUN PROGRAM. Takes a score and a window size; performs all functions to get from the score to the arrays. 
def score_to_data(score, win_size):
    data = get_Data(score)
    win_data = window(data, win_size)
    stored_data = storage(win_data)
    weighted_data = log_weight(stored_data)
    arrayed_data = make_arrays(weighted_data)
    return arrayed_data

# ------------------------------------------------------------------------------------------------------------

# Available corpus
# c = converter.parse('//Users/matthewchiu/Documents/Phase Tracking/Test Scores/MozartK545.xml')
c = converter.parse('//Users/matthewchiu/Documents/Phase Tracking/Scores/MozartK157_expo.xml')
pp = c.stripTies(retainContainers=False)


# Set excerpt (exc). The first format is for files that only contain the exposition. The second format is for full-movement files in which the exposition needs to be excerpted.
exc = pp.measures(1, None)
# exc.measures(1, 4).show()

# collect the magnitudes and phases for each window throughout the piece. The second variable (after exc) is the window sizeand is the only piece of information that should be changed.
magnitude = score_to_data(exc, 8)[0]
phase = score_to_data(exc, 8)[1]
DFTExponentials = score_to_data(exc, 8)[2]
# print(DFTExponentials)


# Takes magnitude or phase data to make a data frame. In mod12, only indexes 0-6 are needed, as the remaining are duplicates.
def make_dataframe(data):
    d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11 = ([], [], [], [], [], [], [], [], [], [], [], [])
    for d in data:
        d0.append(d[0])
        d1.append(d[1])
        d2.append(d[2])
        d3.append(d[3])
        d4.append(d[4])
        d5.append(d[5])
        d6.append(d[6])
        d7.append(d[7])
        d8.append(d[8])
        d9.append(d[9])
        d10.append(d[10])
        d11.append(d[11])
    df_info = {'f0': d0, 'f1': d1, 'f2': d2, 'f3': d3, 'f4': d4, 'f5': d5, 'f6':d6}
    return df_info

# Create magnitude dataframe
mag_df = pd.DataFrame(make_dataframe(magnitude))[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']]
mag_df

# Create phase dataframe
phase_df = pd.DataFrame(make_dataframe(phase))[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']]
phase_df

# ------------------------------------------------------------------------------------------------------------

# # Make area plot with the magnitude of all Fourier components.

# mag_df.plot.area(stacked=False, figsize=(20, 5), color=['xkcd:sky', 'xkcd:sun yellow', 'xkcd:fern', 'xkcd:rose',
#                                                      'xkcd:lilac', 'xkcd:very light brown']).legend(loc='upper right')

# # Make area plot with the phase of all Fourier components. (Not particularly useful)

# phase_df.plot.area(stacked=False, figsize=(20, 5), color=['xkcd:sky', 'xkcd:sun yellow', 'xkcd:fern', 'xkcd:rose',
#                                                      'xkcd:lilac', 'xkcd:very light brown']).legend(loc='upper right')


# # Plot the magnitude and phase of each component. Y axis for magnitude is on the left, Y axis for phase is on the right. There are two overlapping graphs for phase. The lighter colored is for the raw phase data, while the darker colored is for the phase data rounded to the nearest 'node'.
import matplotlib as mpl
import matplotlib.pyplot as plt

def make_graph(mag, phase, magcolor, phasecolor1, phasecolor2):
    mag.plot.area(stacked=False, figsize=(20, 5), color=magcolor)
    phase.plot(stacked=False, secondary_y=True, color=phasecolor1)
    plt.ylim((-3.14,3.14))
    roundedPhase = (round(phase*(12/(2*math.pi))))/(12/(2*math.pi))
    roundedPhase.plot(secondary_y=True, color=phasecolor2)
    # (round((phase*(12/2*math.pi)))/(12/2*math.pi)).plot(secondary_y=True, color=phasecolor2)
    
f1_info = (mag_df['f1'], phase_df['f1'], 'xkcd:sky', 'xkcd:lightish blue', 'xkcd:cobalt')
f2_info = (mag_df['f2'], phase_df['f2'], 'xkcd:sun yellow', 'xkcd:squash', 'xkcd:milk chocolate')
f3_info = (mag_df['f3'], phase_df['f3'], 'xkcd:fern', 'xkcd:swamp', 'xkcd:evergreen')
f4_info = (mag_df['f4'], phase_df['f4'], 'xkcd:rose', 'xkcd:pinkish', 'xkcd:velvet')
f5_info = (mag_df['f5'], phase_df['f5'], 'xkcd:lilac', 'xkcd:dark lavender', 'xkcd:eggplant purple')
f6_info = (mag_df['f6'], phase_df['f6'], 'xkcd:very light brown', 'xkcd:dark taupe', 'xkcd:deep brown')

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# Lazy Matthew doesn't want to recode for DataFrames...


# Make DataFrame an Array for Cosine Sim

phaseArrayFullPiece = []
counter = -1
for index, phase in phase_df.iterrows():
    counter+=1
    phaseArrayFullPiece.append([])
    roundedPhase = (round(phase*(12/(2*math.pi))))/(12/(2*math.pi))
    for element in roundedPhase:
        phaseArrayFullPiece[counter].append(element)

# print(phaseArrayFullPiece)

# ------------------------------------------------------------------------------------------------------------

# Cosine Similarity

def cosSim(x,y):
    cosSim = np.vdot(x, y)/(np.absolute(x)*np.absolute(y))
    z = (cosSim.real, cosSim.imag)
    euclideanAngle = cosSim.real/np.absolute(cosSim)
    phase = np.angle(cosSim)
    return cosSim

fullInformation = []
phaseForCosSimArray = []
phaseForCosSimArrayRounded = []
cosDistanceArray = []
# roundedCosDistanceArray = []

def cosDistance(windowedDftPiece):
    for counter, arrays in enumerate(windowedDftPiece):
        if counter+1 >= len(windowedDftPiece):
            break
        else:
            phaseForCosSimArray.append([])
            phaseForCosSimArrayRounded.append([])
            cosDistanceArray.append([])
            # roundedCosDistanceArray.append([])
            for counter2, i in enumerate(arrays):
                fullInformation.append(cosSim(i, windowedDftPiece[counter+1][counter2]))
                # PHASE INFORMATION BASED ON COSSIM
                phase = np.angle(cosSim(i, windowedDftPiece[counter+1][counter2]))
                phaseForCosSimArray[counter].append(phase)
                # PHASE ROUNDED
                phaseRoundedPrep = np.angle(cosSim(i, windowedDftPiece[counter+1][counter2]))
                phaseRounded = (round(phaseRoundedPrep*(12/(2*math.pi))))/(12/(2*math.pi))
                phaseForCosSimArrayRounded[counter].append(phaseRounded)
                # COSSINE DISTANCE
                euclideanAngle = cosSim(i, windowedDftPiece[counter+1][counter2]).real/np.absolute(cosSim(i, windowedDftPiece[counter+1][counter2]))
                cosDistanceArray[counter].append(1-euclideanAngle)
cosDistance(DFTExponentials)

# Dataframing the info
phaseForCosSimArray_df = pd.DataFrame(make_dataframe(phaseForCosSimArray))[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']]
phaseForCosSimArrayRounded_df = pd.DataFrame(make_dataframe(phaseForCosSimArrayRounded))[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']]
phaseForCosDistanceArray_df = pd.DataFrame(make_dataframe(cosDistanceArray))[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']]

def make_graph2(phase1, phasecolor1):
    phase1.plot.area(stacked = False, figsize=(20, 5), color=phasecolor1)
    plt.ylim((-3.14,3.14))
f1_PhaseInfo = (phaseForCosSimArrayRounded_df['f1'], 'blue')
f2_PhaseInfo = (phaseForCosSimArrayRounded_df['f2'], 'yellow')
f3_PhaseInfo = (phaseForCosSimArrayRounded_df['f3'], 'green')
f4_PhaseInfo = (phaseForCosSimArrayRounded_df['f4'], 'red')
f5_PhaseInfo = (phaseForCosSimArrayRounded_df['f5'], 'purple')
f6_PhaseInfo = (phaseForCosSimArrayRounded_df['f6'], 'brown')


# Leave for a second...
# def make_graph2(phase1, phase2, phasecolor1, phasecolor2):
#     phase1.plot.area(stacked = False, figsize=(20, 5), color=phasecolor1)
#     plt.ylim((-3.14,3.14))
#     phase2.plot.area(stacked = False, secondary_y=True, color=phasecolor2)
#     plt.ylim((-3.14,3.14))
# f1_PhaseInfo = (phaseForCosSimArray_df['f1'], phaseForCosSimArrayRounded_df['f1'], 'xkcd:sky', 'xkcd:lightish blue')
# f2_Phaseinfo = (phaseForCosSimArray_df['f2'], phaseForCosSimArrayRounded_df['f2'], 'xkcd:sun yellow', 'xkcd:squash')
# f3_PhaseInfo = (phaseForCosSimArray_df['f3'], phaseForCosSimArrayRounded_df['f3'], 'xkcd:fern', 'xkcd:swamp')
# f4_PhaseInfo = (phaseForCosSimArray_df['f4'], phaseForCosSimArrayRounded_df['f4'], 'xkcd:rose', 'xkcd:pinkish')
# f5_PhaseInfo = (phaseForCosSimArray_df['f5'], phaseForCosSimArrayRounded_df['f5'], 'xkcd:lilac', 'xkcd:dark lavender')
# f6_PhaseInfo = (phaseForCosSimArray_df['f6'], phaseForCosSimArrayRounded_df['f6'], 'xkcd:very light brown', 'xkcd:dark taupe')
# ------------------------------------------------------------------------------------------------------------
# PLOTTING THINGS
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.set_ylim((-3.14,3.14))
# ax2.set_ylim((-3.14,3.14))

# make_graph(*f1_info)
# make_graph(*f2_info)
# make_graph(*f3_info)
# make_graph(*f4_info)
# make_graph(*f5_info)
# make_graph(*f6_info)

# Cosine Similarity Rounded
# make_graph2(*f1_PhaseInfo)
# make_graph2(*f2_PhaseInfo)
# make_graph2(*f3_PhaseInfo)
# make_graph2(*f4_PhaseInfo)
# make_graph2(*f5_PhaseInfo)
# make_graph2(*f6_PhaseInfo)

plt.show()


