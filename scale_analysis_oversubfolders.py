#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:04:37 2019
@author: alex
"""

import os

import deeplabcut
videopath =['C:\\Users\\analysis\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3609.4d\\05_07_2019\\BRAC36094d 05_07_2019 12_37_39 1_trimmed.mp4',
'C:\\Users\\analysis\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3616.3b\\05_07_2019\\BRAC36163b 05_07_2019 12_37_39 2_trimmed.mp4',
'C:\\Users\\analysis\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3616.3e\\05_07_2019\\BRAC36163e  05_07_2019 12_25_18 4_trimmed.mp4',
'C:\\Users\\analysis\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3800.1e\\05_07_2019\\BRAC38001e 05_07_2019 12_09_27 2_trimmed.mp4']

def getsubfolders(folder):
    ''' returns list of subfolders '''
    return [os.path.join(folder,p) for p in os.listdir(folder) if os.path.isdir(os.path.join(folder,p))]

project='05_07_2019'

shuffle=1

projectpath='C:\\Users\\analysis\\Desktop\\05_07_2019-MC-2019-08-14'

basepath='C:\\Users\\analysis\\Desktop\\Mingran\\analysed_trimmed_data' #data'

config=os.path.join(projectpath,'config.yaml')

'''
Imagine that the data (here: videos of 3 different types) are in subfolders:
    /January/January29 ..
    /February/February1
    /February/February2

    etc.
'''
'''
subfolders=getsubfolders(basepath)
for subfolder in subfolders: #this would be January, February etc. in the upper example
    print("Starting analyze data in:", subfolder)
    subsubfolders=getsubfolders(subfolder)
    for subsubfolder in subsubfolders: #this would be Febuary1, etc. in the upper example...'''
subfolders=getsubfolders(basepath)
'''for subfolder in subfolders:
    print("Starting analyze data in:", subfolder)
    subsubfolders=getsubfolders(subfolder)
    for subsubfolder in subsubfolders:
        print("Starting analyze data in:", subsubfolder)'''
for path in videopath:
    for vtype in ['.mp4']:
        deeplabcut.analyze_videos(config,[path],shuffle=shuffle,videotype=vtype,save_as_csv=True)
        deeplabcut.filterpredictions(config,[path],videotype = vtype, shuffle = shuffle)
        deeplabcut.plot_trajectories(config,[path],videotype=vtype)
        deeplabcut.create_labeled_video(config,[path],videotype=vtype,filtered=True)
