import numpy as np
from collections import namedtuple
from scipy.spatial import distance
import pandas as pd
import auxiliaryfunctions
import os
import os.path
from pathlib import Path
import itertools
from select_nest import select_roi

'''
    Functions to extract time spent by the mouse in each of a list of user defined ROIS
    Contributed by Federico Claudi
    https://github.com/FedeClaudi
    Example usage:
    rois        -->  a dictionary with name and position of each roi
    tracking    -->  a pandas dataframe with X,Y,Velocity for each bodypart
    bodyparts   -->  a list with the name of all the bodyparts

    -----------------------------------------------------------------------------------

   ------------------------------------------------------------------------------------
    if Velocity is not know, it can be calculated using "calc_distance_between_points_in_a_vector_2d":
        vel = calc_distance_between_points_in_a_vector_2d(np.array(tracking.bp.x.values, tracking.bp.y.values))
    which returns a 1d vector with the velocity in pixels/frame [effectively the number pixels a tracked point moved
    from one frame to the next]


'''

def calc_distance_between_points_in_a_vector_2d(v1):
    '''calc_distance_between_points_in_a_vector_2d [for each consecutive pair of points, p1-p2, in a vector, get euclidian distance]
    This function can be used to calculate the velocity in pixel/frame from tracking data (X,Y coordinates)

    Arguments:
        v1 {[np.array]} -- [2d array, X,Y position at various timepoints]

    Raises:
        ValueError

    Returns:
        [np.array] -- [1d array with distance at each timepoint]
    >>> v1 = [0, 10, 25, 50, 100]
    >>> d = calc_distance_between_points_in_a_vector_2d(v1)
    '''
    # Check data format
    if isinstance(v1, dict) or not np.any(v1) or v1 is None:
            raise ValueError(
                'Feature not implemented: cant handle with data format passed to this function')

    # If pandas series were passed, try to get numpy arrays
    try:
        v1, v2 = v1.values, v2.values
    except:  # all good
        pass
    # loop over each pair of points and extract distances
    dist = []

    for n, part in enumerate(v1):
        # Get a pair of points
        for f, pos in enumerate(part):
            if n == 0:  # get the position at time 0, velocity is 0
                p0 = pos
                dist.append(0)
            else:
                p1 = pos  # get position at current frame
                # Calc distance
                dist.append(np.abs(distance.euclidean(p0, p1)))

            # Prepare for next iteration, current position becomes the old one and repeat
                p0 = p1

    return np.array(dist)


def get_roi_at_each_frame(bp_data, rois):
    """
    Given position data for a bodypart and the position of a list of rois, this function calculates which roi is
    the closest to the bodypart at each frame
    :param bp_data: numpy array: [nframes, 2] -> X,Y position of bodypart at each frame
                    [as extracted by DeepLabCut] --> df.bodypart.values
    :param rois: dictionary with the position of each roi. The position is stored in a named tuple with the location of
                    two points defyining the roi: topleft(X,Y) and bottomright(X,Y).
    :return: tuple, closest roi to the bodypart at each frame
    """

    if not isinstance(rois, dict): raise ValueError('rois locations should be passed as a dictionary')

    if not isinstance(bp_data, np.ndarray):
        if not isinstance(bp_data, tuple): raise ValueError('Unrecognised data format for bp tracking data')
        else:
            pos = np.zeros((len(bp_data.x), 2))
            pos[:, 0], pos[:, 1] = bp_data.x, bp_data.y
            bp_data = pos

    # Get the center of each roi
    centers = []
    for index, points in rois.items():
        center_x = (rois.get(index)[0][1][0] + rois.get(index)[1][1][0] + rois.get(index)[2][1][0]) / 3
        center_y = (rois.get(index)[0][1][1] + rois.get(index)[1][1][1] + rois.get(index)[2][1][0]) / 3
        center = np.asarray([center_x, center_y])
        centers.append(center)

    roi_names = list(rois.keys())
    # Calc distance to each roi for each frame
    data_length = bp_data.shape[0]

    distances = np.zeros((len(centers), len(centers[0][0])))
    for idx, center in enumerate(centers):
        dist = np.hypot(np.subtract(center[0], bp_data[0]), np.subtract(center[1], bp_data[1]))
        distances[idx] = dist
    # Get which roi the mouse is in at each frame
    sel_rois = np.argmin(distances, 0)
    roi_at_each_frame = tuple([roi_names[x] for x in sel_rois])
    return roi_at_each_frame




def get_timeinrois_stats(bp_tracking,data, rois, fps=None):
    """
    Quantify number of times the animal enters a roi, cumulative number of frames spend there, cumulative time in seconds
    spent in the roi and average velocity while in the roi.
    In which roi the mouse is at a given frame is determined with --> get_roi_at_each_frame()
    Quantify the ammount of time in each  roi and the avg stay in each roi
    :param data: trackind data is a numpy array with shape (n_frames, 3) with data for X,Y position and Velocity
    :param rois: dictionary with the position of each roi. The position is stored in a named tuple with the location of
                two points defyining the roi: topleft(X,Y) and bottomright(X,Y).
    :param fps: framerate at which video was acquired
    :return: dictionary
    # Testing
    >>> position = namedtuple('position', ['topleft', 'bottomright'])
    >>> rois = {'middle': position((300, 400), (500, 800))}
    >>> data = np.zeros((23188, 3))
    >>> res = get_timeinrois_stats(data, rois, fps=30)
    >>> print(res)
    """

    def get_indexes(lst, match):
        return np.asarray([i for i, x in enumerate(lst) if x == match])

    # get roi at each frame of data
    data_rois = get_roi_at_each_frame(bp_tracking, rois)
    data_time_inrois = {name: data_rois.count(name) for name in set(data_rois)}  # total time (frames) in each roi
    vels=[]
    # number of enters in each roi
    transitions = [n for i, n in enumerate(list(data_rois)) if i == 0 or n != list(data_rois)[i - 1]]
    transitions_count = {name: transitions.count(name) for name in transitions}

    # avg time spend in each roi (frames)
    avg_time_in_roi = {transits[0]: time / transits[1]
                       for transits, time in zip(transitions_count.items(), data_time_inrois.values())}

    # avg time spend in each roi (seconds)
    if fps is not None:
        data_time_inrois_sec = {name: t / fps for name, t in data_time_inrois.items()}
        avg_time_in_roi_sec = {name: t / fps for name, t in avg_time_in_roi.items()}
    else:
        data_time_inrois_sec, avg_time_in_roi_sec = None, None
    # get avg velocity in each roi
    avg_vel_per_roi = {}
    for name in set(data_rois):
        indexes = get_indexes(data_rois, name)
        for index in indexes:
            vel=data[2][index]
            vels.append(vel)
        avg_vel_per_roi[name] = np.average(np.asarray(vels))

    results = dict(transitions_per_roi=transitions_count,
                   cumulative_time_in_roi=data_time_inrois,
                   cumulative_time_in_roi_sec=data_time_inrois_sec,
                   avg_time_in_roi=avg_time_in_roi,
                   avg_time_in_roi_sec=avg_time_in_roi_sec,
                   avg_vel_in_roi=avg_vel_per_roi)

    return results



videopath =['C:\\Users\\caom\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3609.4d\\05_07_2019\\BRAC36094d 05_07_2019 12_37_39 1_trimmed.mp4',
            'C:\\Users\\caom\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3616.3b\\05_07_2019\\BRAC36163b 05_07_2019 12_37_39 2_trimmed.mp4',
            'C:\\Users\\caom\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3616.3e\\05_07_2019\\BRAC36163e  05_07_2019 12_25_18 4_trimmed.mp4',
            'C:\\Users\\caom\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3800.1e\\05_07_2019\\BRAC38001e 05_07_2019 12_09_27 2_trimmed.mp4']

config = 'C:\\Users\\caom\\Desktop\\05_07_2019-MC-2019-08-14\\config.yaml'

shuffle=1
for video in videopath:

    # set up parameter
    videotype='.mp4'
    filtered=True
    trainingsetindex=0
    cfg = auxiliaryfunctions.read_config(config)
    videofolder = str(Path(video).parents[0])
    trainFraction = cfg['TrainingFraction'][trainingsetindex]
    bodyparts2plot = cfg['bodyparts']
    pcutoff = cfg['pcutoff']
    DLCscorer = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction) #automatically loads corresponding model (even training iteration based on snapshot index)
    scorer=DLCscorer
    Videos=auxiliaryfunctions.Getlistofvideos(video,videotype)
    pup_list=['pupA_head','pupA_body','pupA_tail','pupB_head','pupB_body','pupB_tail']
    body_list=['mouse','left_ear','right_ear','left_shoulder','right_shoulder','left_back_leg','right_back_leg']
    pup_list_1=['pupA','pupA','pupA','pupB','pupB','pupB']
    rois={}
    roi_at_each_frames=[]
    i=0
    tuple1=[]
    tuple2=[]
    velocities=[]
    nest_roi=select_roi(videopath)
    # set up dataframe
    if filtered==False:
        dataname = str(Path(video).stem) + DLCscorer + '.h5'
        Dataframe = pd.read_hdf(os.path.join(videofolder,dataname))

    else:#for filtered output
        try:
            Dataframe = pd.read_hdf(os.path.join(videofolder,str(Path(video).stem) + DLCscorer + 'filtered.h5'))

        except FileNotFoundError:
            print("No filtered predictions found, using frame-by-frame output instead.")
            Dataframe = pd.read_hdf(os.path.join(videofolder,str(Path(video).stem) + DLCscorer + 'filtered'+'.h5'))

    for bp in pup_list:
        if i<3:
            Index=Dataframe[scorer][bp]['likelihood'].values < pcutoff
            X=Dataframe[scorer][bp]['x'].values
            Y=Dataframe[scorer][bp]['y'].values
            roi=[str(bp),(X,Y)]
            tuple1.append(roi)
            i=i+1
            rois['pupA']=tuple1
        elif i<6:
            Index=Dataframe[scorer][bp]['likelihood'].values < pcutoff
            X=Dataframe[scorer][bp]['x'].values
            Y=Dataframe[scorer][bp]['y'].values
            roi=[str(bp),(X,Y)]
            tuple2.append(roi)
            i=i+1
            rois['pupB']=tuple2
    for bp in body_list:
        Index=Dataframe[scorer][bp]['likelihood'].values < pcutoff
        X=Dataframe[scorer][bp]['x'].values
        Y=Dataframe[scorer][bp]['y'].values
        bp_tracking=np.concatenate(([[X],[Y]]),axis=0)
        velocity=calc_distance_between_points_in_a_vector_2d(bp_tracking)
        roi_at_each_frame=get_roi_at_each_frame(bp_tracking,rois)
        data=[[X],[Y],velocity]
        result=get_timeinrois_stats(bp_tracking,data, rois, fps=30)
        velocities.append(velocity)
        roi_at_each_frames.append(bp_tracking)





    i=0
    while i <= len(rois):
        ls=[].append(i)
        i=i+1

    #storing as a csv file with the number of frames on the first column
