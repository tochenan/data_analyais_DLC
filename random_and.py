import deeplabcut
import tensorflow as tf

'''deeplabcut.create_new_project('1_2_4_chamber','MC',['C:\\Users\\analysis\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3477.2b\\05_08_2019\\BRAC34772b top_left 05_08_2019 12_40_54 1_trimmed.mp4'],
working_directory='C:\\Users\\analysis\\Desktop')'''

videopath =['C:\\Users\\analysis\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3477.2b\\13_08_2019\\BRAC34772b top_right 13_08_2019 14_39_52 2_trimmed.mp4',
'C:\\Users\\analysis\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3583.3e\\23_07_2019\\BRAC35833e 23_07_2019 13_35_06 4_trimmed.mp4',
'C:\\Users\\analysis\\Desktop\\Mingran\\analysed_trimmed_data\\BRAC3616.3c\\23_07_2019\\BRAC36163c 23_07_2019 12_41_11 3_trimmed.mp4']
config = 'C:\\Users\\analysis\\Desktop\\1_2_4_chamber-MC-2019-08-23\\config.yaml'


'''deeplabcut.add_new_videos(config,videopath)
deeplabcut.extract_frames(config,'automatic','kmeans')'''
deeplabcut.label_frames(config)

deeplabcut.check_labels(config)
deeplabcut.create_training_dataset(config)
deeplabcut.train_network(config)


deeplabcut.extract_outlier_frames(config,videopath)
deeplabcut.refine_labels(config)
deeplabcut.merge_datasets(config)
deeplabcut.create_training_dataset(config)
deeplabcut.train_network(config)
