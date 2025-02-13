"""
Prepare the SHPHERE Challenge dataset for localisation at home environment.
"""
import os
import requests
import numpy as np
import pandas as pd

# download the SPHERE Challenge dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'train')
if not os.path.exists(input_file_path):
    data_url = 'http://data.bris.ac.uk/datasets/8gccwpx47rav19vk8x4xapcog/8gccwpx47rav19vk8x4xapcog.zip'
    data_file = os.path.basename(data_url)
    r = requests.get(data_url)
    with open(os.path.join(os.path.dirname(__file__), data_file), 'wb') as f:
        f.write(r.content)

    # unzip the enwik8 dataset
    import zipfile
    with zipfile.ZipFile(os.path.join(os.path.dirname(__file__), data_file), 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(__file__))

    # unzip data files
    file_name = ['test.zip', 'train.zip']
    for file in file_name:
        with zipfile.ZipFile(os.path.join(os.path.dirname(__file__), data_file.split('.')[0], file), 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(__file__))

    os.remove(os.path.join(os.path.dirname(__file__), os.path.basename(data_url)))


# # preprocess the data for activity recognition
# df_list = []
# for id in ['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009', '00010']:
#     # format the data (accelerometer and RSSI) for activity recognition
#     df = pd.read_csv(os.path.join(input_file_path, id, 'acceleration.csv'))
#     df.replace(np.nan, -120, inplace=True)
#     df['t'] = pd.to_datetime(df['t'], unit='s')
#     df.set_index('t', inplace=True)
#     df = df.resample('1s').mean()
#     df.dropna(inplace=True)

#     # format the targe
#     df_tgt = pd.read_csv(os.path.join(input_file_path, id, 'targets.csv'))
#     df_tgt['t'] = pd.to_datetime(df_tgt['start'], unit='s')
#     df_tgt.set_index('t', inplace=True)
#     df_tgt.dropna(inplace=True)
#     df_tgt.drop(['start', 'end'], axis=1, inplace=True)
#     labels = list(df_tgt)
#     # add maximum target index column
#     for n, row in df_tgt.iterrows():
#         df_tgt.loc[n, 'target'] = np.nan if 't_' in labels[np.argmax(row)] else np.argmax(row)
#     df_tgt.dropna(inplace=True)

#     # concatenate the data (accelerometer and RSSI) and the target
#     df = pd.concat([df, df_tgt], axis=1, join='inner')
#     # append to the df_list
#     df_list.append(df)

# # concatenate the all data
# df = pd.concat(df_list, axis=0, ignore_index=True)

# # save the data
# df.to_csv(os.path.join(os.path.dirname(__file__), 'activity_recognition.csv'))
