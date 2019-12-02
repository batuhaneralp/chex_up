# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import model_from_json
from keras_efficientnets import custom_objects
from keras.preprocessing import image
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("input_path")
ap.add_argument("output_path")

args = vars(ap.parse_args())

valid_df = pd.read_csv(args["input_path"])
csv_path = args["output_path"]

finding_labels = []
for col in valid_df.columns[5:11]:
    finding_labels.append(col)

# * VALIDATION DATA PREPARATION
labels = []
for label in finding_labels:
    temp = []
    temp.append(valid_df[label].tolist())
    labels.append(temp[0])

labels_t = list(map(list, zip(*labels)))

valid_df['disease_vec'] = labels_t


IMG_SIZE = (300, 300)
core_idg = ImageDataGenerator(samplewise_center=True,
                              samplewise_std_normalization=True,
                              horizontal_flip=True,
                              vertical_flip=False,
                              height_shift_range=0.05,
                              width_shift_range=0.1,
                              rotation_range=5,
                              shear_range=0.1,
                              fill_mode='reflect',
                              zoom_range=0.15)


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    # print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                              class_mode='sparse',
                                              **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = ''  # since we have the full path
    #print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


test_X, test_Y = next(flow_from_dataframe(core_idg,
                                          valid_df,
                                          path_col='Path',
                                          y_col='disease_vec',
                                          target_size=IMG_SIZE,
                                          color_mode='grayscale',
                                          batch_size=1024))  # one big batch

# load json and create model
json_file = open('data/300_300_B2_0.4_0.4_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("data/300_300_B2_0.4_0.4_weights.best.hdf5")
print("Loaded model from disk")


pred_Y = loaded_model.predict(test_X, batch_size=32, verbose=True)

np.savetxt(csv_path, pred_Y, delimiter=',')

"""
fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
for (idx, c_label) in enumerate(finding_labels):
    fpr, tpr, thresholds = roc_curve(
        test_Y[:, idx].astype(int), pred_Y[:, idx])
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('trained_net.png')
"""
