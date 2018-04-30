'''Run this first part only once, if you want to download the entire
LFW file (200MB). Otherwise use the people object stored in 
the 'pickle' file faces.pickle (provided by instructor)'''
'''################################
import pickle
from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
with open('faces.pickle', 'wb') as f:
    pickle.dump(people, f)
####################################'''

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import mglearn

with open('faces.pickle', 'rb') as f:
    people = pickle.load(f)
print("people.images.shape: {}".format(people.images.shape))
print("Number of classes: {}".format(len(people.target_names)))

#limit nr. of faces of one person to 50:
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]

# scale the grey-scale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability:
X_people = X_people / 255.
print

