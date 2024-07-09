#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Common imports
import numpy as np
import os
from backend import import_excel, export_excel

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
# style.use('bmh')
from mpl_toolkits.mplot3d import Axes3D

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import seaborn as sns

import keras
import tensorflow as tf

import random

import sys

sys.path.append("..")

import dataset, network

# # Load Data

# In[2]:


# Preprocessing

scenario = "sinus"  # sinus, helix
n_instance = 1000
n_features = 2
Z = 2
scales = ['-1-1', '0-1']
scaled = '-1-1'
nodes = 2

# In[3]:


if scenario in ("3d", "helix"):
    X_train, y_train, X_test, y_test, X_valid, y_valid = dataset.get_dataset(n_instance, scenario)
    print("X_train= x,y", X_train.shape)
    print("y_train= z", y_train.shape)

    ax = plt.subplot(projection='3d')
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='orange')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.tight_layout()


else:
    X_train, y_train, X_test, y_test, X_valid, y_valid = dataset.get_dataset(n_instance, scenario)
    plt.scatter(X_train, y_train, c='orange', label='Sample Data')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.tight_layout()

# In[4]:


# storage data
os.system('mkdir Dataset')
os.system('mkdir AAE')
os.system('mkdir AAE/Models')
os.system('mkdir AAE/Losses')
os.system('mkdir AAE/Random_test')
# export_excel(X_train, 'Dataset/X_train')
# export_excel(y_train, 'Dataset/y_train')

# print(X_train.shape,y_train.shape)
X_train = import_excel('Dataset/X_train')
y_train = import_excel('Dataset/y_train')
print('made dataset')

# # AAE

# ### Architecture

# In[5]:


encoder = network.build_encoder(Z, nodes, n_features)
# print("Encoder:\n")
# encoder.summary()


decoder = network.build_decoder(Z, nodes, n_features)
# print("Decoder:\n")
# decoder.summary()

discriminator = network.build_discriminator(Z)
# print("Discriminator:\n")
# discriminator.summary()


# ### Preprocessing

# In[6]:


import Model

GANorWGAN = 'WGAN'
epochs = 2000
BATCH_SIZE = 100
n_dis = 4
n_endis = 1
n_decoder = 2
# n_autoencoder=1


# In[7]:


aae = Model.AAE(Z, n_features, BATCH_SIZE, GANorWGAN, nodes, n_dis, n_decoder)

# In[8]:


train_dataset, scaler, X_train_scaled = aae.preproc(X_train, y_train, scaled)

print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_train_scaled", X_train_scaled.shape)

# ## Training

# In[ ]:


hist = aae.train(Z, BATCH_SIZE, train_dataset, epochs, scaler, scaled, X_train_scaled)
# Dropout0.8+delete Flatten, epoch=2000好
# 0126Dropout0.6+delete Flatten+增加encoder/decoder


# 1203LeakyReLU+Flatten
# Change e/d architexture+1203
# 試試backend/4


# self.n_critic=10
# n_c=5
# complex dis_network + more n_autoencoder


# ### predict from the decoder

# In[ ]:


# predict the labels of the data values on the basis of the trained model.
# sampling from the latent space without prediction

latent_values = np.random.normal(loc=0, scale=1, size=([1000, Z]))
predicted_values = aae.decoder(latent_values)

predicted_values2 = aae.decoder(aae.encoder(X_train_scaled))
predicted_values3 = aae.encoder(X_train_scaled)
predicted_values4 = scaler.inverse_transform(X_train_scaled)

if scaled == '-1-1':
    predicted_values = scaler.inverse_transform(predicted_values)
    predicted_values2 = scaler.inverse_transform(predicted_values2)
    # predicted_values3 = scaler.inverse_transform(predicted_values3)

elif scaled == '0-1':
    predicted_values = scaler.inverse_transform(predicted_values)
    predicted_values2 = scaler.inverse_transform(predicted_values2)

if n_features == 3:
    print("Predicted Values:", predicted_values.shape)
    print("latent_space:", Z)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("epochs:", epochs)

    ab = plt.subplot(projection='3d')
    ab.scatter(predicted_values[:, 0], predicted_values[:, 1], predicted_values[:, 2])
    ab.set_ylabel('Y')
    ab.set_zlabel('Z')
    ab.set_xlabel('X')

    print("X-Y 2D slices:")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=False, sharex=False)
    plt.ylim(-1.5, 1.5)
    plt.xlim(-1.5, 1.5)
    axes[0].scatter(X_train[:, 0], X_train[:, 1])
    axes[0].scatter(predicted_values[:, 0], predicted_values[:, 1])
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    plt.ylim(-1.5, 1.5)
    plt.xlim(-2, 22)
    axes[1].scatter(X_train[:, 1], y_train)
    axes[1].scatter(predicted_values[:, 1], predicted_values[:, 2])
    axes[1].set_xlabel("Y")
    axes[1].set_ylabel("Z")

    plt.xlim(-1.5, 1.5)
    plt.ylim(-2, 22)
    axes[2].scatter(X_train[:, 0], y_train)
    axes[2].scatter(predicted_values[:, 0], predicted_values[:, 2])
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Z")

    plt.tight_layout()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=False, sharex=False)

    ac = np.where(np.logical_and(X_train[:, 0] >= -0.8 - 0.05, X_train[:, 0] <= -0.8 + 0.05), X_train[:, 1], None)
    ad = np.where(np.logical_and(predicted_values[:, 0] >= -0.8 - 0.05, predicted_values[:, 0] <= -0.8 + 0.05),
                  predicted_values[:, 1], None)
    axes[0].scatter(ac, y_train)
    axes[0].scatter(ad, predicted_values[:, 2])
    axes[0].set_xlabel("Y(X=-0.8)")
    axes[0].set_ylabel("Y")

    ae = np.where(np.logical_and(X_train[:, 0] >= 0.0 - 0.05, X_train[:, 0] <= 0.0 + 0.05), X_train[:, 1], None)
    af = np.where(np.logical_and(predicted_values[:, 0] >= 0.0 - 0.05, predicted_values[:, 0] <= 0.0 + 0.05),
                  predicted_values[:, 1], None)
    axes[1].scatter(ae, y_train)
    axes[1].scatter(af, predicted_values[:, 2])
    axes[1].set_xlabel("Y(X=0.0)")
    axes[1].set_ylabel("Z")

    ag = np.where(np.logical_and(X_train[:, 0] >= 0.8 - 0.05, X_train[:, 0] <= 0.8 + 0.05), X_train[:, 1], None)
    ah = np.where(np.logical_and(predicted_values[:, 0] >= 0.8 - 0.05, predicted_values[:, 0] <= 0.8 + 0.05),
                  predicted_values[:, 1], None)
    axes[2].scatter(ag, y_train)
    axes[2].scatter(ah, predicted_values[:, 2])
    axes[2].set_xlabel("Y(X=0.8)")
    axes[2].set_ylabel("Z")

    plt.tight_layout()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=False, sharex=False)
    ac = np.where(np.logical_and(X_train[:, 1] >= 0.2 - 0.05, X_train[:, 1] <= 0.2 + 0.05), X_train[:, 0], None)
    ad = np.where(np.logical_and(predicted_values[:, 1] >= 0.2 - 0.05, predicted_values[:, 1] <= 0.2 + 0.05),
                  predicted_values[:, 0], None)
    axes[0].scatter(ac, y_train)
    axes[0].scatter(ad, predicted_values[:, 2])
    axes[0].set_xlabel("X(Y=0.2)")
    axes[0].set_ylabel("Z")

    ae = np.where(np.logical_and(X_train[:, 1] >= 0.5 - 0.05, X_train[:, 1] <= 0.5 + 0.05), X_train[:, 0], None)
    af = np.where(np.logical_and(predicted_values[:, 1] >= 0.5 - 0.05, predicted_values[:, 1] <= 0.5 + 0.05),
                  predicted_values[:, 0], None)
    axes[1].scatter(ae, y_train)
    axes[1].scatter(af, predicted_values[:, 2])
    axes[1].set_xlabel("X(Y=0.5)")
    axes[1].set_ylabel("Z")

    ag = np.where(np.logical_and(X_train[:, 1] >= 0.8 - 0.05, X_train[:, 1] <= 0.8 + 0.05), X_train[:, 0], None)
    ah = np.where(np.logical_and(predicted_values[:, 1] >= 0.8 - 0.05, predicted_values[:, 1] <= 0.8 + 0.05),
                  predicted_values[:, 0], None)
    axes[2].scatter(ag, y_train)
    axes[2].scatter(ah, predicted_values[:, 2])
    axes[2].set_xlabel("X(Y=0.8)")
    axes[2].set_ylabel("Z")

    plt.tight_layout()


else:
    print("Predicted Values:", predicted_values.shape)
    # plt.scatter(X_train, y_train,c='orange') #sample

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=False, sharex=False)

    axes[0].scatter(predicted_values3[:, 0], predicted_values3[:, 1], c='pink')  # encoder(X_train_scaled)
    axes[0].scatter(latent_values[:, 0], latent_values[:, 1], c='grey')
    axes[0].set_ylabel('Y')
    axes[0].set_xlabel('X')

    axes[1].scatter(predicted_values2[:, 0], predicted_values2[:, 1], )  # encoder/decoder
    # axes[1].scatter(predicted_values4[:,0],predicted_values4[:,1],c='grey')#X_trained_scaled
    axes[1].set_ylabel('Y')
    axes[1].set_xlabel('X')

    axes[2].scatter(predicted_values[:, 0], predicted_values[:, 1], c='red')  # decoder(latent space)
    # axes[2].scatter(predicted_values4[:,0],predicted_values4[:,1],c='grey')#X_trained_scaled
    axes[2].set_ylabel('Y')
    axes[2].set_xlabel('X')

    plt.tight_layout()

# ### Applying the prediction function

# In[ ]:


# define these for desired prediction
x_input = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
n_points = 900
y_min = -1
y_max = 1

# produces an input of fixed x coordinates with random y values
predict1 = np.full((n_points // 9, n_features), x_input[0])
predict2 = np.full((n_points // 9, n_features), x_input[1])
predict3 = np.full((n_points // 9, n_features), x_input[2])
predict4 = np.full((n_points // 9, n_features), x_input[3])
predict5 = np.full((n_points // 9, n_features), x_input[4])
predict6 = np.full((n_points // 9, n_features), x_input[5])
predict7 = np.full((n_points // 9, n_features), x_input[6])
predict8 = np.full((n_points // 9, n_features), x_input[7])
predict9 = np.full((n_points // 9, n_features), x_input[8])

predictthis = np.concatenate((predict1, predict2, predict3, predict4, predict5, predict6, predict7, predict8, predict9))
predictthis = scaler.fit_transform(predictthis)
input_test = predictthis.reshape(n_points, n_features).astype('float32')

print("input_test :", input_test.shape)
plt.scatter(input_test[:, 0], input_test[:, 1], c='grey')
plt.ylabel('Y')
plt.xlabel('X')
plt.tight_layout()

# In[ ]:


# X_generated = aae.generator.predict(input_test)
X_generated = aae.decoder(aae.encoder.predict(input_test))
X_generated = scaler.inverse_transform(X_generated)
print("X_generated :", X_generated.shape)

# In[ ]:


if scenario in ("3d", "helix"):
    print("latent_space=", latent_space)
    print("Epochs=", epochs)
    print("BATCH_SIZE=", BATCH_SIZE)
    print("use_bias=", use_bias)

    ax = plt.subplot(projection='3d')
    ax.scatter(X_generated[:, 0], X_generated[:, 1], X_generated[:, 2], label='Generated Data')

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.legend(loc='best')
    plt.tight_layout()

    print("X-Y 2D slices:")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=False, sharex=False)
    plt.ylim(-1.5, 1.5)
    plt.xlim(-1.5, 1.5)
    axes[0].scatter(X_train[:, 0], X_train[:, 1])
    axes[0].scatter(X_generated[:, 0], X_generated[:, 1])
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    plt.ylim(-1.5, 1.5)
    plt.xlim(-2, 22)
    axes[1].scatter(X_train[:, 1], y_train)
    axes[1].scatter(X_generated[:, 1], X_generated[:, 2])
    axes[1].set_xlabel("Y")
    axes[1].set_ylabel("Z")

    plt.xlim(-1.5, 1.5)
    plt.ylim(-2, 22)
    axes[2].scatter(X_train[:, 0], y_train)
    axes[2].scatter(X_generated[:, 0], X_generated[:, 2])
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Z")

    plt.tight_layout()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=False, sharex=False)

    ac = np.where(np.logical_and(X_train[:, 0] >= -0.8 - 0.05, X_train[:, 0] <= -0.8 + 0.05), X_train[:, 1], None)
    ad = np.where(np.logical_and(X_generated[:, 0] >= -0.8 - 0.05, X_generated[:, 0] <= -0.8 + 0.05), X_generated[:, 1],
                  None)
    axes[0].scatter(ac, y_train)
    axes[0].scatter(ad, X_generated[:, 2])
    axes[0].set_xlabel("Y(X=-0.8)")
    axes[0].set_ylabel("Y")

    ae = np.where(np.logical_and(X_train[:, 0] >= 0.0 - 0.05, X_train[:, 0] <= 0.0 + 0.05), X_train[:, 1], None)
    af = np.where(np.logical_and(X_generated[:, 0] >= 0.0 - 0.05, X_generated[:, 0] <= 0.0 + 0.05), X_generated[:, 1],
                  None)
    axes[1].scatter(ae, y_train)
    axes[1].scatter(af, X_generated[:, 2])
    axes[1].set_xlabel("Y(X=0.0)")
    axes[1].set_ylabel("Z")

    ag = np.where(np.logical_and(X_train[:, 0] >= 0.8 - 0.05, X_train[:, 0] <= 0.8 + 0.05), X_train[:, 1], None)
    ah = np.where(np.logical_and(X_generated[:, 0] >= 0.8 - 0.05, X_generated[:, 0] <= 0.8 + 0.05), X_generated[:, 1],
                  None)
    axes[2].scatter(ag, y_train)
    axes[2].scatter(ah, X_generated[:, 2])
    axes[2].set_xlabel("Y(X=0.8)")
    axes[2].set_ylabel("Z")

    plt.tight_layout()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=False, sharex=False)
    ac = np.where(np.logical_and(X_train[:, 1] >= 0.2 - 0.05, X_train[:, 1] <= 0.2 + 0.05), X_train[:, 0], None)
    ad = np.where(np.logical_and(X_generated[:, 1] >= 0.2 - 0.05, X_generated[:, 1] <= 0.2 + 0.05), X_generated[:, 0],
                  None)
    axes[0].scatter(ac, y_train)
    axes[0].scatter(ad, X_generated[:, 2])
    axes[0].set_xlabel("X(Y=0.2)")
    axes[0].set_ylabel("Z")

    ae = np.where(np.logical_and(X_train[:, 1] >= 0.5 - 0.05, X_train[:, 1] <= 0.5 + 0.05), X_train[:, 0], None)
    af = np.where(np.logical_and(X_generated[:, 1] >= 0.5 - 0.05, X_generated[:, 1] <= 0.5 + 0.05), X_generated[:, 0],
                  None)
    axes[1].scatter(ae, y_train)
    axes[1].scatter(af, X_generated[:, 2])
    axes[1].set_xlabel("X(Y=0.5)")
    axes[1].set_ylabel("Z")

    ag = np.where(np.logical_and(X_train[:, 1] >= 0.8 - 0.05, X_train[:, 1] <= 0.8 + 0.05), X_train[:, 0], None)
    ah = np.where(np.logical_and(X_generated[:, 1] >= 0.8 - 0.05, X_generated[:, 1] <= 0.8 + 0.05), X_generated[:, 0],
                  None)
    axes[2].scatter(ag, y_train)
    axes[2].scatter(ah, X_generated[:, 2])
    axes[2].set_xlabel("X(Y=0.8)")
    axes[2].set_ylabel("Z")

    plt.tight_layout()


else:
    print("Generated Data:", X_generated.shape)
    plt.scatter(X_train, y_train, c='orange')
    plt.scatter(X_generated[:, 0], X_generated[:, 1])
    plt.scatter(predicted_values4[:, 0], predicted_values4[:, 1], c='grey')  # X_trained_scaled
    # plt.scatter(predicted_values2[:,0],predicted_values2[:,1])
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.tight_layout()

# In[ ]:




