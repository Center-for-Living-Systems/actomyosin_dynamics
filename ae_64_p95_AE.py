# a direct run of the actomyosin data of AE
# Liya Ding
# 2024.03

import tensorflow as tf
from ../utils import actomyosin_data, plot64
from ../model.autoencoder_64 import AE, VAE, CVAE
from ../train_utils.autoencoder import AETrain, VAETrain, CVAETrain
import time
import argparse
import datetime 
from packaging import version
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import tifffile
import os
import warnings
from itertools import cycle, islice

from skimage.transform import resize
from skimage.io import imsave, imread

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

#########################################################################################

ae_type = 'AE'
net_type = 'conv'
latent_dim = 2,
num_epochs = 200,
learn_rate = 0.001,
batch_size = 100
train_buf = 1200

#########################################################################################


data_dir = '/net/projects/CLS/actomyosin_dynamics/data/LifeAct-NMY2-GFP_patchsize_64_p95_all/NMY2_wt'
patch_size = 64

filenames = [x for x in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, x)) and ('.tif' in x)]

num_of_samples = len(filenames)

train_images = np.zeros([num_of_samples, patch_size, patch_size])
train_labels= np.zeros([num_of_samples])

for filename_ind in range(num_of_samples):
    filename = filenames[filename_ind] 
             
    train_img = tifffile.imread(os.path.join(data_dir,filename))
    small_train_img = train_img/((train_img.max()))
    train_images[filename_ind,:,:] = small_train_img
    train_labels[filename_ind] = int(filename[1:5]+filename[6:10]+filename[11:15])


train_images = train_images.reshape(train_images.shape[0], 64, 64, 1).astype('float32')

#########################################################################################

TRAIN_BUF = train_buf
BATCH_SIZE = batch_size

train_dataset_image = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)
train_dataset_label = tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE)
train_dataset = tf.data.Dataset.zip((train_dataset_image, train_dataset_label)).shuffle(TRAIN_BUF)

#########################################################################################

model = AE(latent_dim, net_type=net_type)
model_ID = ae_type+'_ld'+str(latent_dim)+'_nt'+net_type+'_bs'+ str(batch_size)+'_buf'+str(TRAIN_BUF)

#########################################################################################

optimizer = tf.keras.optimizers.Adam(learn_rate)

for epoch in range(1, num_epochs + 1):
    t = time.time()
    last_loss = 0
    for train_x, _ in train_dataset:
        gradients, loss = AETrain.compute_gradients(model, train_x)
        AETrain.apply_gradients(optimizer, gradients, model.trainable_variables)
        last_loss = loss
    if epoch % 10 == 0:
        print('Epoch {}, Loss: {}, Remaining Time at This Epoch: {:.2f}'.format(
            epoch, last_loss, time.time() - t
        ))


#########################################################################################



n = 10

fig_dist = plt.figure(figsize=(8, 8))
ax_dist = fig_dist.add_subplot(111)

flag_sample = 1

for x_input, y_input in train_dataset:
    if flag_sample == 1:        
        x_input_sample, y_input_sample = map(lambda x: x[:n], (x_input, y_input))
        z = model.encode(x_input_sample).numpy()

        fig1, axarr1 = plt.subplots(2, n, figsize=(n, 2))
        x_input_sample = x_input_sample.numpy().reshape([n, 64, 64])
        x_output = model.decode(z).numpy().reshape([n, 64, 64])

        for i in range(n):
            axarr1[0, i].axis('off')
            axarr1[1, i].axis('off')
            axarr1[0, i].imshow(x_input_sample[i],cmap=plt.cm.gray)
            axarr1[1, i].imshow(x_output[i],cmap=plt.cm.gray)

        fig1.savefig("results/"+model_ID+"_reconstruction.png")
        
        z = model.encode(x_input)
        Z_array = z.numpy()
        Label_array = y_input
        flag_sample = 0
    else:
        z = model.encode(x_input)
        Z_array = np.concatenate((Z_array,z.numpy()), axis=0)
        Label_array = np.concatenate((Label_array,y_input.numpy()), axis=0)
        
    
ax_dist.scatter(Z_array[:,0], Z_array[:,1], color='blue',s=0.5)
    
fig_dist.savefig("results/"+model_ID+"_distribution.png")



#########################################################################################
n_samples = 500
seed = 30
rng = np.random.RandomState(seed)
# Anisotropicly distributed data
random_state = 170
no_structure = rng.rand(n_samples, 2), None
# ============
# Set up cluster parameters
# ============
fig_clustering = plt.figure(figsize=(5, 5))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)

plot_num = 1

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 2,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    "allow_single_cluster": True,
    "hdbscan_min_cluster_size": 15,
    "hdbscan_min_samples": 3,
    "random_state": 42,
}

datasets = [
    (no_structure, {}),
]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    
    params = default_base.copy()

    params.update(algo_params)
   
    X = Z_array
    y = np.zeros(X.shape[0])

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params["n_neighbors"], include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(
        n_clusters=params["n_clusters"],
        random_state=params["random_state"],
    )
    spectral = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        eigen_solver="arpack",
        affinity="nearest_neighbors",
        random_state=params["random_state"],
    )
    
    clustering_algorithms = (
        # ("MeanShift", ms),
        ("Spectral\nClustering", spectral),
        # ("MiniBatch\nKMeans", two_means),
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                + "connectivity matrix is [0-9]{1,2}"
                + " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                + " may not work as expected.",
                category=UserWarning,
            )
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)


        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        # plt.xlim(-2.5, 2.5)
        # plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        plot_num += 1

plt.show()

fig_clustering.savefig("results/"+model_ID+"_clustering.png")

#########################################################################################

dec_dir = '/net/projects/CLS/actomyosin_dynamics/data/decoded_results/'
# os.mkdir(dec_dir)
# 

now=datetime.datetime.now()
current_time = now.isoformat()[5:19]
print(current_time)


model_dec_dir = '/net/projects/CLS/actomyosin_dynamics/data/decoded_results/'+ model_ID+'_' + current_time
analysis_dir = '/net/projects/CLS/actomyosin_dynamics/analysis'

if not os.path.isdir(model_dec_dir):          
    os.mkdir(model_dec_dir)



csv_output = pd.DataFrame(columns=['title', 'labelnumber','cluster_ID'])
for x_input, y_input in train_dataset:
    z = model.encode(x_input)
    x_output = model.decode(z).numpy().reshape([len(y_input), 64, 64])
    
    # y_pred = algorithm.predict(z.numpy())
    for i in range(z.numpy().shape[0]):
        input_label = y_input[i].numpy()
        input_img = x_input[i,:,:,0].numpy().squeeze()
        output_img = x_output[i,:,:]
        label_str = str(int(input_label)).zfill(12)
        ind = np.where(Label_array == input_label)        
        pred_ID = y_pred[ind[0][0]]        
        
        title = 't'+label_str[0:4]+'x'+label_str[4:8]+'y'+label_str[8:12]+'ps64.tif'
        # tifffile.imsave(os.path.join(C0_dir,title), input_img)
        # tifffile.imwrite(os.path.join(model_dec_dir,'dec_'+title), output_img)
        
        s = pd.Series([title, input_label,pred_ID],
                           index=['title', 'labelnumber','cluster_ID'])
        csv_output = pd.concat([csv_output, s], ignore_index=True)
        


csv_output.to_csv(os.path.join(analysis_dir,model_ID+'_clustering.csv'))
                            
             
       
#########################################################################################

z0_max = np.percentile(Z_array[:,0], 99)
z0_min = np.percentile(Z_array[:,0], 1)
z1_max = np.percentile(Z_array[:,1], 99)
z1_min = np.percentile(Z_array[:,1], 1)

nxy = 10
x = np.linspace(z0_min, z0_max,nxy)
y = np.linspace(z1_min, z1_max,nxy)
xv, yv = np.meshgrid(x, y)

# plt.scatter(xv,yv)

zv = np.concatenate((xv.reshape([nxy*nxy,1]),yv.reshape([nxy*nxy,1])), axis=1)

x_output_manifold = model.decode(zv).numpy().reshape([nxy,nxy, 64, 64])


fig_manifold, axarr = plt.subplots(nxy, nxy, figsize=(nxy, nxy))
for i in range(nxy):
    for j in range(nxy):    
        axarr[i, j].axis('off')
        axarr[i, j].imshow(x_output_manifold[i,j,:,:],cmap=plt.cm.gray)
        
    
fig_manifold.savefig("results/"+model_ID+"_manifold.png")

#########################################################################################


#########################################################################################



#########################################################################################


#########################################################################################



#########################################################################################


#########################################################################################



#########################################################################################


#########################################################################################



#########################################################################################


#########################################################################################



#########################################################################################


#########################################################################################



#########################################################################################