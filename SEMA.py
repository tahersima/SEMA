""" SEMA
Scanning Electron Microscopy (SEM) images Autoencoder (A)
This is a python and tensorflow script to train an unsupervised learning to 
compress SEM images to a latent space representation and reconstruct the image.
 
Author: Mohammad H. tahersima; January 2019
www.tahersima.com
"""
## Import packages and set the environment -----------------------------------
import os
import argparse
#import numpy as np
import cv2
#import pickle
import time
#from sklearn.model_selection import train_test_split
#import tensorflow as tf
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--sampleIm','--si', help='show a sample image?', required = False, type = bool, default = 'False')
parser.add_argument('--filerename','--fr', help='rename files?', required = False, type = bool, default = 'False')
parser.add_argument('--load','--l', help='load a previously saved model?', required = False, type = bool, default = 'False')
parser.add_argument('--save','--s', help='save the model?', required = False, type = bool, default = 'False')
parser.add_argument('--epoch','--e', help='number of training epochs', required = False, type = int, default = 500)
parser.add_argument('--dropout','--do', help='number between o to 1', required = False, type = float, default = 0.5)
parser.add_argument('--splitratio','--sr', help='portion of data as train data', required = False, type = float, default = 0.8)
args = parser.parse_args()

start_time = time.time()

## FUNCTIONS ------------------------------------------------------
def init_plotting():
    plt.rcParams['figure.figsize'] = (6, 6)
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['ytick.minor.size'] = 2
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'top right'
    plt.rcParams['axes.linewidth'] = 1

    plt.rcParams['image.cmap'] = 'viridis'
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')


def file_rename(): 
    i = 0
    for filename in os.listdir("Devices/"): 
        dst ="sem_" + str(i) + ".jpg"
        src ='DATA/Devices/'+ filename 
        dst ='DATA/Devices/'+ dst 
        # rename() function will rename all the files 
        os.rename(src, dst) 
        i += 1

## LOAD DATA -----------------------------------------------------------
## CHANGE DIRECTORY TO LOCATION OF FILES
print("\nCurrent directory is at " + os.getcwd())
print("\nChaning the directory ...")
os.chdir(r"...\_SEMA")
print("\nCurrent directory is at " + os.getcwd())

if args.filerename == 'True':
    file_rename()
## SHOW A SAMPLE IMAGE
if args.sampleIm == 'True':
    img = cv2.imread(r'DATA\Devices\sem_1.jpg',0)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --> import and split data
    
## Training Parameters ------------------------------------------------------
learning_rate = 0.001
batch_size = 100
epochN= args.epoch
epochLength=train_data.shape[0]

training_steps = epochN*(epochLength//batch_size)
training_steps = epochN*(test_data.shape[0]*2//batch_size)
display_step=training_steps//epochN

hidden1=100
hidden2=50
num_input=image_size_1 * image_size_2

logs_path=r"/logs"
tf.reset_default_graph()

## Tensorflow Graph -------------------------------------------------
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, hidden1])),
    'encoder_h2': tf.Variable(tf.random_normal([hidden1, hidden2])),
    'decoder_h1': tf.Variable(tf.random_normal([hidden2, hidden1])),
    'decoder_h2': tf.Variable(tf.random_normal([hidden1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([hidden1])),
    'encoder_b2': tf.Variable(tf.random_normal([hidden2])),
    'decoder_b1': tf.Variable(tf.random_normal([hidden1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2
# Building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

# Construct model
encode = encoder(X)
decode = decoder(encode)

y_pred = decode # Prediction: reconstructed image
y_true = X # Targets: the input data

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver() # saver object to save all variables
tf.summary.scalar ("loss", loss) # summary for loss and/or accuracy
summary = tf.summary.merge_all() # merge all summaries

## Start the Session --------------------------------------------------------
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init) 
    writer = tf.summary.FileWriter (logs_path, graph=tf.get_default_graph())
    if args.load == 'True':
        saver.restore(sess, ".\SEMA")
        
    for epoch in range (1, epochN):
        ## --> train
        
        ## --> test
        
# Plot loss vs step  --------------------------------------------------------
print("\n\n It took %s seconds to run this script" %(time.time() - start_time))
print("\n\n Loss at each step")
plt.figure()
## --> plot loss

# Visulizzzz! ---------------------------------------------------------------
print("\nOriginal Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper")
plt.show()

print("\nReconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper")
plt.show()