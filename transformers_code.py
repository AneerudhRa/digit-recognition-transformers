# -*- coding: utf-8 -*

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.metrics import confusion_matrix

import anvil.server
# anvil.server.connect("enter-your-anvil-key-here-and-keep-it-secret")

"""# Data Loading"""

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

ndata_train = x_train.shape[0]
ndata_test = x_test.shape[0]

x_train = x_train.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))


xshape = x_train.shape[1:4]
xshape

"""# CNN Model"""

NNmodel = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(input_shape=xshape),
        tf.keras.layers.Dense(64,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
        ])

NNmodel.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

NNmodel.summary()

"""## Cross-Validation Test"""

NNmodel.fit(x_train,y_train,epochs=30,validation_split=0.2,batch_size=200)

"""## Presentation of Misclassification"""

y_pred = NNmodel.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

misclassified_indices = np.where(y_pred_classes != y_test)[0]

num_rows = 5
num_cols = 5
num_samples = num_rows * num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, idx in enumerate(misclassified_indices[:num_samples]):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f'ID: {idx}\nTrue: {y_test[idx]}, Predicted: {y_pred_classes[idx]}')
    ax.axis('off')

plt.tight_layout()
plt.show()

"""Here we can see, some of the misclassified images are hard to recognize even with human eyes. Therefore, the quality of data available dictates that it is impossible to achieve 100% accuracy - one can be close enough though with a well parameterized model.

The nature of data aside, there are indeed common misclassifications across certain numbers. 7 and 1 are intuitively easily mixed up, as the difference is merely the length of the little top bar; 6 and 0 could also be a common pair as it is sometimes hard to identify if a handwritten number is just a closed oval, or with a little tail on top.

The following part trains the model on the entire dataset, and shows common misclassified pairs using a confusion matrix.

## Training the Entire Dataset
"""

NNmodel = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(input_shape=xshape),
        tf.keras.layers.Dense(64,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
        ])
NNmodel.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

NNmodel.fit(x_train,y_train,epochs=30,batch_size=200)

y_pred = NNmodel.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print('This model predicts '+str(NNmodel.evaluate(x_test,y_test)[1]*100) +'% of the test data correctly')

"""Fitted the entire training data, the model predicts 99.06% test images correctly. This implies that the model successfully grasped the majority of featureness in each number, and identified handwritings quite clearly. It is quite obvious to see that numbers with unique characteristics, like 4 and 0, are hardly misclassified, while numbers that are very close to others, only differentiated by small features such as closeness of a circle (like 6 and 5) or color-filledness of a certain area (like 8 and 1), are often mislabeled.

From the confusion matrix, we can also find commonly misclassified pairs: a total of four 7s are misclassified as 1, but quite surprisingly that the model deals really well between 0 and 6. But 3 and 8 seem to be an unexpected headache - 5s through 9s are continously misclassified as 3, and almost every number can be misclassified into 8; probably because the model cannot clearly define the closedness or stand-outness of certain local features.

# Vision Transformer
"""

# this is written as a tensorflow "layer".  it's just a vector the same size as the
# output of the previous layer. the vector is initialized randomly, but we'll use
# gradient descent to update the values in the vector
#
# it's purpose is to be appended to the beginning of the sequence of vectors fed into
# the transformer.  then after the transformer runs on the whole data, we just grab
# the resulting zero-th vector...the class token...and use that as the portfolio weights
class ClassToken(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

ndata_train = x_train.shape[0]
ndata_test = x_test.shape[0]

def build_ViT(n,m,block_size,hidden_dim,num_layers,num_heads,key_dim,mlp_dim,dropout_rate,num_classes):
    # n is number of rows of blocks
    # m is number of cols of blocks
    # block_size is number of pixels (with rgb) in each block

    inp = tf.keras.layers.Input(shape=(n*m,block_size))
    inp2 = tf.keras.layers.Input(shape=(n*m))
    mid = tf.keras.layers.Dense(hidden_dim)(inp) # transform to vectors with different dimension
    # the positional embeddings
#     positions = tf.range(start=0, limit=n*m, delta=1)
    emb = tf.keras.layers.Embedding(input_dim=n*m, output_dim=hidden_dim)(inp2) # learned positional embedding for each of the n*m possible possitions
    mid = mid + emb # for some reason, tf.keras.layers.Add causes an error, but + doesn't?
    # create and append class token to beginning of all input vectors
    token = ClassToken()(mid) # append class token to beginning of sequence
    mid = tf.keras.layers.Concatenate(axis=1)([token, mid])

    for l in range(num_layers): # how many Transformer Head layers are there?
        ln  = tf.keras.layers.LayerNormalization()(mid) # normalize
        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,value_dim=key_dim)(ln,ln,ln) # self attention!
        add = tf.keras.layers.Add()([mid,mha]) # add and norm
        ln  = tf.keras.layers.LayerNormalization()(add)
        den = tf.keras.layers.Dense(mlp_dim,activation='relu')(ln) # maybe should be relu...who knows...
        den = tf.keras.layers.Dropout(dropout_rate)(den) # regularization
        den = tf.keras.layers.Dense(hidden_dim)(den) # back to the right dimensional space
        den = tf.keras.layers.Dropout(dropout_rate)(den)
        mid = tf.keras.layers.Add()([den,add]) # add and norm again
    ln = tf.keras.layers.LayerNormalization()(mid)
    fl = ln[:,0,:] # just grab the class token for each image in batch
    clas = tf.keras.layers.Dense(num_classes,activation='softmax')(fl) # probability that the image is in each category
    mod = tf.keras.models.Model([inp,inp2],clas)
    mod.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return mod

"""## Parameterization"""

n = 4
m = 4
block_size = 49
hidden_dim = 96
num_layers = 6
num_heads = 4
key_dim = hidden_dim//num_heads
mlp_dim = hidden_dim
dropout_rate = 0.1
num_classes = 10



trans = build_ViT(n,m,block_size,hidden_dim,num_layers,num_heads,key_dim,mlp_dim,dropout_rate,num_classes)
trans.summary()

x_train_ravel = np.zeros((ndata_train,n*m,block_size))
for img in range(ndata_train):
    ind = 0
    for row in range(n):
        for col in range(m):
            x_train_ravel[img,ind,:] = x_train[img,(row*7):((row+1)*7),(col*7):((col+1)*7)].ravel()
            ind += 1

x_test_ravel = np.zeros((ndata_test,n*m,block_size))
for img in range(ndata_test):
    ind = 0
    for row in range(n):
        for col in range(m):
            x_test_ravel[img,ind,:] = x_test[img,(row*7):((row+1)*7),(col*7):((col+1)*7)].ravel()
            ind += 1

pos_feed_train = np.array([list(range(n*m))]*ndata_train)
pos_feed_test = np.array([list(range(n*m))]*ndata_test)

"""## Cross-Validation Test"""

trans.fit([x_train_ravel,pos_feed_train],y_train,epochs=100,batch_size=800,validation_split=0.2)

"""The ViT model did not end up having 99% accuracy, but it is interesting to talk about the parameterization process:


1.   Patch size: it is better to use a 4×4 patch than 7×7 or 1×1 - it is not too large to over-generalize the patterns, nor too small to lose certain patterns in the continuity of lines.
2.   Hidden dimension: we experimented from 4 to 512. It turns out that a super large hidden dimension will make the model difficult to train with a slow learning process, but a too simple hidden dimension does not grasp that much information. 96 was a good number that converges quickly with good accuracy.
3.   Number of layers and heads: similar to hidden dimension, these two parameters also suffer from huge overfitting and overcomplication problems. It turns out that a simple MHA structure is enough to learn the information. Increasing these two parameters booms the computational time and difficulty of convergence drastically, so we set them as low as possible.

## Presentation of Misclassification
"""

y_pred_prob = trans.predict([x_test_ravel, pos_feed_test])
y_pred_classes = np.argmax(y_pred_prob, axis=1)

misclassified_indices = np.where(y_pred_classes != y_test)[0]

num_rows = 5
num_cols = 5
num_samples = num_rows * num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, idx in enumerate(misclassified_indices[:num_samples]):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f'ID: {idx}\nTrue: {y_test[idx]}, Predicted: {y_pred_classes[idx]}')
    ax.axis('off')

plt.tight_layout()
plt.show()

"""The transformer model does not do as well as the CNN model in identifying some clear pictures. The transformer model suffered a lot from overfitting and did not recognize certain patterns very well. One key reason could be that the MHA mechanism does not grasp local continuity that well - the model obviously did not identify the top of a number in misclassifications such as 33. Some further fine-tunings might be required to train the model, such as data augmentation methods, but the existence of unclear images still make it impossible for the model to perform 100% accurately.

## Training the Entire Dataset
"""

n = 4
m = 4
block_size = 49
hidden_dim = 96
num_layers = 6
num_heads = 4
key_dim = hidden_dim//num_heads
mlp_dim = hidden_dim
dropout_rate = 0.1
num_classes = 10



trans = build_ViT(n,m,block_size,hidden_dim,num_layers,num_heads,key_dim,mlp_dim,dropout_rate,num_classes)
trans.fit([x_train_ravel,pos_feed_train],y_train,epochs=100,batch_size=800)

y_pred_prob = trans.predict([x_test_ravel, pos_feed_test])
y_pred_classes = np.argmax(y_pred_prob, axis=1)

conf_matrix = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print('This model predicts '+str(trans.evaluate([x_test_ravel,pos_feed_test],y_test)) +'% of the test data correctly')

"""Fitted the entire training data, the model predicts 98.57% test images correctly. We can see quick convergence of the model at the beginning, but overfitting and almost no convergence over 98% accuracy. This signals that after the model exhausts its power, there are still certain local patterns it cannot recognize.

Transformer model particularly deals bad with number pairs that look alike at high level such as mosaicized 4 and 9, 5 and 6, etc. Even with positional encoding, ViT model cannot distinguish locality very clearly, therefore mixed up numbers that are similar in composition but different in specific localtions of particular patterns.
"""

anvil.server.wait_forever()