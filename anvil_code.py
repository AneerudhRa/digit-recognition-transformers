#!/usr/bin/env python
# coding: utf-8

# In[23]:


#get_ipython().system('pip install anvil-uplink')


# In[24]:


# this is for our official site
import anvil.server
# anvil.server.connect("enter-your-anvil-key-here-and-keep-it-secret")


# In[25]:


import pandas as pd
import numpy as np
from PIL import Image
import anvil.media
from tensorflow.keras.models import load_model
from keras.models import load_model
from io import StringIO
import tensorflow as tf
import keras
from io import BytesIO


# In[26]:


class ClassTokenLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(ClassTokenLayer, self).__init__()

  # initialize class token
  def build(self, input_shape):
    initializer = tf.keras.initializers.RandomNormal()
    self.w = self.add_weight(
      shape = (1, 1, input_shape[-1]),
      initializer = initializer,
      trainable = True,
      dtype = tf.float32)

  # class token
  def call_clt(self, inputs):
    batch = tf.shape(inputs)[0]
    hidden_dim = self.w.shape[-1]
    clt = tf.broadcast_to(self.w, [batch, 1, hidden_dim])
    clt = tf.cast(clt, dtype = inputs.dtype)
    return clt

with keras.utils.custom_object_scope({'ClassToken': ClassTokenLayer}):
  model = load_model('trans.h5')


# In[27]:


@anvil.server.callable
def checking_file(file):

    file = file.get_bytes().decode('utf-8')
    file_csv = pd.read_csv(StringIO(file), header = None)
    file_array = np.array(file_csv)

    if file_array.shape != (28, 28):
        return 'Size is the problem'

    if (file_array < 0).any() or (file_array > 255).any():
        return 'pixel intensity is incorrect'

    max_val = np.max(file_array)
    if max_val <= 1:
        if np.any((file_array < 0) | (file_array > 1)):
            return 'Scaled pixel intensity is incorrect'

    return 'Success!'


# In[28]:


@anvil.server.callable
def display_image(file):

    file = file.get_bytes().decode('utf-8')
    file_csv = pd.read_csv(StringIO(file), header = None)
    file_array = np.array(file_csv)

    max_val = np.max(file_array)
    if max_val <= 1:
      file_array = (file_array * 255).astype(np.uint8)

    image_display = Image.fromarray(file_array.astype('uint8'))
    image_bytes = BytesIO()
    image_display.save(image_bytes, format = 'PNG')
    image_bytes = image_bytes.getvalue()

    return anvil.BlobMedia('image/png', image_bytes, name = 'img')


# In[29]:


@anvil.server.callable
def cnn_prediction(file):

    model = load_model('NNmodel.h5')
    file = file.get_bytes().decode('utf-8')
    file_array = pd.read_csv(StringIO(file), header = None).values.reshape(28, 28)

    max_val = np.max(file_array)
    if max_val > 1:
      file_array = file_array.astype(np.float64) / 255

    prediction = np.argmax(model.predict(np.expand_dims(file_array, axis = 0)))

    return prediction


# In[30]:


@anvil.server.callable
def transformer_prediction(file):

    file = file.get_bytes().decode('utf-8')
    file_csv = pd.read_csv(StringIO(file), header = None)
    file_array = file_csv.values.reshape(16, 49)

    max_val = np.max(file_array)
    if max_val > 1:
        file_array = file_array.astype(np.float64) / 255

    position_feed = np.array([list(range(16))])
    pred_output = model.predict([np.expand_dims(file_array, axis = 0), position_feed])
    prediction = np.argmax(pred_output)

    return prediction


# In[ ]:


anvil.server.wait_forever()

