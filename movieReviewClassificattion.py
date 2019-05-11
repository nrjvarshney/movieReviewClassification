
# coding: utf-8

# # classify review as positive or negatiave based on text content of the reviews

# In[1]:


from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# num_words = 10000 means you'll keep the top 10000 most frequently occuring words in the training data. rare words will be discarded
# 
# each review is a list of word indices (encoding a sequence of words).  train_labels and test_labels are lists of 0s and 1s, where 0 stands for negative and 1 stands for positive:

# In[2]:


len(train_data), len(test_data)


# In[4]:


test_labels[0]


# In[5]:


train_data[0]


# In[6]:


# because we are limitting the words to 10000, no word index will exceed 10000
max([max(sequence) for sequence in train_data])


# In[7]:


# decoding review back to words
word_index = imdb.get_word_index()
reverse_word_index = dict(
[(value, key) for (key,value) in word_index.items()])

decoded_review = ' '.join(
[reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# In[8]:


decoded_review


# word_index is a dict mapping words to index
# indices are offset by 3 becoz 0,1,2 are reserved indices for

# In[9]:


# encoding the input data

import numpy as np
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i,sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# one-hot-encoding of size 10000

# In[10]:


x_train[0]


# In[12]:


# vectorizing labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.array(test_labels).astype('float32')


# In[13]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# last layer is using sigmoid so as to output a prob. 


# In[14]:


model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])


# crossentropy is the best choice when you're dealing with models that output probabilities. but could use others also like mean_squared_error

# In[17]:


# can pass custom optimizers also
# custom losses 
# custom metrics
from keras import losses
from keras import metrics
from keras import optimizers

model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
             loss = losses.binary_crossentropy,
             metrics = [metrics.binary_accuracy])


# # creating validation data

# In[18]:


x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# In[19]:


model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics = ['acc'])

history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs = 20,
                   batch_size = 512,
                   validation_data = (x_val, y_val))


# In[21]:


history_dict = history.history
history_dict.keys()


# In[23]:


import matplotlib.pyplot as plt

loss_values = history_dict['loss']
val_loss_values = history_dict["val_loss"]

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label = "training loss")
# bo is for blue dot
plt.plot(epochs, val_loss_values, 'b', label = "validation loss")
# b is for solid blue line
plt.title("Training and validation loss")
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.legend()

plt.show()


# In[24]:


plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#  As you can see, the training loss decreases with every epoch, and the training accuracy
# increases with every epoch. That’s what you would expect when running gradientdescent
# optimization—the quantity you’re trying to minimize should be less with
# every iteration. But that isn’t the case for the validation loss and accuracy: they seem to
# peak at the fourth epoch. This is an example of what we warned against earlier: a
# model that performs better on the training data isn’t necessarily a model that will do
# better on data it has never seen before. In precise terms, what you’re seeing is overfitting:
# after the second epoch, you’re overoptimizing on the training data, and you end
# up learning representations that are specific to the training data and don’t generalize
# to data outside of the training set.
#  In this case, to prevent overfitting, you could stop training after three epochs.

# # retraining from scratch

# In[25]:


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


# In[26]:


results


# In[27]:


model.predict(x_test)


# As you can see, the network is confident for some samples (0.99 or more, or 0.01 or
# less) but less confident for others (0.6, 0.4).
# 
# # Further Experiments
# You used two hidden layers. Try using one or three hidden layers, and see how
# doing so affects validation and test accuracy.
#  Try using layers with more hidden units or fewer hidden units: 32 units, 64 units,
# and so on.
#  Try using the mse loss function instead of binary_crossentropy.
#  Try using the tanh activation (an activation that was popular in the early days of
# neural networks) instead of relu.
