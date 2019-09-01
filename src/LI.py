#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np
import pandas as pd
import math
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model


# In[96]:


#dataset
data_path = 'data'
PATH_X_TRAIN='./data/x_train.txt'
PATH_Y_TRAIN='./data/y_train.txt'
PATH_X_TEST='./data/x_test.txt'
PATH_Y_TEST='./data/y_test.txt'

#model 
INPUT_SIZE=1000


# In[97]:


#read dataset
def read_txt(path):
    handle = open(path, "r", encoding='utf8')
    df = pd.DataFrame(handle.readlines())
    handle.close()
    return df

x_train=read_txt(PATH_X_TRAIN)
y_train=read_txt(PATH_Y_TRAIN)
x_test=read_txt(PATH_X_TEST)
y_test=read_txt(PATH_Y_TEST)


# In[ ]:


#texts preprocessing
texts_train=x_train.values
texts_test=x_test.values
texts_train=[s[0].replace('\n','').lower() for s in texts_train]
texts_test=[s[0].replace('\n','').lower() for s in texts_test]
tokenizer=Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tokenizer.fit_on_texts(texts_train)

#sequences
texts_train=tokenizer.texts_to_sequences(texts_train)
texts_test=tokenizer.texts_to_sequences(texts_test)

texts_train=pad_sequences(texts_train, maxlen=INPUT_SIZE, padding='post')
texts_train=np.array(texts_train)
texts_test=pad_sequences(texts_test, maxlen=INPUT_SIZE, padding='post')
texts_test=np.array(texts_test)


# In[ ]:


#classes preprocessing
classes_train=y_train.values
classes_test=y_test.values

#dictionary for languages
classes = np.unique(np.array(classes_train))
nums=np.arange(len(classes))
d = dict(zip(classes,nums))

classes_train=[d[c[0]] for c in classes_train] 
classes_test=[d[c[0]] for c in classes_test] 
classes_train=to_categorical(classes_train)
classes_test=to_categorical(classes_test)


# In[ ]:


voc_size=len(tokenizer.word_index)
voc_size


# In[ ]:


#embeddings
embeddings_weights=[]
embeddings_weights.append(np.zeros(voc_size))


for char, i in tokenizer.word_index.items():
    onehot=np.zeros(voc_size)
    onehot[i-1]=1
    embeddings_weights.append(onehot)
embeddings_weights=np.array(embeddings_weights)


# In[88]:


embedding_layer=Embedding(voc_size+1,
                         voc_size,
                         input_length=1000,
                         weights=[embeddings_weights])


# In[89]:


conv_layers=[[256, 7, 3],
            [256, 7, 3],
            [256, 7, 3],
            [256, 3, -1],
            [256, 3, -1],
            [256, 3, -1],
            [256, 3, 3]]

fully_connected_layers=[1024, 1024]
num_of_classes=len(classes)
dropout=0.5
optimizer='adam'
loss='categorical_crossentropy'


# In[91]:


inputs=Input(shape=(1000,), name='input', dtype='int64')
x=embedding_layer(inputs)
for filter_num, filter_size, pooling_size in conv_layers:
    x=Conv1D(filter_num, filter_size)(x)
    x=Activation('relu')(x)
    if pooling_size!=-1:
        x=MaxPooling1D(pool_size=pooling_size)(x)
x=Flatten()(x)
for dense_size in fully_connected_layers:
    x=Dense(dense_size, activation='softmax')(x)
    x=Dropout(dropout)(x)
predictions=Dense(num_of_classes, activation='softmax')(x)
model=Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.summary()


# In[92]:


model.fit(texts_train, classes_train, 
          validation_data = (texts_test, classes_test),
         batch_size=128,
         epochs=10,
         verbose=2)


# In[ ]:




