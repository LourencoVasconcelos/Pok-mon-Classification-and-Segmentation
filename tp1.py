#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""

from tp1_utils import load_data
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Input, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import tp1_utils
from tensorflow.keras.applications import VGG16, InceptionV3, MobileNetV2, ResNet50V2, mobilenet_v2, vgg16, inception_v3, resnet_v2
from tensorflow.keras.callbacks import EarlyStopping

data = load_data()
valid_X = data['train_X'][:500]
train_X = data['train_X'][500:]
test_X = data['test_X']

valid_masks = data['train_masks'][:500]
train_masks = data['train_masks'][500:]
test_masks = data['test_masks']

valid_classes = data['train_classes'][:500]
train_classes = data['train_classes'][500:]
test_classes = data['test_classes']

valid_labels = data['train_labels'][:500]
train_labels = data['train_labels'][500:]
test_labels = data['test_labels']

transfer_train = mobilenet_v2.preprocess_input(train_X*255)
transfer_train = mobilenet_v2.preprocess_input(valid_X*255)
transfer_train = mobilenet_v2.preprocess_input(test_X*255)

def plot_accuracy(histories, legends, filename, metric = 'val_categorical_accuracy'):
    for history in histories:
        plt.plot(history.history[metric])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(legends, loc='lower right')
    plt.savefig(filename)
    plt.close()

def plot_accuracy2(history, filename, metric='categorical_accuracy'):
    train = history.history[metric]
    val = history.history['val_' + metric]
    plt.plot(train, 'g', label='Training')
    plt.plot(val, 'b', label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filename)



def ex1(dropout = 0.5, batch_size = 50, optimizer=Adam(learning_rate=0.0005)):
    callback = EarlyStopping(monitor='val_loss', patience = 10)
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(32, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(32, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(32, (6, 6), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(6, 6))(layer)
    

    features = Flatten(name='features')(layer)
    layer = Dense(100, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(dropout)(layer)
    layer = Dense(25, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(dropout)(layer)
    layer = Dense(10)(layer)
    layer = Activation("softmax")(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = model.fit(train_X, train_classes, validation_data=(valid_X,valid_classes), batch_size=batch_size, epochs=50, callbacks=[callback])
    return history, model





def ex1_2_layers():
    callback = EarlyStopping(monitor='val_loss', patience = 5)
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(32, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(32, (6, 6), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(6, 6))(layer)
    

    features = Flatten(name='features')(layer)
    layer = Dense(100, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(25, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(10)(layer)
    layer = Activation("softmax")(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = model.fit(train_X, train_classes, validation_data=(valid_X,valid_classes), batch_size=15, epochs=500, callbacks=[callback])
    return history

def ex1_4_layers():
    callback = EarlyStopping(monitor='val_loss', patience = 5)
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(32, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(32, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(32, (6, 6), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(32, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(6, 6))(layer)

    features = Flatten(name='features')(layer)
    layer = Dense(100, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(25, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(10)(layer)
    layer = Activation("softmax")(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = model.fit(train_X, train_classes, validation_data=(valid_X,valid_classes), batch_size=15, epochs=500, callbacks=[callback])
    return history

def ex1_more_nodes():
    callback = EarlyStopping(monitor='val_loss', patience = 5)
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(32, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(32, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(32, (6, 6), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(6, 6))(layer)
    

    features = Flatten(name='features')(layer)
    layer = Dense(100, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(25, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(10)(layer)
    layer = Activation("softmax")(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = model.fit(train_X, train_classes, validation_data=(valid_X,valid_classes), batch_size=15, epochs=500, callbacks=[callback])
    return history

def ex1_less_nodes():
    callback = EarlyStopping(monitor='val_loss', patience = 5)
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(16, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(16, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(16, (6, 6), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(6, 6))(layer)
    

    features = Flatten(name='features')(layer)
    layer = Dense(100, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(25, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(10)(layer)
    layer = Activation("softmax")(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = model.fit(train_X, train_classes, validation_data=(valid_X,valid_classes), batch_size=15, epochs=500, callbacks=[callback])
    return history
    
def ex1_baseline():
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(32, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(32, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)


    features = Flatten(name='features')(layer)
    layer = Dense(512)(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(10)(layer)
    layer = Activation("softmax")(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = model.fit(train_X, train_labels, validation_data=(valid_X,valid_labels), batch_size=15, epochs=30)
    return history

def ex2_more_cneurons(batch_size = 15, optimizer= Adam(learning_rate=0.0005)):
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(128, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    
    
    layer = Conv2D(128, (6, 6), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(6, 6), padding = "same")(layer)
    
    
    features = Flatten(name='features')(layer)
    layer = Dense(100, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(50, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)    
    layer = Dense(10)(layer)
    layer = Activation("sigmoid")(layer)        
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(train_X, train_labels, validation_data=(valid_X,valid_labels), batch_size=batch_size, epochs=50)
    return history

def ex2_growing_cneurons(batch_size = 15, optimizer= Adam(learning_rate=0.0005)):
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(32, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    
    
    layer = Conv2D(128, (6, 6), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(6, 6), padding = "same")(layer)
    
    
    features = Flatten(name='features')(layer)
    layer = Dense(100, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(50, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)    
    layer = Dense(10)(layer)
    layer = Activation("sigmoid")(layer)        
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(train_X, train_labels, validation_data=(valid_X,valid_labels), batch_size=batch_size, epochs=50)
    return history

def ex2(batch_size = 15, optimizer= Adam(learning_rate=0.0005)):
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(128, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    
    
    layer = Conv2D(32, (6, 6), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(6, 6), padding = "same")(layer)
    
    
    features = Flatten(name='features')(layer)
    layer = Dense(100, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(50, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)    
    layer = Dense(10)(layer)
    layer = Activation("sigmoid")(layer)        
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(train_X, train_labels, validation_data=(valid_X,valid_labels), batch_size=batch_size, epochs=50)
    return history, model

def ex2_100_25(batch_size = 15, optimizer= Adam(learning_rate=0.0005)):
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(128, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    
    
    layer = Conv2D(32, (6, 6), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(6, 6), padding = "same")(layer)
    
    
    features = Flatten(name='features')(layer)
    layer = Dense(100, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(50, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)    
    layer = Dense(10)(layer)
    layer = Activation("sigmoid")(layer)        
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(train_X, train_labels, validation_data=(valid_X,valid_labels), batch_size=batch_size, epochs=50)
    return history

def ex2_100_50(batch_size = 15, optimizer= Adam(learning_rate=0.0005)):
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(128, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    
    
    layer = Conv2D(32, (6, 6), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(6, 6), padding = "same")(layer)
    
    
    features = Flatten(name='features')(layer)
    layer = Dense(100, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(50, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)    
    layer = Dense(10)(layer)
    layer = Activation("sigmoid")(layer)        
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(train_X, train_labels, validation_data=(valid_X,valid_labels), batch_size=batch_size, epochs=50)
    return history

def ex2_200_100(batch_size = 15, optimizer= Adam(learning_rate=0.0005)):
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(128, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    
    
    layer = Conv2D(32, (6, 6), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(6, 6), padding = "same")(layer)
    
    
    features = Flatten(name='features')(layer)
    layer = Dense(100, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(50, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)    
    layer = Dense(10)(layer)
    layer = Activation("sigmoid")(layer)        
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(train_X, train_labels, validation_data=(valid_X,valid_labels), batch_size=batch_size, epochs=50)
    return history

def ex2_baseline(batch_size = 15, optimizer= Adam(learning_rate=0.0005)):
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(32, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    layer = Conv2D(32, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(3, 3))(layer)
    
    
    layer = Conv2D(32, (6, 6), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(6, 6), padding = "same")(layer)
    
    
    features = Flatten(name='features')(layer)
    layer = Dense(100, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(25, activity_regularizer=l2())(features)
    layer = Activation("relu")(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.5)(layer)    
    layer = Dense(10)(layer)
    layer = Activation("sigmoid")(layer)        
    model = Model(inputs=inputs, outputs=layer)
    model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(train_X, train_labels, validation_data=(valid_X,valid_labels), batch_size=batch_size, epochs=50)
    return history

def ex3(learning_rate=0.0005, batch_size=15):
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(64, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer) 
    
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = MaxPooling2D(pool_size=(2, 2), padding = "same")(layer)
    
    layer = Conv2D(256, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = Conv2D(256, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  

    
    layer = Conv2D(256, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(256, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = UpSampling2D(size=(2,2))(layer)
    
    layer = Conv2D(128, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(128, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = UpSampling2D(size=(2,2))(layer)

    layer = Conv2D(64, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(64, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    
    layer = Conv2D(1, (2,2), padding ="same")(layer)
    layer = Activation("sigmoid")(layer)
    
    
    model = Model(inputs = inputs, outputs = layer)
    model.compile(optimizer = Adam(learning_rate), loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(train_X, train_masks, validation_data=(valid_X,valid_masks), batch_size=batch_size, epochs=30)
    
    predicts = model.predict(test_X)
    tp1_utils.overlay_masks('test_overlay.png',test_X[:20],predicts[:20],width=10)
    tp1_utils.compare_masks('compare_masks.png', test_masks[:20],predicts[:20], width=10)
    return history, model


def ex3_1transposed():
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(64, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer) 
    
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = MaxPooling2D(pool_size=(2, 2), padding = "same")(layer)
    
    layer = Conv2D(256, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  

    
    layer = Conv2DTranspose(256, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = UpSampling2D(size=(2,2))(layer)
    
    layer = Conv2DTranspose(128, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = UpSampling2D(size=(2,2))(layer)

    layer = Conv2DTranspose(64, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    
    layer = Conv2DTranspose(1, (2,2), padding ="same")(layer)
    layer = Activation("sigmoid")(layer)
    
    
    model = Model(inputs = inputs, outputs = layer)
    model.compile(optimizer = Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['binary_accuracy'])
    #model.summary()
    history = model.fit(train_X, train_masks, validation_data=(valid_X,valid_masks), batch_size=15, epochs=30)
    
    predicts = model.predict(test_X)
    tp1_utils.overlay_masks('test_overlay.png',test_X[:20],predicts[:20],width=10)
    tp1_utils.compare_masks('compare_masks.png', test_masks[:20],predicts[:20], width=10)
    
    return history

def ex3_1layer_conv():
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(64, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer) 
    
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = MaxPooling2D(pool_size=(2, 2), padding = "same")(layer)
    
    layer = Conv2D(256, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  

    
    layer = Conv2D(256, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = UpSampling2D(size=(2,2))(layer)
    
    layer = Conv2D(128, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = UpSampling2D(size=(2,2))(layer)

    layer = Conv2D(64, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    
    layer = Conv2D(1, (2,2), padding ="same")(layer)
    layer = Activation("sigmoid")(layer)
    
    
    model = Model(inputs = inputs, outputs = layer)
    model.compile(optimizer = Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['binary_accuracy'])
    #model.summary()
    history = model.fit(train_X, train_masks, validation_data=(valid_X,valid_masks), batch_size=15, epochs=30)
    
    predicts = model.predict(test_X)
    tp1_utils.overlay_masks('test_overlay.png',test_X[:20],predicts[:20],width=10)
    tp1_utils.compare_masks('compare_masks.png', test_masks[:20],predicts[:20], width=10)
    
    return history



def ex3_3convolutions():
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(64, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer) 
    
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = MaxPooling2D(pool_size=(2, 2), padding = "same")(layer)
    
    layer = Conv2D(256, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = Conv2D(256, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = Conv2D(256, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  

    
    layer = Conv2D(256, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(256, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(256, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = UpSampling2D(size=(2,2))(layer)
    
    layer = Conv2D(128, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(128, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(128, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = UpSampling2D(size=(2,2))(layer)

    layer = Conv2D(64, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(64, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(64, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    
    layer = Conv2D(1, (2,2), padding ="same")(layer)
    layer = Activation("sigmoid")(layer)
    
    
    model = Model(inputs = inputs, outputs = layer)
    model.compile(optimizer = Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(train_X, train_masks, validation_data=(valid_X,valid_masks), batch_size=15, epochs=30)
    
    predicts = model.predict(test_X)
    tp1_utils.overlay_masks('test_overlay_baseline.png',test_X[:20],predicts[:20],width=10)
    tp1_utils.compare_masks('compare_masks_baseline.png', test_masks[:20],predicts[:20], width=10)
    return history

def ex3_3convolutions_transposed():
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(64, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer) 
    
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = MaxPooling2D(pool_size=(2, 2), padding = "same")(layer)
    
    layer = Conv2D(256, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = Conv2D(256, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = Conv2D(256, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  

    
    layer = Conv2DTranspose(256, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2DTranspose(256, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2DTranspose(256, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = UpSampling2D(size=(2,2))(layer)
    
    layer = Conv2DTranspose(128, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2DTranspose(128, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2DTranspose(128, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = UpSampling2D(size=(2,2))(layer)

    layer = Conv2DTranspose(64, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2DTranspose(64, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2DTranspose(64, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    
    layer = Conv2DTranspose(1, (2,2), padding ="same")(layer)
    layer = Activation("sigmoid")(layer)
    
    
    model = Model(inputs = inputs, outputs = layer)
    model.compile(optimizer = Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(train_X, train_masks, validation_data=(valid_X,valid_masks), batch_size=15, epochs=30)
    
    predicts = model.predict(test_X)
    tp1_utils.overlay_masks('test_overlay_baseline.png',test_X[:20],predicts[:20],width=10)
    tp1_utils.compare_masks('compare_masks_baseline.png', test_masks[:20],predicts[:20], width=10)
    return history

def ex3_2convolutions_bpool():
    inputs = Input(shape=(64,64,3),name='inputs')
    layer = Conv2D(64, (3, 3), padding="same")(inputs)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2D(64, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer) 
    
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = Conv2D(128, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = MaxPooling2D(pool_size=(2, 2), padding = "same")(layer)
    
    layer = Conv2D(256, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  
    layer = Conv2D(256, (3, 3), padding="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)  

    
    layer = Conv2DTranspose(256, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2DTranspose(256, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = UpSampling2D(size=(2,2))(layer)
    
    layer = Conv2DTranspose(128, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2DTranspose(128, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = UpSampling2D(size=(2,2))(layer)

    layer = Conv2DTranspose(64, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    layer = Conv2DTranspose(64, (3,3), padding ="same")(layer)
    layer = Activation("relu")(layer)
    layer = BatchNormalization(axis=-1)(layer)
    
    layer = Conv2DTranspose(1, (2,2), padding ="same")(layer)
    layer = Activation("sigmoid")(layer)
    
    
    model = Model(inputs = inputs, outputs = layer)
    model.compile(optimizer = Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['binary_accuracy'])
    #model.summary()
    history = model.fit(train_X, train_masks, validation_data=(valid_X,valid_masks), batch_size=15, epochs=30)
    
    predicts = model.predict(test_X)
    tp1_utils.overlay_masks('test_overlay_2conv.png',test_X[:20],predicts[:20],width=10)
    tp1_utils.compare_masks('compare_masks_2conv.png', test_masks[:20],predicts[:20], width=10)
    return history



def ex4_1(keras_class=VGG16, k_class=vgg16):
    
    T_X = k_class.preprocess_input(train_X*255)
    V_X = k_class.preprocess_input(valid_X*255)
    inputs = Input(shape=(64,64,3),name='inputs')
    inputs = UpSampling2D(size=(2,2))(inputs)

    pre_model = keras_class(weights='imagenet', input_tensor=inputs, include_top=False)
    for layer in pre_model.layers:
        layer.trainable = False

    model = models.Sequential()
    model.add(pre_model)
    model.add(Flatten(name='features'))
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(optimizer = Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = model.fit(T_X, train_classes, validation_data=(V_X,valid_classes), batch_size=25, epochs=30)
    return history, model

def ex4_2(keras_class=VGG16, k_class=vgg16):
    
    T_X = k_class.preprocess_input(train_X*255)
    V_X = k_class.preprocess_input(valid_X*255)
    
    inputs = Input(shape=(64,64,3),name='inputs')
    inputs = UpSampling2D(size=(2,2))(inputs)

    pre_model = keras_class(weights='imagenet', input_tensor=inputs, include_top=False)
    for layer in pre_model.layers:
        layer.trainable = False

    model = models.Sequential()
    model.add(pre_model)
    model.add(Flatten(name='features'))
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation("sigmoid"))
    model.compile(optimizer = Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(T_X, train_labels, validation_data=(V_X,valid_labels), batch_size=25, epochs=30)
    return history, model

def ex4_1_trained(keras_class):
    inputs = Input(shape=(64,64,3),name='inputs')
    inputs = UpSampling2D(size=(2,2))(inputs)

    pre_model = keras_class(weights='imagenet', input_tensor=inputs, include_top=False)
    pre_model.summary()
    model = models.Sequential()
    model.add(pre_model)
    model.add(Flatten(name='features'))
    model.add(Dense(1000))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.9))
    model.add(Dense(250))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.9))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = model.fit(train_X, train_classes, validation_data=(valid_X,valid_classes), batch_size=25, epochs=50)
    return history

# %% Best models

#_,model1 = ex1()
#_,model2 = ex2()
#_,model3 = ex3()
#_,model4 = ex4_1()
#_,model5 = ex4_2()
#model1.evaluate(test_X, test_classes)
#model2.evaluate(test_X, test_labels)
#model3.evaluate(test_X, test_masks)
#model4.evaluate(vgg16.preprocess_input(test_X*255), test_classes)
#model5.evaluate(vgg16.preprocess_input(test_X*255), test_labels)


# %% Some of the plots


#plot_accuracy([ex3(), ex3(batch_size=50), ex3(batch_size=100), ex3(batch_size=150)], ['Batch size=15','Batch size=50', 'Batch size=100', 'Batch size=150'], 'ex3_batch_size.png', metric='val_binary_accuracy')

#plot_accuracy([ex3(), ex3(learning_rate=0.0001), ex3(learning_rate=0.001), ex3(learning_rate=0.01)], ['Learning Rate=0.0005','Learning Rate=0.0001', 'Learning Rate=0.001', 'Learning Rate=0.01'], 'ex3_learning_rates.png', metric='val_binary_accuracy')


#his1 = ex3()
#plot_accuracy([his1, ex3_2convolutions_bpool()], ['Convolutional only','Convolutional and Transposed layers'], 'ex3_baseline_vs_transposed.png', metric='val_binary_accuracy')
#plot_accuracy([his1, ex3_1layer_conv()], ['2 Convolutionals','1 Convolutional'], 'ex3_baseline_vs_single.png', metric='val_binary_accuracy')
#plot_accuracy([his1, ex3_3convolutions()], ['2 Convolutionals','3 Convolutional'], 'ex3_baseline_vs_3conv.png', metric='val_binary_accuracy')
#plot_accuracy([his1, ex3_3convolutions_transposed()], ['2 Convolutionals','3 Convolutional Transposed'], 'ex3_baseline_vs_3transposed.png', metric='val_binary_accuracy')
#plot_accuracy([his1, ex3_1transposed()], ['2 Convolutionals','1 Convolutional Transposed'], 'ex3_baseline_vs_1transposed.png', metric='val_binary_accuracy')

#plot_accuracy([ex2_100_25(), ex2_100_50(), ex2_200_100()], ['dense layers = 100-25-10', 'dense layers = 100-50-10', 'dense layers = 200-100-10'], 'ex2_dense_layers.png', metric='val_binary_accuracy')
#plot_accuracy2(ex4_1_trained(MobileNetV2), 'ex4_1_trainedMobile.png', metric='categorical_accuracy')

#ex1()
#ex2()
#ex3()
#his1,_ = ex4_1(MobileNetV2, mobilenet_v2)
#his2,_ = ex4_1(InceptionV3, inception_v3)
#his3,_ = ex4_1(VGG16, vgg16)
#his4,_ = ex4_1(ResNet50V2, resnet_v2)
#plot_accuracy([his1, his2, his3, his4],['MobileNetV2', 'InceptionV3', 'VGG16', 'ResNet50V2'], 'ex6_Networks_ex1_train.png', metric='categorical_accuracy')
#plot_accuracy([his1, his2, his3, his4],['MobileNetV2', 'InceptionV3', 'VGG16', 'ResNet50V2'], 'ex6_Networks_ex1_valid.png', metric='val_categorical_accuracy')

#his1,_ = ex4_2(MobileNetV2, mobilenet_v2)
#his2,_ = ex4_2(InceptionV3, inception_v3)
#his3,_ = ex4_2(VGG16, vgg16)
#his4,_ = ex4_2(ResNet50V2, resnet_v2)
#plot_accuracy([his1, his2, his3, his4],['MobileNetV2', 'InceptionV3', 'VGG16', 'ResNet50V2'], 'ex6_Networks_ex2_train.png', metric='binary_accuracy')
#plot_accuracy([his1, his2, his3, his4],['MobileNetV2', 'InceptionV3', 'VGG16', 'ResNet50V2'], 'ex6_Networks_ex2_valid.png', metric='val_binary_accuracy')

