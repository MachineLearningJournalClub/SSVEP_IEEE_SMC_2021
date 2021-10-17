#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io
import random
import xgboost as xgb
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn import metrics, decomposition

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import std
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.keras.layers.core import RepeatVector
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.util.nest import _IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ


from load_data import load_patient_trial


#%%
SAMPLE_LEN = 5
channels = 8
latent_dim = 3

session_data_1, labels, session_data_2, labels = load_patient_trial(path = '/home/luca/Desktop/Ricerca/MLJC/SSVEP_IEEE_SMC_2021/new_data/subject_1_fvep_led_training_1.mat')

x_train, x_test, y_train, y_test = train_test_split(session_data_1, session_data_1, train_size = 0.85)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size = 0.5)

#%%
new_batch = []
for i, batch in enumerate(session_data_1):
    print(batch.shape[1]//SAMPLE_LEN)
    new_batch.extend(np.array_split(batch, batch.shape[1]//SAMPLE_LEN, axis = 1)[1:batch.shape[1]//SAMPLE_LEN-2])


#%%
np.array(new_batch)

#%%

def create_model():

    input = keras.Input(shape = (sample_len,channels))  # Variable-length sequence of ints
    x = layers.LSTM(64, return_sequences=True)(input)
    x = layers.LSTM(64, return_sequences=True)(x)
    #x = layers.RepeatVector(sample_len)(x)
    #x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(16))(x)
    x = layers.Dense(latent_dim, name = 'embedding')(x)
    x = layers.Dense(16)(x)
    output = layers.Dense(channels, activation='linear')(x)

    model = keras.Model(inputs=[input],outputs=[output])

    return model

model = create_model()

keras.utils.plot_model(model, "keras_LSTM_autoencoder.png", show_shapes=True)


#%%
model.summary()

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse') #tf.keras.metrics.mean_squared_error

history = model.fit(x_train, y_train, 
                    validation_data = (x_val, y_val), 
                    epochs=100, batch_size=32)

plt.plot(history.history['loss'])
plt.yscale("log")
plt.show()


def get_embedding(sequences, model):
    
    encoder_model = keras.Model(inputs = model.input, 
                                outputs = model.get_layer('embedding').output)
    try:
        emb = encoder_model.predict(sequences)[:,-1,:]
    except:
        sequences = list(sequences)
        emb = encoder_model.predict(sequences)[:,-1,:]
    return emb

def plot_3D_embedding(sequences, model):
    emb_vec = get_embedding(sequences, model)

    try:
        emb_vec = emb_vec.reshape(3,len(list(emb_vec)))
    except:
        'Embedding dim != 3'

    fig = go.Figure(data=[go.Scatter3d(x=emb_vec[0], y=emb_vec[1], z=emb_vec[2],
                                    mode='markers')])
    fig.show()

def plot_curve_comparison(sequences, model, n_sample = 1, title = ''):

    for i in range(n_sample):
        pred = model.predict(sequences[i])

        plt.plot(sequences[i], label = 'Truth')
        plt.plot(pred.reshape(sample_len), label = 'Pred')
        plt.title(title)
        plt.legend()
        plt.show()

def plot_reconstruction_error(sequences, model, n_sample = 1, title = ''):

    res = model.predict(sequences)
    
    try:
        s_len = len(sequences)
    except:
        s_len = sequences.shape[0]

    for i in np.random.randint(0, s_len - 1, size = n_sample):
        plt.plot(res[i] - sequences[i])
        plt.title(title)
        plt.show()

#%%

plot_reconstruction_error(x_train, model, 2, title = 'Train: Rec - Truth')
plot_reconstruction_error(x_test, model, 2, title = 'Test: Rec - Truth')
plot_reconstruction_error(x_val, model, 2, title = 'Val: Rec - Truth')

if channels == 1:
    plot_curve_comparison(x_train, model, title = 'Train')
    plot_curve_comparison(x_test, model, title = 'Test')
    plot_curve_comparison(x_val, model, title = 'Validation')

if latent_dim == 3:
    plot_3D_embedding(x_train, model)
    plot_3D_embedding(x_test, model)
    plot_3D_embedding(x_val, model)

#%%

if channels == 1:
    def stress_seq(rand = 'normal'):
        ampl = np.random.uniform(0.1, 1.)
        freq = np.random.uniform(0.01, 0.1)
        high_noise = np.random.uniform(0.1, 0.5)
        low_noise  = np.random.uniform(0.1, 0.5)
        std = np.random.uniform(0.1,0.3)

        if rand == 'normal':
            recorded_data = ampl*np.random.normal(0, std, size = (channels, sample_len))
        else:
            recorded_data = ampl*np.random.uniform(-low_noise, high_noise, size=( channels, sample_len))
        
        recorded_data += ampl*np.cos(freq*np.arange(0,sample_len))

        #recorded_data = np.array([np.cumsum(j) for j in recorded_data])
        data = recorded_data.reshape(1, sample_len, channels)
        return recorded_data, data

    stress_test = [stress_seq()[1] for i in range(10)]

    for i in stress_test:
        preds = model.predict(i)

        for j in range(channels):
            plt.plot(i.reshape(sample_len))

        plt.plot(preds[0])
        plt.show()

# %%
if channels == 1:
    for patient,p in zip(data, range(len(data))):
        print('Patient number %d ' % (p + 1))
        for channel,i in zip(patient, range(len(patient))):
            
            pred = model.predict(channel)

            rmse = np.sqrt(metrics.mean_squared_error(channel.reshape(-1), pred.reshape(-1)))

            print('\tPred on channel %d has RMSE of %f' % (i, rmse))

# %%

if channels == 1:
    # SINGLE CHANNEL
    for patient,p in zip(data_labeled, range(len(data_labeled))):
        print('\n\nPatient number %d ' % (p + 1))
        for channel,i in zip(patient, range(len(patient))):
            
            data_serie = list([list(i) for i in channel[:,0]])
            label_serie = list([list(i) for i in channel[:,1]])

            emb_serie = get_embedding(data_serie, model)

            x_b_train, x_b_test, y_b_train, y_b_test = train_test_split(emb_serie, label_serie, test_size = 0.2, stratify=label_serie)

            xgb_classifier = xgb.XGBClassifier(use_label_encoder=False)
            xgb_classifier.fit(x_b_train, y_b_train, eval_metric='error')

            xgb_preds = xgb_classifier.predict(x_b_test)

            rmse = np.sqrt(metrics.mean_squared_error(channel.reshape(-1), channel.reshape(-1)))
            
            print('\tChannel %d - Classifier ROC-AUC: %f' % (i,np.sqrt(metrics.roc_auc_score(y_b_test, xgb_preds))))
            
    ##################################
    # UNION OF CHANNELS

    for patient,p in zip(data_labeled, range(len(data_labeled))):
        print('\n\nPatient number %d ' % (p + 1))
        emb_serie = []
        for channel,i in zip(patient, range(len(patient))):
            
            data_serie = list([list(i) for i in channel[:,0]])
            label_serie = list([list(i) for i in channel[:,1]])

            emb = get_embedding(data_serie, model)
            emb = list(list(i) for i in emb)

            if i == 0:
                emb_serie = emb
            else:
                for j in range(len(channel)):
                    for k in emb[j]:
                        emb_serie[j].append(k)

        x_b_train, x_b_test, y_b_train, y_b_test = train_test_split(np.array(emb_serie), label_serie, test_size = 0.2, stratify=label_serie)

        xgb_classifier = xgb.XGBClassifier(use_label_encoder=False)
        xgb_classifier.fit(x_b_train, y_b_train, eval_metric='error')

        xgb_preds = xgb_classifier.predict(x_b_test)
        
        print('\tClassifier ROC-AUC: %f' % np.sqrt(metrics.roc_auc_score(y_b_test, xgb_preds)))

    ####################################
    # ONE CHANNEL ALONE

    labeled_seq = np.array([[serie[i: i + sample_len],matrixS1['trig'][i]] for i in range(len(serie)) if matrixS1['trig'][i] != 0][:-20])

    data_serie = list([list(i) for i in labeled_seq[:,0]])
    label_serie = list([list(i) for i in labeled_seq[:,1]])

    emb_serie = get_embedding(data_serie, model)

    x_b_train, x_b_test, y_b_train, y_b_test = train_test_split(emb_serie, label_serie, test_size = 0.2, stratify=label_serie)

    xgb_classifier = xgb.XGBClassifier(use_label_encoder=True)
    xgb_classifier.fit(x_b_train, y_b_train)

    xgb_preds = xgb_classifier.predict(x_b_test)
#%%
   
if channels != 1:
    def labeled_multichannel_data_factory():
        data = []
        labels = []
        for i in range(5):
            patient = []
            patient_lab = []
            matrixS = scipy.io.loadmat('./data/wetransfer-01a2dc/p300/S' + str(i+1) + '.mat')
            for serie in matrixS['y'].T:
                fs = matrixS['fs'][0,0]
                offset = int(0.3*fs)

                seq = np.array([serie[i + offset - int(sample_len/2): i + offset + int(sample_len/2)] for i in range(len(serie)) if matrixS['trig'][i] != 0][:-20])
                label = np.array([0.5*matrixS['trig'][i] + 0.5 for i in range(len(serie)) if matrixS['trig'][i] != 0][:-20])
                patient.append(seq)
                patient_lab.append(label)
            
            patient = np.array(patient).reshape(-1,sample_len,channels)
            label = np.array(patient_lab)
            data.append(patient)
            labels.append(label[0])
        return data, labels

    dat, lab = labeled_multichannel_data_factory()
    dat = dat/np.max(data[0])*10

    for patient,i in zip (dat, range(len(dat))):
        emb = get_embedding(patient, model)

        x_b_train, x_b_test, y_b_train, y_b_test = train_test_split(emb, lab[i], test_size = 0.2, stratify = lab[i])

        xgb_classifier = xgb.XGBClassifier(use_label_encoder=False)
        xgb_classifier.fit(x_b_train, y_b_train, eval_metric='error')

        xgb_preds = xgb_classifier.predict(x_b_test)

        print('\nPatient number %d' % i)
        print('Classifier ROC-AUC: %f' % np.sqrt(metrics.roc_auc_score(y_b_test, xgb_preds)))

        confusion_matrix = metrics.confusion_matrix(y_b_test, xgb_preds)
        cm_plot = metrics.ConfusionMatrixDisplay(confusion_matrix)
        cm_plot.plot()

        plt.show()
        #xgb.plot_importance(xgb_classifier)


# %%

plot_pairplot = True

if channels != 1:
    for patient,i in zip (dat, range(len(dat))):
        emb = get_embedding(patient, model)
        pd_emb = pd.DataFrame(emb)
        pd_emb['signal'] = lab[i]

        if plot_pairplot:
            g = sns.pairplot(pd_emb, corner = True, hue = 'signal')
            g.map_lower(sns.kdeplot, levels=3, color=".2")
            plt.show()

        pca = decomposition.PCA(n_components=2)
        emb_pca = pca.fit(emb).transform(emb)

        plt.scatter([emb_pca.T[0][j] for j in range(len(emb_pca)) if lab[i][j] == 0], 
                    [emb_pca.T[1][j] for j in range(len(emb_pca)) if lab[i][j] == 0],
                    label = 'No signal')
        plt.scatter([emb_pca.T[0][j] for j in range(len(emb_pca)) if lab[i][j] == 1], 
                    [emb_pca.T[1][j] for j in range(len(emb_pca)) if lab[i][j] == 1],
                    label = 'Signal')
        plt.legend()
        plt.title('PCA')
        plt.show()





# %%
