from pathlib import Path
import mat73
import mne

def load_session(path, sampling_freq, eeg_ch_names):
    # load raw numpy data
    mat = mat73.loadmat(path)
    mat = mat['y']

    # build metadata structure
    ch_names = eeg_ch_names + ['stim1'] 
    ch_types = (['eeg'] * 8) + ['stim']
    info = mne.create_info(ch_names, sfreq=sampling_freq, ch_types=ch_types)

    mat[1:9] = mat[1:9] * 1E-6 # mne expect raw data in Volt, loaded data is in micro volts!
    raw = mne.io.RawArray(mat[1:10], info) # index 1-10 only for EEG and stim channels

    return raw

def add_labels_to_stim(raw, n_trials=4):
    events = mne.find_events(raw, stim_channel='stim1', output='step')

    on_set_mask = events[:,2] == 1 # stim channel goes from 0 to 1
    on_set_events = events[on_set_mask, 0]

    off_set_mask = events[:,2] == 0  # stim channel goes from 1 to 0
    off_set_events = events[off_set_mask,0]

    raw_copy = raw.copy() # do not effect the original raw signal
    data = raw_copy._data # directly access the underlying numpy data!

    # loop trough events and encode the label as a stim channel event_id
    for i, (on_set, off_set) in enumerate(zip(on_set_events, off_set_events)):
        trial_label = (i % n_trials) +1 # just a trick to avoid looping i in range(1,5)
        data[8, on_set:off_set] = trial_label # modify the stim channel (index 8)

    return raw_copy



def get_trials(patient_session_path):

    SFREQ = 256
    CH_NAMES = ['P1','P3','P5','P7','P10','O2','O7','O8']

    raw = load_session(patient_session_path, SFREQ, CH_NAMES)
    raw2 = add_labels_to_stim(raw)

    modified_stim, times = raw2[8] # 8 index of stim channel

    stim_events = {'9Hz': 1, '10Hz': 2, '12Hz': 3, '15Hz': 4}
    events = mne.find_events(raw2, stim_channel='stim1')
    TRIAL_DURATION = 7.35
    epochs = mne.Epochs(raw, events, event_id=stim_events, tmin=-0.005, tmax=TRIAL_DURATION, picks=['eeg'], baseline=None) # each trial is about 7.35 s from onset stimulus 

    return epochs.get_data()

def load_patient_trial(subject=1):
    subj_session1 = Path(f'subject_{subject}_fvep_led_training_1.mat')
    subj_session2 = Path(f'subject_{subject}_fvep_led_training_2.mat')

    session_data_1 = get_trials(subj_session1)
    session_data_2 = get_trials(subj_session2)
    labels = [0,1,2,3] * 5
    
    return session_data_1, labels, session_data_2, labels