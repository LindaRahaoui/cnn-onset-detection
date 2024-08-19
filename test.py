"""Main module."""
from librosa import load, get_samplerate, pitch_tuning, hz_to_midi, time_to_samples, onset, stft
from scipy.signal import find_peaks, hilbert, peak_widths, butter, filtfilt, resample
import numpy as np
import pretty_midi as pm
import matplotlib.pyplot as plt
import crepe
from scipy.io import wavfile

import os.path
from pathlib import Path
def Create_Note_list():
    frequency_note = {
        8.18: 'C-1', 8.661: 'C#-1',8.66:'Db-1', 9.18: 'D-1', 9.721: 'D#-1', 9.72:'Eb-1', 10.3: 'E-1', 10.91: 'F-1', 11.561: 'F#-1', 11.56:'Gb-1',
        12.25: 'G-1', 12.981: 'G#-1', 12.98: 'Ab-1', 13.75: 'A-1', 14.571: 'A#-1', 14.57: 'Bb-1', 15.43: 'B-1', 16.35: 'C0', 17.321: 'C#0', 17.32:'Db0',
        18.35: 'D0', 19.451: 'D#0', 19.45: 'Eb0', 20.6: 'E0', 21.83: 'F0', 23.121: 'F#0', 23.12:'Gb0', 24.5: 'G0', 25.961: 'G#0', 25.96:'Ab0',
        27.5: 'A0', 29.141: 'A#0', 29.14: 'Bb0', 30.87: 'B0', 32.7: 'C1', 34.651: 'C#1', 34.65: 'Db1', 36.71: 'D1', 38.891: 'D#1', 38.89:'Eb1',
        41.2: 'E1', 43.65: 'F1', 46.251: 'F#1', 46.25:'Gb1', 49.0: 'G1', 51.911: 'G#1', 51.91:'Ab1', 55.0: 'A1', 58.271: 'A#1', 58.27: 'Bb1',
        61.74: 'B1', 65.41: 'C2', 69.31: 'C#2', 69.3: 'Db2', 73.42: 'D2', 77.781: 'D#2', 77.78: 'Eb2', 82.41: 'E2', 87.31: 'F2',
        92.51: 'F#2',92.5: 'Gb2', 98.0: 'G2', 103.831: 'G#2', 103.83: 'Ab2', 110.0: 'A2', 116.541: 'A#2', 116.54: 'Bb2', 123.47: 'B2', 130.81: 'C3',
        138.591: 'C#3',138.59:'Db3', 146.83: 'D3', 155.561: 'D#3', 155.56:'Eb3', 164.81: 'E3', 174.61: 'F3', 185.01: 'F#3', 185.0: 'Gb3', 196.0: 'G3',
        207.651: 'G#3',207.65:'Ab3', 220.0: 'A3', 233.081: 'A#3', 233.08:'Bb3', 246.94: 'B3', 261.63: 'C4', 277.181: 'C#4', 277.18:'Db4', 293.66: 'D4',
        311.131: 'D#4',311.13: 'Eb4', 329.63: 'E4', 349.23: 'F4', 369.991: 'F#4', 369.99:'Gb4', 392.0: 'G4', 415.31: 'G#4', 415.3: 'Ab4', 440.0: 'A4',
        466.161: 'A#4',466.16:'Bb4', 493.88: 'B4', 523.25: 'C5', 554.371: 'C#5', 554.37:'Db5', 587.33: 'D5', 622.251: 'D#5', 622.25:'Eb5', 659.25: 'E5',
        698.46: 'F5', 739.991: 'F#5', 739.99:'Gb5', 783.99: 'G5', 830.611: 'G#5', 830.61:'Ab5', 880.0: 'A5', 932.331: 'A#5', 932.33: 'Bb5', 987.77: 'B5',
        1046.5: 'C6', 1108.731: 'C#6', 1108.73: 'Db6', 1174.66: 'D6', 1244.511: 'D#6', 1244.51:'Eb6', 1318.51: 'E6', 1396.91: 'F6', 1479.981: 'F#6', 1479.98: 'Gb6',
        1567.98: 'G6', 1661.221: 'G#6', 1661.22: 'Ab6', 1760.0: 'A6', 1864.661: 'A#6', 1864.66:'Bb6', 1975.53: 'B6', 2093.0: 'C7'
    }
    liste_notes = ['C-1', 'C#-1', 'Db-1', 'D-1', 'D#-1', 'Eb-1', 'E-1', 'F-1', 'F#-1', 'Gb-1', 'G-1', 'G#-1', 'Ab-1', 'A-1', 'A#-1', 'Bb-1', 'B-1', 'C0', 'C#0', 'Db0', 'D0', 'D#0', 'Eb0', 'E0', 'F0', 'F#0', 'Gb0', 'G0', 'G#0', 'Ab0', 'A0', 'A#0', 'Bb0', 'B0', 'C1', 'C#1', 'Db1', 'D1', 'D#1', 'Eb1', 'E1', 'F1', 'F#1', 'Gb1', 'G1', 'G#1', 'Ab1', 'A1', 'A#1', 'Bb1', 'B1', 'C2', 'C#2', 'Db2', 'D2', 'D#2', 'Eb2', 'E2', 'F2', 'F#2', 'Gb2', 'G2', 'G#2', 'Ab2', 'A2', 'A#2', 'Bb2', 'B2', 'C3', 'C#3', 'Db3', 'D3', 'D#3', 'Eb3', 'E3', 'F3', 'F#3', 'Gb3', 'G3', 'G#3', 'Ab3', 'A3', 'A#3', 'Bb3', 'B3', 'C4', 'C#4', 'Db4', 'D4', 'D#4', 'Eb4', 'E4', 'F4', 'F#4', 'Gb4', 'G4', 'G#4', 'Ab4', 'A4', 'A#4', 'Bb4', 'B4', 'C5', 'C#5', 'Db5', 'D5', 'D#5', 'Eb5', 'E5', 'F5', 'F#5', 'Gb5', 'G5', 'G#5', 'Ab5', 'A5', 'A#5', 'Bb5', 'B5', 'C6', 'C#6', 'Db6', 'D6', 'D#6', 'Eb6', 'E6', 'F6', 'F#6', 'Gb6', 'G6', 'G#6', 'Ab6', 'A6', 'A#6', 'Bb6', 'B6', 'C7']
    return liste_notes ,  frequency_note

def get_note_guessed_from_fname(note_list: list, fname: str):
    """Extract MIDI note name based on wav filename. 
    Note name should be in the filename

    Args:
        note_list (list[str]): List of possible notes as strings
        fname (str): name of wav file

    Returns:
        A tuple containing :
            str or None : the note name
            int or None : the note number in pretty_midi format or None if note name was not in file name
    """
    midi_note = None
    note_name = None
    for note_candidate in note_list:
        if note_candidate in fname:
            note_name = note_candidate
            midi_note = pm.note_name_to_number(note_candidate)
            break
    return note_name, midi_note

def run_crepe(audio_path):
    sr, audio = wavfile.read(str(audio_path))
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
    
    return frequency, confidence


def steps_to_samples(step_val, sr, step_size=0.01):
    return int(step_val * (sr * step_size))


def samples_to_steps(sample_val, sr, step_size=0.01):
    return int(sample_val / (sr * step_size))


def freqs_to_midi(freqs, tuning_offset=0):
    return np.nan_to_num(hz_to_midi(freqs) - tuning_offset, neginf=0)


def calculate_tuning_offset(freqs):
    tuning_offset = pitch_tuning(freqs)
    print(f"Tuning offset: {tuning_offset * 100} cents")
    return tuning_offset


def parse_f0(f0_path):
    data = np.genfromtxt(f0_path, delimiter=',', names=True)
    return np.nan_to_num(data['frequency']), np.nan_to_num(data['confidence'])
    
def save_f0(f0_path, frequency, confidence):
    # create the save directory if it doesn't exist
    np.savetxt(f0_path, np.stack([np.linspace(0, 0.01 * len(frequency), len(frequency)).astype('float'), frequency.astype('float'), confidence.astype('float')], axis=1), fmt='%10.7f', delimiter=',', header='time,frequency,confidence', comments='')
    return

def load_audio(audio_path, cached_amp_envelope_path, default_sample_rate, detect_amplitude, save_amp_envelope):
    if cached_amp_envelope_path.exists():
        # if we have a cached amplitude envelope, no need to load audio
        filtered_amp_envelope = np.load(cached_amp_envelope_path, allow_pickle=True)['filtered_amp_envelope']
        # sr = get_samplerate(audio_path)
        sr = default_sample_rate # this is mainly to make tests work without having to load audio
        y = None
    else:
        try:
            y, sr = load(str(audio_path), sr=None)
        except:
            print("Error loading audio file. Amplitudes will be set to 80")
            detect_amplitude = False
            y = None
            pass

        amp_envelope = np.abs(hilbert(y))

        scaled_amp_envelope = np.interp(amp_envelope, (amp_envelope.min(), amp_envelope.max()), (0, 1))
        # low pass filter the amplitude envelope
        b, a = butter(4, 50, 'low', fs=sr)
        filtered_amp_envelope = filtfilt(b, a, scaled_amp_envelope)[::(sr//100)]
    
    if save_amp_envelope:
        np.savez(cached_amp_envelope_path, filtered_amp_envelope=filtered_amp_envelope)
    
    return sr, y, filtered_amp_envelope, detect_amplitude    


def process(freqs,
            conf,
            audio_path,
            output_label="transcription",
            sensitivity=0.001,
            use_smoothing=False,
            min_duration=0.03,
            min_velocity=6,
            disable_splitting=False,
            use_cwd=True,
            tuning_offset=False,
            detect_amplitude=True,
            save_amp_envelope=False,
            default_sample_rate=44100,
            save_analysis_files=False,):
    #
    print (audio_path)
   
    fname = audio_path.stem
    note_list,_ = Create_Note_list()
    cached_amp_envelope_path = audio_path.with_suffix(".amp_envelope.npz")
    sr, y, filtered_amp_envelope, detect_amplitude = load_audio(audio_path, cached_amp_envelope_path, default_sample_rate, detect_amplitude, (save_analysis_files or save_amp_envelope))
    _, midi_note = get_note_guessed_from_fname(note_list=note_list, fname=fname)
   

    if use_cwd:
        # write to location that the bin was run from
        output_filename = audio_path.stem
    else:
        # write to same folder as the orignal audio file
        output_filename = str(audio_path.parent) + "/" + audio_path.stem

    print(os.path.abspath(audio_path))
    
    if save_analysis_files:
        f0_path = audio_path.with_suffix(".f0.csv")
        if not f0_path.exists():
            print(f"Saving f0 to {f0_path}")
            save_f0(f0_path, freqs, conf)  

    if not disable_splitting:
        onsets_path = str(audio_path.with_suffix('.onsets.npz'))
        if not os.path.exists(onsets_path):
            print(f"Onsets file not found at {onsets_path}")
            print("Running onset detection...")
            
            from madmom.features import CNNOnsetProcessor
            
            onset_activations = CNNOnsetProcessor()(str(audio_path))
            if save_analysis_files:
                np.savez(onsets_path, activations=onset_activations)
        else:
            print(f"Loading onsets from {onsets_path}")
            onset_activations = np.load(onsets_path, allow_pickle=True)['activations']

        onsets = np.zeros_like(onset_activations)
        onsets[find_peaks(onset_activations, distance=4, height=0.8)[0]] = 1

    if tuning_offset == False:
        tuning_offset = calculate_tuning_offset(freqs)
    else:
        tuning_offset = tuning_offset / 100

    # get pitch gradient
    midi_pitch = freqs_to_midi(freqs, tuning_offset)
    pitch_changes = np.abs(np.gradient(midi_pitch))
    pitch_changes = np.interp(pitch_changes,
                              (pitch_changes.min(), pitch_changes.max()),
                              (0, 1))

    # get confidence peaks with peak widths (prominences)
    conf_peaks, conf_peak_properties = find_peaks(1 - conf,
                                                  distance=4,
                                                  prominence=sensitivity)

    # combine pitch changes and confidence peaks to get change point signal
    change_point_signal = (1 - conf) * pitch_changes
    change_point_signal = np.interp(
        change_point_signal,
        (change_point_signal.min(), change_point_signal.max()), (0, 1))
    peaks, peak_properties = find_peaks(change_point_signal,
                                        distance=4,
                                        prominence=sensitivity)
    _, _, transition_starts, transition_ends = peak_widths(change_point_signal, peaks, rel_height=0.5)
    transition_starts = list(map(int, np.round(transition_starts)))
    transition_ends = list(map(int, np.round(transition_ends)))

    # get candidate note regions - any point between two peaks in the change point signal
    transitions = [(s, f, 'transition') for (s, f) in zip(transition_starts, transition_ends)]
    note_starts = [0] + transition_ends
    note_ends = transition_starts + [len(change_point_signal) + 1]
    note_regions = [(s, f, 'note') for (s, f) in (zip(note_starts, note_ends))]

    if detect_amplitude:
        # take the amplitudes within 6 sigma of the mean
        # helps to clean up outliers in amplitude scaling as we are not looking for 100% accuracy
        amp_mean = np.mean(filtered_amp_envelope)
        amp_sd = np.std(filtered_amp_envelope)
        # filtered_amp_envelope = amp_envelope.copy()
        filtered_amp_envelope[filtered_amp_envelope > amp_mean + (6 * amp_sd)] = 0
        global_max_amp = max(filtered_amp_envelope)

    segment_list = []
    for a, b, label in sum(zip(note_regions, transitions), ()):
        if label == 'transition':
            continue

        # Handle an edge case where rounding could cause
        # an end index for a note to be before the start index
        if a > b:
            continue
        elif b - a <= 1:
            continue

        if detect_amplitude:
            max_amp = np.max(filtered_amp_envelope[a:b])
            scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))
        else:
            scaled_max_amp = 80
        print(np.round(np.median(midi_pitch[a:b])))
        segment_list.append({
            'pitch': np.round(np.median(midi_pitch[a:b])),
            'conf': np.median(conf[a:b]),
            'transition_strength': 1 - conf[a], # TODO: make use of the dip in confidence as a measure of how strong an onset is
            'amplitude': scaled_max_amp,
            'start_idx': a,
            'finish_idx': b,
        })

    # segment list contains our candidate notes
    # now we iterate through them and merge if two adjacent segments have the same median pitch
    notes = []
    sub_list = []
    for a, b in zip(segment_list, segment_list[1:]):
        # TODO: make use of variance in segment to catch glissandi?
        # if np.var(midi_pitch[a[1][0]:a[1][1]]) > 1:
        #     continue

        if np.abs(a['pitch'] -
                  b['pitch']) > 0.5:  # or a['transition_strength'] > 0.4:
            sub_list.append(a)
            notes.append(sub_list)
            sub_list = []
        else:
            sub_list.append(a)

    # catch any segments at the end
    if len(sub_list) > 0:
        notes.append(sub_list)


    # garder les notes si le pitch coorespond à la note

    output_midi = pm.PrettyMIDI()
    instrument = pm.Instrument(
        program=pm.instrument_name_to_program('Acoustic Grand Piano'))

    velocities = []
    durations = []
    output_notes = []

    # Filter out notes that are too short or too quiet
    for x_s in notes:
        x_s_filt = [x for x in x_s if x['amplitude'] > min_velocity]
        if len(x_s_filt) == 0:
            continue
        median_pitch = np.median(np.array([y['pitch'] for y in x_s_filt]))
        median_confidence = np.median(np.array([y['conf'] for y in x_s_filt]))
        seg_start = x_s_filt[0]['start_idx']
        seg_end = x_s_filt[-1]['finish_idx']
        time_start = 0.01 * seg_start
        time_end = 0.01 * seg_end
        sample_start = time_to_samples(time_start, sr=sr)
        sample_end = time_to_samples(time_end, sr=sr)
        max_amp = np.max(filtered_amp_envelope[seg_start:seg_end])
        scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))

        valid_amplitude = scaled_max_amp > min_velocity
        valid_duration = (time_end - time_start) > min_duration
        
        # TODO: make use of confidence strength
        valid_confidence = True  # median_confidence > 0.1

        if valid_amplitude and valid_confidence and valid_duration:
            output_notes.append({
                'pitch':
                    int(np.round(median_pitch)),
                'velocity':
                    round(scaled_max_amp),
                'start_idx':
                    seg_start,
                'finish_idx':
                    seg_end,
                'conf':
                    median_confidence,
                'transition_strength':
                    x_s[-1]['transition_strength']
            })

    # Handle repeated notes
    # Here we use a standard onset detection algorithm from madmom
    # with a high threshold (0.8) to re-split notes that are repeated
    # Repeated notes have a pitch gradient of 0 and are therefore
    # not separated by the algorithm above
    if not disable_splitting:
        onset_separated_notes = []
        for n in output_notes:
            n_s = n['start_idx']
            n_f = n['finish_idx']

            last_onset = 0
            if np.any(onsets[n_s:n_f] > 0.7):
                onset_idxs_within_note = np.argwhere(onsets[n_s:n_f] > 0.7)
                for idx in onset_idxs_within_note:
                    if idx[0] > last_onset + int(min_duration / 0.01):
                        new_note = n.copy()
                        new_note['start_idx'] = n_s + last_onset
                        new_note['finish_idx'] = n_s + idx[0]
                        onset_separated_notes.append(new_note)
                        last_onset = idx[0]

            # If there are no valid onsets within the range
            # the following should append a copy of the original note,
            # but if there were splits at onsets then it will also clean up any tails
            # left in the sequence
            new_note = n.copy()
            new_note['start_idx'] = n_s + last_onset
            new_note['finish_idx'] = n_f
            onset_separated_notes.append(new_note)
            output_notes = onset_separated_notes

    if detect_amplitude:
        # Trim notes that fall below a certain amplitude threshold
        timed_output_notes = []
        for n in output_notes:
            timed_note = n.copy()

            # Adjusting the start time to meet a minimum amp threshold
            s = timed_note['start_idx']
            f = timed_note['finish_idx']

            if f - s > (min_duration / 0.01):
                # TODO: make noise floor configurable
                noise_floor = 0.01  # this will vary depending on the signal
                s_samp = steps_to_samples(s, sr)
                f_samp = steps_to_samples(f, sr)
                s_adj_samp_all = s_samp + np.where(
                    filtered_amp_envelope[s:f] > noise_floor)[0]

                if len(s_adj_samp_all) > 0:
                    s_adj_samp_idx = s_adj_samp_all[0]
                else:
                    continue

                s_adj = samples_to_steps(s_adj_samp_idx, sr)

                f_adj_samp_idx = f_samp - np.where(
                    np.flip(filtered_amp_envelope[s:f]) > noise_floor)[0][0]
                if f_adj_samp_idx > f_samp or f_adj_samp_idx < 1:
                    print("something has gone wrong")

                f_adj = samples_to_steps(f_adj_samp_idx, sr)
                if f_adj > f or f_adj < 1:
                    print("something has gone more wrong")

                timed_note['start'] = s_adj * 0.01
                timed_note['finish'] = f_adj * 0.01
                timed_output_notes.append(timed_note)
            else:
                timed_note['start'] = s * 0.01
                timed_note['finish'] = f * 0.01

    

    return timed_output_notes, filtered_amp_envelope

"""
def process(freqs,
            conf,
            audio_path,
            output_label="transcription",
            sensitivity=0.001,
            use_smoothing=False,
            min_duration=0.03,
            min_velocity=9,
            disable_splitting=False,
            use_cwd=True,
            tuning_offset=False,
            detect_amplitude=True,
            save_amp_envelope=False,
            default_sample_rate=44100,
            save_analysis_files=False,):
    
    display = False
    # Etape 1 : Chargement de l'audio
   
    fname = audio_path.stem
    note_list,_ = Create_Note_list()
    cached_amp_envelope_path = audio_path.with_suffix(".amp_envelope.npz")
    sr, y, filtered_amp_envelope, detect_amplitude = load_audio(audio_path, cached_amp_envelope_path, default_sample_rate, detect_amplitude, (save_analysis_files or save_amp_envelope))
    _, midi_note = get_note_guessed_from_fname(note_list=note_list, fname=fname)


    if display:
      DisplayAudio(audio_path)

    if use_cwd:
        # write to location that the bin was run from
        output_filename = audio_path.stem
    else:
        # write to same folder as the orignal audio file
        output_filename = str(audio_path.parent) + "/" + audio_path.stem

    print(os.path.abspath(audio_path))
    
    # Étape 2 : Sauvegarde des fichiers d'analyse (optionnel)
    if save_analysis_files:
        f0_path = audio_path.with_suffix(".f0.csv")
        if not f0_path.exists():
            print(f"Saving f0 to {f0_path}")
            save_f0(f0_path, freqs, conf)  

    # Étape 3 : Détection des onsets
    if not disable_splitting:
        onsets = detect_onsets(audio_path)

    # Étape 4 : Calcul du décalage de l'accordage
    if tuning_offset == False:
        tuning_offset = calculate_tuning_offset(freqs)
    else:
        tuning_offset = tuning_offset / 100
    # Nouvelle etape : filtrer les fréquences pour ne gardes que celle autour de la note que l'on veut
   
    
    # Étape 5 : Conversion des fréquences en pitch MIDI
    # get pitch gradient
    midi_pitch = freqs_to_midi(freqs, tuning_offset)
  

    pitch_changes = np.abs(np.gradient(midi_pitch))# Calcul des changements de pitch
    pitch_changes = np.interp(pitch_changes,
                              (pitch_changes.min(), pitch_changes.max()),
                              (0, 1))# Normalisation des changements de pitch
  
   
    # Étape 6 : Détection des pics de confiance
    # get confidence peaks with peak widths (prominences)
    conf_peaks, conf_peak_properties = find_peaks(1 - conf,
                                                  distance=4,
                                                  prominence=sensitivity)
    
    # combine pitch changes and confidence peaks to get change point signal
    #Cela crée un signal qui combine les informations des deux sources, mettant en évidence les points où les deux valeurs sont élevées.
    change_point_signal = (1 - conf) * pitch_changes
    change_point_signal = np.interp(
        change_point_signal,
        (change_point_signal.min(), change_point_signal.max()), (0, 1))
    peaks, peak_properties = find_peaks(change_point_signal,
                                        distance=4,
                                        prominence=sensitivity)
    #peaks : Contient les indices des pics détectés dans change_point_signal.
    #Calcul des largeurs des pics pour déterminer les transitions :
    #_, _, transition_starts, transition_ends = peak_widths(change_point_signal, peaks, rel_height=0.5)
    # transition_starts = list(map(int, np.round(transition_starts)))
    #transition_ends = list(map(int, np.round(transition_ends)))
    print("peaks :",peaks)
    print("conf_peak :",conf_peaks)
   

    _, _, transition_starts_conf, transition_ends_conf = peak_widths(change_point_signal, conf_peaks, rel_height=0.5)
    transition_starts = list(map(int, np.round(transition_starts_conf)))
    transition_ends = list(map(int, np.round(transition_ends_conf)))
   
    # Étape 7 : Détection des régions de notes candidates
    # get candidate note regions - any point between two peaks in the change point signal
    transitions = [(s, f, 'transition') for (s, f) in zip(transition_starts, transition_ends)]
  
    note_starts = [0] + transition_ends
    note_ends = transition_starts + [len(change_point_signal) + 1]
    note_regions = [(s, f, 'note') for (s, f) in (zip(note_starts, note_ends))]




    # Étape 8 : Détection de l'amplitude (optionnel)
    if detect_amplitude:
        # take the amplitudes within 6 sigma of the mean
        # helps to clean up outliers in amplitude scaling as we are not looking for 100% accuracy
        amp_mean = np.mean(filtered_amp_envelope)
        amp_sd = np.std(filtered_amp_envelope)
        # filtered_amp_envelope = amp_envelope.copy()
        filtered_amp_envelope[filtered_amp_envelope > amp_mean + (6 * amp_sd)] = 0
        global_max_amp = max(filtered_amp_envelope)
 

    
    # Étape 9 : Création de la liste des segments
    segment_list = []
    for a, b, label in sum(zip(note_regions, transitions), ()):
        if label == 'transition':
            continue

        # Handle an edge case where rounding could cause
        # an end index for a note to be before the start index
        if a > b:
            continue
        elif b - a <= 1:
            continue

        if detect_amplitude:
            max_amp = np.max(filtered_amp_envelope[a:b])
            scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))
          
        else:
            scaled_max_amp = 80
        
        
           
        segment_list.append({
            'pitch': np.round(np.median(midi_pitch[a:b])),
            'conf': np.median(conf[a:b]),
            'transition_strength': 1 - conf[a], # TODO: make use of the dip in confidence as a measure of how strong an onset is
            'amplitude': scaled_max_amp,
            'start_idx': a,
            'finish_idx': b,
        })

    ## Étape 10 : Filtrage et fusion des segments
    # segment list contains our candidate notes
    # now we iterate through them and merge if two adjacent segments have the same median pitch
    notes = []
    sub_list = []
    for a, b in zip(segment_list, segment_list[1:]):
        # TODO: make use of variance in segment to catch glissandi?
        # if np.var(midi_pitch[a[1][0]:a[1][1]]) > 1:
        #     continue

        if np.abs(a['pitch'] -
                  b['pitch']) > 0.5:  # or a['transition_strength'] > 0.4:
            sub_list.append(a)
            notes.append(sub_list)
            sub_list = []
        else:
            sub_list.append(a)

    # catch any segments at the end
    if len(sub_list) > 0:
        notes.append(sub_list)


    # garder les notes si le pitch coorespond à la note
    # Étape 11 : Création des notes de sortie
    output_midi = pm.PrettyMIDI()
    instrument = pm.Instrument(
        program=pm.instrument_name_to_program('Acoustic Grand Piano'))

    velocities = []
    durations = []
    output_notes = []
    min_diff=8
    # Filter out notes that are too short or too quiet
    for x_s in notes:
        
        #x_s_filt = [x for x in x_s if x['amplitude'] > min_velocity and abs(x['pitch']-midi_note)<min_diff]
        x_s_filt = [x for x in x_s if x['amplitude'] > min_velocity and midi_note-1<=x['pitch']<=midi_note+1]
        #x_s_filt = [x for x in x_s if x['pitch']==midi_note]
        #x_s_filt = [x for x in x_s if abs(x['pitch']-midi_note)<min_diff]    
        
        if len(x_s_filt) == 0:
            continue
        median_pitch = np.median(np.array([y['pitch'] for y in x_s_filt]))
        median_confidence = np.median(np.array([y['conf'] for y in x_s_filt]))
        seg_start = x_s_filt[0]['start_idx']
        seg_end = x_s_filt[-1]['finish_idx']
        time_start = 0.01 * seg_start
        time_end = 0.01 * seg_end
        sample_start = time_to_samples(time_start, sr=sr)
        sample_end = time_to_samples(time_end, sr=sr)
        max_amp = np.max(filtered_amp_envelope[seg_start:seg_end])
        scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))

        valid_amplitude = scaled_max_amp > min_velocity
        valid_duration = (time_end - time_start) > min_duration
        
        # TODO: make use of confidence strength
        valid_confidence = True  # median_confidence > 0.1

        #if valid_amplitude and valid_confidence and valid_duration:
        output_notes.append({
            'pitch':
                int(np.round(median_pitch)),
            'velocity':
                round(scaled_max_amp),
            'start_idx':
                seg_start,
            'finish_idx':
                seg_end,
            'conf':
                median_confidence,
            'transition_strength':
                x_s[-1]['transition_strength']
        })
   
    # Étape 13 : Gestion des notes répétées 
    # Handle repeated notes
    # Here we use a standard onset detection algorithm from madmom
    # with a high threshold (0.8) to re-split notes that are repeated
    # Repeated notes have a pitch gradient of 0 and are therefore
    # not separated by the algorithm above
    if not disable_splitting:
        onset_separated_notes = []
        for n in output_notes:
            n_s = n['start_idx']
            n_f = n['finish_idx']

            last_onset = 0
            if np.any(onsets[n_s:n_f] > 0.7):
                onset_idxs_within_note = np.argwhere(onsets[n_s:n_f] > 0.7)
                for idx in onset_idxs_within_note:
                    if idx[0] > last_onset + int(min_duration / 0.01):
                        new_note = n.copy()
                        new_note['start_idx'] = n_s + last_onset
                        new_note['finish_idx'] = n_s + idx[0]
                        onset_separated_notes.append(new_note)
                        last_onset = idx[0]

            # If there are no valid onsets within the range
            # the following should append a copy of the original note,
            # but if there were splits at onsets then it will also clean up any tails
            # left in the sequence
            new_note = n.copy()
            new_note['start_idx'] = n_s + last_onset
            new_note['finish_idx'] = n_f
            onset_separated_notes.append(new_note)
            output_notes = onset_separated_notes
    
    if detect_amplitude:
        # Trim notes that fall below a certain amplitude threshold
        timed_output_notes = []
        for n in output_notes:
            timed_note = n.copy()

            # Adjusting the start time to meet a minimum amp threshold
            s = timed_note['start_idx']
            f = timed_note['finish_idx']

            if f - s > (min_duration / 0.01):
                # TODO: make noise floor configurable
                noise_floor = 0.01  # this will vary depending on the signal
                s_samp = steps_to_samples(s, sr)
                f_samp = steps_to_samples(f, sr)
                s_adj_samp_all = s_samp + np.where(
                    filtered_amp_envelope[s:f] > noise_floor)[0]

                if len(s_adj_samp_all) > 0:
                    s_adj_samp_idx = s_adj_samp_all[0]
                else:
                    continue

                s_adj = samples_to_steps(s_adj_samp_idx, sr)

                f_adj_samp_idx = f_samp - np.where(
                    np.flip(filtered_amp_envelope[s:f]) > noise_floor)[0][0]
                if f_adj_samp_idx > f_samp or f_adj_samp_idx < 1:
                    print("something has gone wrong")

                f_adj = samples_to_steps(f_adj_samp_idx, sr)
                if f_adj > f or f_adj < 1:
                    print("something has gone more wrong")

                timed_note['start'] = s_adj * 0.01
                timed_note['finish'] = f_adj * 0.01
                timed_output_notes.append(timed_note)
            else:
                timed_note['start'] = s * 0.01
                timed_note['finish'] = f * 0.01
  
    return timed_output_notes, filtered_amp_envelope
    """