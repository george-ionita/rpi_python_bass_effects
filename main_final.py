
"""
Raspberry Pi Audio Effects Unit for Bass Guitar
Author: George-Andrei Ionita

"""
# ------------- MODULES ---------------------------
# -----------------------------------------------------
import pianohat
import pyaudio
import math
import time
import numpy as np
from scipy import fft, arange
from array import *
from random import randint
import wavio
from scipy.io.wavfile import write
import datetime

# ------------- VARIABLES ---------------------------
# -----------------------------------------------------
# Initialise buffer variables

CHANNELS = 1
RATE = 44100
CHUNK = 1024
p = pyaudio.PyAudio()

#Create file to export latency analysis data to

f1 = open('./logfile', 'w+')

# This is 3 and not 4 because effects count starts at 0

max_effects = 3

# Default values for audio effects

effects = np.array([{'effect_on': False, 'effect_amount': 0},
                   {'effect_on': False, 'effect_amount': 7},
                   {'effect_on': False, 'effect_amount': 2},
                   {'effect_on': False, 'effect_amount': 2}])

max_amplitude_input = 0
max_amplitude_output = 0
effect_on_counter = 1
effect_current = 0
is_first_sample = True

# Initialise hanning window

hanning = np.hanning(CHUNK)

# Initialise the Impulse Response and read the IR

impulse_1 = 'Ampeg V4-B_custom beta52.wav'
impulse_2 = 'Ampeg V4-B_TE_sm7.wav'
impulse_3 = 'GK09 -Bass Orange5_K.wav'

ir = wavio.read(impulse_1)
ir = ir.data

# Add zero padding to the Impulse Response(IR) such that the IR and the Input Signal have same length

zeroes = np.zeros((CHUNK - len(ir), ), dtype=np.float32)
ir = np.append(ir, zeroes)

# Do FFT for IR

ir_fft = np.fft.fft(ir)


# ------------- FUNCTIONS ---------------------------
# -------------------------------------------------------
"""
 * This is the main function that needs to be called to run the program.
 * It initialises the main variables and calls the PyAudio stream.	
"""
def main():

    # Initialise Variables

    global max_amplitude_output
    global max_amplitude_input

    global current_window
    global previous_window

    global is_first_sample

    pianohat.auto_leds(True)
    current_window = []
    previous_window = []

    # Initialise PyAudio stream and pass 'callback' as callback function

    stream = p.open(
        format=pyaudio.paFloat32,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        input=True,
        output_device_index=0,
        frames_per_buffer=CHUNK,
        stream_callback=callback,
        )

    # Keep the audio card from overloading by restarting the stream continously

    while True:
        stream.start_stream()
        time.sleep(20)
        stream.stop_stream()

    stream.close()
    p.terminate()

"""
 * This function gets called on every chunk of samples of size CHUNK.
 * It decodes the data and applies processing to the signal.
 * @param {in_data} a chunk of samples of size CHUNK.
 * @param {frame_count} number of frames
 * @param {time_info} dictionary with the following keys: input_buffer_adc_time, current_time, and output_buffer_dac_time; 
 * @param {flag} PortAudio flag.
"""
def callback(
    in_data,
    frame_count,
    time_info,
    flag,
    ):

    # Print timestamp to file for latency analysis

    f1.write('Start callback: %s' % datetime.datetime.now())
    f1.write('\n')

    # Decode from byte string to float

    audio_data = np.fromstring(in_data, dtype=np.float32)

    # Call function to apply processing on signal

    audio_data = do_processing(audio_data)

    # Encode processed signal back to byte string

    out_data = audio_data.astype(np.float32).tostring()

    # Print timestamp to file

    f1.write('End callback: %s' % datetime.datetime.now())
    f1.write('\n')

    return (out_data, pyaudio.paContinue)

"""
 * This function responds to the GPIO event "on_note"
 * It changes the effect amount to the selected key on GPIO controller
 * @param {ch} represents the note value pressed on the Pimoroni PianoHat (0-15).
 * @param {evt} represents a pressed key(true) or depressed key (false)
 """
def controller(ch, evt):

    global effects
    if evt is True:

        # Change the effect's amount

        effects[effect_current]['effect_amount'] = ch
        print 'Amount:', effects[effect_current]['effect_amount']


"""
 * This function responds to the GPIO event "instrument"
 * It turns the current effect on or off when the "Instrument" button is clicked on the GPIO controller
 * @param {ch} represents the note value pressed on the Pimoroni PianoHat (0-15).
 * @param {evt} represents a pressed key(true) or depressed key (false)
"""
def effect_on(ch, evt):

    global effect_on_counter

    effect_on_counter += 1

    if effect_on_counter == 2 and effects[effect_current]['effect_on'] == True:
        
        effects[effect_current]['effect_on'] = False
        effect_on_counter = 0

    if effect_on_counter == 2 and effects[effect_current]['effect_on'] == False:
        
        effects[effect_current]['effect_on'] = True
        effect_on_counter = 0

    print 'Current_effect:', effect_current
    print 'Effect on: ', effects[effect_current]['effect_on']

"""
 * This function responds to the GPIO event "octave_up"
 * Moves the current effect counter up
 * @param {ch} represents the note value pressed on the Pimoroni PianoHat (0-15).
 * @param {evt} represents a pressed key(true) or depressed key (false)
"""
def switch_up(ch, evt):

    global effect_current

    # Check if last effect in chain

    if evt is True and effect_current < max_effects:

        effect_current += 1
        print 'Current_effect:', effect_current
        print 'Effect on: ', effects[effect_current]['effect_on']


"""
 * This function responds to the GPIO event "octave_down"
 * Moves the current effect counter down
 * @param {ch} represents the note value pressed on the Pimoroni PianoHat (0-15).
 * @param {evt} represents a pressed key(true) or depressed key (false)
"""
def switch_down(ch, evt):

    # Check if first effect in chain

    global effect_current

    if evt is True and effect_current > 0:

        effect_current -= 1
        print 'Current_effect: ', effect_current
        print 'Effect on: ', effects[effect_current]['effect_on']


"""
 * This function handles all the audio processing happeninng with the signal
 * @param {input_signal} represents the represents the incoming signal of size CHUNK to be processed
"""
def do_processing(input_signal):

    global max_amplitude_input
    global is_first_sample
    global max_amplitude_output

    # Listen to controller events

    note = pianohat.on_note(controller)
    oct_up = pianohat.on_octave_up(switch_up)
    oct_down = pianohat.on_octave_down(switch_down)
    oct_inst = pianohat.on_instrument(effect_on)

    # Convert to native array using the 'array' module for faster processing

    input_signal = array('f', input_signal)

    # Apply audio effects that have been selected by the user

    if effects[0]['effect_on'] is True:
        input_signal = amp_convolution(input_signal, hanning)
        f1.write('Amp Convolution : on')
        f1.write('\n')

    # Add little amounts to the effect amount to prevent 0 division

    if effects[1]['effect_on'] is True:
        amount = effects[1]['effect_amount']
        input_signal = mod_effect(input_signal, amount + 0.01)
        f1.write('Modulation : on')
        f1.write('\n')

    if effects[2]['effect_on'] is True:
        amount = effects[2]['effect_amount']
        input_signal = hard_clipping(input_signal, float(amount) / 100 + 0.005)
        f1.write('Distortion : on')
        f1.write('\n')

    if effects[3]['effect_on'] is True:
        amount = effects[3]['effect_amount']
        input_signal = chorus_effect(input_signal, 10, amount + 0.01)
        f1.write('Chorus : on')
        f1.write('\n')

    # Convert back to numpy array

    input_signal = np.array(input_signal)

    # Normalise output using a global maximum

    local_max = np.max(np.abs(input_signal))
    if local_max > max_amplitude_output:
        max_amplitude_output = local_max
    input_signal = np.float32(input_signal / float(local_max))

    return input_signal

"""
 * This function carries out the Amp Simulation audio effect
 * @param {input_signal} represents the represents the incoming signal of size CHUNK to be processed
 * @param {window} represents the hanning window passed in
"""
def amp_convolution(input_signal, window):

    global current_window
    global previous_window

    # apply windowing

    # if it's the first 1024 samples, initialise previois window with first half of the next hanning window
    # and current window with second half

    if len(previous_window) == 0:

        previous_window = window[:512]
        current_window = window[512:]

        # apply window 1

        windowed_signal = input_signal * window

        # apply current window to second half of original signal

        overlap_signal = input_signal[512:] * previous_window

        # add zeroes so this can be multiplied to the signal

        zeroes = np.zeros(512, dtype=np.float32)
        overlap_signal = np.append(zeroes, overlap_signal)

        # overlap add with second half of signal

        windowed_signal *= overlap_signal
    else:

        # old current window is now previous

        previous_window = current_window
        current_window = window[:512]

        # apply window 1

        windowed_signal = input_signal * window

        # apply second half of previous window to first half of signal

        overlap_signal_previous = previous_window * input_signal[:512]
        windowed_signal[:512] *= overlap_signal_previous

        # now apply first half of current window to second half of signal

        overlap_signal_current = current_window * input_signal[512:]
        windowed_signal[512:] *= overlap_signal_current

        # prepare next window

        current_window = window[512:]

    # do FFT for IS

    input_signal_fft = np.fft.fft(windowed_signal)

    # convolve the two signals

    convoluted_data = ir_fft * input_signal_fft

    # get back to the time domain using IFFT

    convoluted_data_ifft = np.fft.ifft(convoluted_data)
    convoluted_data_ifft = convoluted_data_ifft.real

    #convert from complex
    convoluted_data_ifft = np.float32(convoluted_data_ifft)

    return convoluted_data_ifft

"""
 * This function represents a chorus audio effect
 * @param {input_signal} represents the represents the incoming signal of size CHUNK to be processed
 * @param {delay_ms} represents the delay parameter of the effect
 * @param {rate} represents the represents the rate parameter of the effect
"""
def chorus_effect(input_signal, delay_ms, rate):

    global delayed_signal_first_section
    global delayed_signal_second_section
    global is_first_sample

    # 1 ms = 44.1 samples

    delay_samples = delay_ms * 44.1

    if is_first_sample is True:

        # apply sinusoid on signal

        Fs = 440
        f = rate
        sample = 1024
        x = np.arange(sample)
        y = np.sin(2 * np.pi * f * x / Fs)

        input_signal_copy = input_signal * y

        # split delayed signal into first section ( to be multiplied with current frame) and next section (to be multiplied with the next frame)

        delayed_signal_first_section = input_signal_copy[:1024
            - delay_samples]

        zeroes = np.zeros(delay_samples, dtype=np.float32)
        delayed_signal_first_section = np.append(zeroes,
                delayed_signal_first_section)

        delayed_signal_second_section = input_signal_copy[1024
            - delay_samples:]

        # add modulated copied signal to original signal

        input_signal = input_signal * delayed_signal_first_section

        is_first_sample = False
    else:

        # apply sinusoid on signal

        Fs = 440
        f = rate
        sample = 1024
        x = np.arange(sample)
        y = np.sin(2 * np.pi * f * x / Fs)

        input_signal_copy = input_signal * y

        # obtain first section of delayed signal

        delayed_signal_first_section = input_signal_copy[:1024
            - delay_samples]

        # form signal to be multiplied to the input from tail of previous delayed signal
        # and first section of next delayed signal

        new_signal = np.append(delayed_signal_second_section,
                               delayed_signal_first_section)

        # update delayed signal second section to contain current delayed signal tail
        # to be used in next window

        delayed_signal_second_section = input_signal_copy[1024
            - delay_samples:]

        # finally, multiply copied signal to original signal

        input_signal = input_signal * new_signal

    return input_signal

"""
 * This function represents a modulation audio effect
 * @param {input_signal} represents the represents the incoming signal of size CHUNK to be processed
 * @param {delay_ms} represents the window applied to signal
 * @param {sine_rate} represents the represents the rate of the sine wave 
  * @param {frequency} represents the frequency of the sine wave 
"""   
def mod_effect(input_signal, sine_rate, frequency=440):

    rand = randint(0, 2000)
    Fs = frequency + rand
    f = sine_rate
    sample = 1024
    x = np.arange(sample)
    y = np.sin(2 * np.pi * f * x / Fs)

    return input_signal * y  

"""
 * This function represents a hard clipping audio effect
 * @param {input_signal} represents the represents the incoming signal of size CHUNK to be processed
 * @param {th} represents the threshold at which point clipping occurs 
""" 
def hard_clipping(input_signal, th):

    for i in range(1024):
        if input_signal[i] < -th:
            input_signal[i] = -th
        if input_signal[i] > th:
            input_signal[i] = th
    return input_signal

#Call the main function
main()
