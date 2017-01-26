import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as pyplot
import utils.utils

#WINDOW_SIZE = 2048 # размер окна, в котором делается fft
#WINDOW_STEP = 512 # шаг окна
WINDOW_SIZE = 5000 # размер окна, в котором делается fft
WINDOW_STEP = 2000 # шаг окна


def plot(filename,points={}):
    utils.utils.mp3towav(filename)
    sample_rate, wave_data = scipy.io.wavfile.read('audio.wav')
    fig = pyplot.figure()
    # ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    # ax.specgram(wave_data,
    #             NFFT=WINDOW_SIZE, noverlap=WINDOW_SIZE - WINDOW_STEP, Fs=sample_rate)

    #sampFreq, snd = wavfile.read('data/02r_cut_cut.wav')
    # convert in -1;1
    wave_data= wave_data / (2. ** 15)
    # choose one chanel

    timeArray = np.arange(0, wave_data.shape[0], 1)
    timeArray = (timeArray / sample_rate) * 1000
    pyplot.plot(timeArray,wave_data)
    #pyplot.plot(wave_data)
    if(len(points)!=0):
        for p in points.keys():
            pyplot.plot([p,p],[-5*(1/points[p]),5*(1/points[p])],color='green',linewidth=10)
    pyplot.show()


#if(__name__=="__main__"):
  #  plot("audio.wav")