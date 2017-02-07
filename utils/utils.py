#import sox
import scipy.io.wavfile as wavfile
import librosa
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import contextlib
import wave

#Awful code, all of it
def mp3towav(filename,outputname):
    """
    Converts mp3 to wav
    :param filename: input file name (mp3)
    :param outputname: output file name (wav)
    :return: nothing
    """
    #tfm = sox.Transformer()
    #tfm.remix(num_output_channels=1)
    #tfm.build(filename, outputname)

    subprocess.call(["sox",filename,outputname,"remix","1"])


def get_duration(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

def save_to_file(filename,content):
    f = open(filename, "w")
    for line in content:
        f.write(line)
        f.write("\r\n")
    f.close()

def get_beginning_of_file(old_filename,new_filename,duration):
    rate,wave_data=wavfile.read(old_filename)
    wavfile.write(new_filename,rate,wave_data[:duration*rate])

def change_bitrate(old_filename,new_filename,new_bitrate):
    #sox.core.sox([old_filename, "-b", "16", new_filename, "rate", str(new_bitrate)])
    subprocess.call(["sox",old_filename, "-b", "16", new_filename, "rate", str(new_bitrate)])

def load_with_changed_bitrate(filename,changed):
    wave_data,rate=librosa.load(filename,sr=changed)
    #byte_data=np.frombuffer(wave_data.astype(np.float16),dtype=np.byte)

    # timeArray = np.arange(0, wave_data.shape[0], 1)
    # timeArray = (timeArray / rate) * 1000
    # plt.plot(timeArray, wave_data)
    # timeArray = np.arange(0, byte_data.shape[0], 1)
    # timeArray = (timeArray / rate) * 1000
    # plt.plot(timeArray, byte_data)
    # plt.show()
    #wave_data=np.array(wave_data)
    #wavfile.write("test_changed_bitrate.wav",rate,wave_data)
    librosa.output.write_wav("test_changed_bitrate.wav",wave_data,sr=changed)
    return rate,wave_data

if(__name__=="__main__"):
    load_with_changed_bitrate("../beginning_out_test.wav",32000)
