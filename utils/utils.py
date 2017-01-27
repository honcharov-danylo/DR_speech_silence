import sox
import scipy.io.wavfile as wavfile


#Awful code, all of it
def mp3towav(filename,outputname):
    """
    Converts mp3 to wav
    :param filename: input file name (mp3)
    :param outputname: output file name (wav)
    :return: nothing
    """
    tfm = sox.Transformer()
    tfm.remix(num_output_channels=1)
    tfm.build(filename, outputname)


def save_to_file(filename,content):
    f = open(filename, "w")
    for line in content:
        f.write(line)
        f.write("\r\n")
    f.close()

def get_beginning_of_file(old_filename,new_filename,duration):
    rate,wave_data=wavfile.read(old_filename)
    wavfile.write(new_filename,rate,wave_data[:duration*rate])


