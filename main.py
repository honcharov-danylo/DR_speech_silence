import sox_splitter
import utils.analysis
import os
import vad_splitter

def main(path_to_file):
    utils.utils.mp3towav(path_to_file,
                         "audio_temp.wav")
    path_to_file="audio_temp.wav"
    ss=sox_splitter.sox_splitter()
    text = ss.send_to_google_api(path_to_file)
    utils.utils.save_to_file("sox_splitter_output.txt", text)


    #utils.analysis.plot(path_to_file,ss.get_points())


if(__name__=="__main__"):
    #main("data"+os.sep+"007NTWY_U2_RC.mp3")
    main("/home/leon/DataRoot/100_Trainee/52772751_903.wav")
