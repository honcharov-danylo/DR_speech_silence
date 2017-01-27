import sox_splitter
import utils.analysis
import utils.utils
import os
import vad_splitter
import webercvad.webercvad as webercvad

def main(path_to_file):
    utils.utils.mp3towav(path_to_file,
                         "audio_temp.wav")
    path_to_file="audio_temp.wav"
    ss=sox_splitter.sox_splitter()
    text = ss.send_to_google_api(path_to_file,"ru_RU")
    utils.utils.save_to_file("sox_splitter_output.txt", text)

    #utils.analysis.plot(path_to_file,ss.get_points())


if(__name__=="__main__"):
    #main("data"+os.sep+"007NTWY_U2_RC.mp3")
    #main("/home/leon/DataRoot/100_Trainee/52772751_903.wav")
    #webercvad.webercvad.parse_audio(3, path="out_test2.wav")
    #utils.utils.get_beginning_of_file("out_test.wav","beginning_out_test.wav",60)
    #utils.analysis.plot_audio_chunks_duration("temp_web")
    wb=webercvad.webercvad()
    utils.utils.save_to_file("webrtc_vad.txt",wb.send_to_google_api("beginning_out_test.wav","ru_RU",agressivity=1))
    #main("out_test.wav")
