import sox_splitter
import utils.analysis
import utils.utils
import os
import vad_splitter
import webercvad.webercvad as webercvad
import time

def main(path_to_file):
    utils.utils.mp3towav(path_to_file,
                         "audio_temp.wav")
    path_to_file="audio_temp.wav"
    ss=sox_splitter.sox_splitter()
    text = ss.send_to_google_api(path_to_file,"ru_RU")
    utils.utils.save_to_file("sox_splitter_output.txt", text)

    #utils.analysis.plot(path_to_file,ss.get_points())


if(__name__=="__main__"):
    #utils.utils.get_beginning_of_file("ask.wav","beginning_out_test_2.wav",180)
    #utils.utils.change_bitrate("53147366_906.wav","new_bitrate_function_testing.wav",32000)
    #utils.utils.get_beginning_of_file("out_test.wav","beginning_out_test.wav",180)

  #  time_start=time.time()
    #main("beginning_out_test.wav")
    #time_end=time.time()
    #print(time_end-time_start)
    #main("beginning_out_test_2.wav")

    #main("ask.wav")
    #main("/home/leon/DataRoot/100_Trainee/52772751_903.wav")
    #webercvad.webercvad.parse_audio(3, path="out_test2.wav")
    #utils.analysis.plot_audio_chunks_duration("temp_web")
    wb=webercvad.webercvad()
    #print(wb.parse_audio_from_file(1,"beginning_out_test.wav"))
    #lst=wb.parse_audio_with_increasing_aggression(0,"beginning_out_test.wav",15,max_agression=3)
    #print(lst)
    #wb.write_wave_by_list(lst)
    #time_start = time.time()
    utils.utils.save_to_file("webrtc_vad.txt", wb.send_to_google_api("beginning_out_test.wav", "ru_RU", agressivity=1))

    # time_end = time.time()
    # print(time_end-time_start)
    #utils.utils.save_to_file("webrtc_vad.txt", wb.send_to_google_api("beginning_out_test.wav", "ru_RU", agressivity=1))
    #utils.utils.save_to_file("webrtc_vad.txt",wb.send_to_google_api("d2.wav","ru_RU",agressivity=1))
    #main("out_test.wav")
