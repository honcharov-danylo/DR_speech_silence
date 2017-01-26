import vad.vad
import utils.analysis
import os
import utils.utils
import api_wrapper.api_wrapper as api_wrapper
import re
import json

class vad_splitter:
    def vad_split(self,filename):
        """
        Finding speech labels
        :param filename: name of input file
        :return: dictionary of speech information (when starts and when ends)
        """
        self.voice_activity_detection = vad.vad.VoiceActivityDetector(filename)
        speech_labels=self.voice_activity_detection.convert_windows_to_readible_labels(self.voice_activity_detection.detect_speech())
        print(speech_labels)
        return speech_labels

    def send_to_google_api(self,filename):
        """
        Send to google api using api_wrapper
        :param filename: name of input file
        :return: list of recognized strings
        """
        labels=self.vad_split(filename)
        aw = api_wrapper.api_wrapper()
        result=[]
        for speech in labels:
            current_sound = v.voice_activity_detection.data[
                            speech['speech_begin'] * v.voice_activity_detection.rate:speech[
                                                                                         'speech_end'] * v.voice_activity_detection.rate]
            google_res = aw.send_data(current_sound, v.voice_activity_detection.rate)
            regexp = re.findall(r"\":\s*\"[^\"]*", json.dumps(google_res))
            for line in regexp:
                line = bytes(line, "utf-8").decode("unicode_escape")
                result.append(line[4:])
                print(line[4:])
        return result

if(__name__=="__main__"):
    #utils.utils.mp3towav("data" + os.sep + "007NTWY_U2_RC.mp3", "temp" + os.sep + "audio_vad.wav") #testing on some file
    v=vad_splitter()
    text=v.send_to_google_api("/home/leon/DataRoot/100_Trainee/52772751_903.wav")
    utils.utils.save_to_file("vad_splitter_output.txt",text)