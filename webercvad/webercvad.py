import collections
import contextlib
import sys
import wave
import api_wrapper.api_wrapper as api_wrapper
import scipy.io.wavfile
import re
import json
import os
import matplotlib.pyplot as pyplot
import utils.utils
import numpy as np

import webrtcvad


class webercvad:
    def __init__(self):
        self.chunk_time_list = []
        self.max_len=0


    def read_wave(self,path):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate

    class Frame(object):
        def __init__(self, bytes, timestamp, duration):
            self.bytes = bytes
            self.timestamp = timestamp
            self.duration = duration

    def frame_generator(self,frame_duration_ms, audio, sample_rate):
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield self.Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self,sample_rate, frame_duration_ms,
                      padding_duration_ms, vad, frames):
        counter = -1
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []
        for frame in frames:
            if not triggered:
                ring_buffer.append(frame)
                num_voiced = len([f for f in ring_buffer
                                  if vad.is_speech(f.bytes, sample_rate)])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    counter += 1
                    self.chunk_time_list.append([ring_buffer[0].timestamp])
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append(frame)
                num_unvoiced = len([f for f in ring_buffer
                                    if not vad.is_speech(f.bytes, sample_rate)])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    self.chunk_time_list[counter].append(frame.timestamp + frame.duration)
                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
        if triggered:
            pass
        sys.stdout.write('\n')
        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames])

    def write_wave(self,path, audio, sample_rate):
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

    def parse_audio(self,aggression, path, duration=20,frame_duration_ms=30, padding_duration_ms=300):
        audio, sample_rate = self.read_wave(path)
        vad = webrtcvad.Vad(int(aggression))
        frames = self.frame_generator(frame_duration_ms, audio, sample_rate)
        frames = list(frames)
        segments = self.vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames)

        seg_array={}
        for i, segment in enumerate(segments):
                pass


            # path = 'temp_web/chunk-%002d.wav' % (i,)
            # print(' Writing %s' % (path,))
            # self.write_wave(path, segment, sample_rate)
            # write_wave("temp_web"+os.sep+str(i)+"out.wav",segment,sample_rate)

        print('Success', len(self.chunk_time_list), 'chunks')

        return self.chunk_time_list

    #def parse_audio_with_increasing_aggression(self):


    def send_to_google_api(self,filename, languageCode, agressivity=2):
        """
        Send to google api using api_wrapper
        :param filename: name of input file
        :param languageCode: a BCP-47 language tag
        :param agressivity: agressivity of algorithm
        :return: list of recognized strings
        """
        rate, wave_data = scipy.io.wavfile.read(filename)
        self.max_len=len(wave_data)/rate
        labels = self.parse_audio(agressivity, filename)
        aw = api_wrapper.api_wrapper()
        result = []
        #self.built_plot(filename)
        for speech in labels:
            if (len(speech) != 1):
                current_sound = wave_data[speech[0] * rate:speech[1] * rate]
            else:
                current_sound = wave_data[speech[0]:]
            google_res = aw.send_data(current_sound, rate, languageCode=languageCode)
            # print(google_res)

            regexp = re.findall(r"\":\s*\"[^\"]*", json.dumps(google_res))
            for line in regexp:
                line = bytes(line, "utf-8").decode("unicode_escape")
                result.append(line[4:])
                print(line[4:])
        return result

    def built_plot(self,path):
        fig=pyplot.figure()
        pyplot.plot()
        sample_rate,wave_data=scipy.io.wavfile.read(path)
        wave_data = wave_data / (2. ** 15)
        # choose one chanel
        timeArray = np.arange(0, wave_data.shape[0], 1)
        timeArray = (timeArray / sample_rate) * 1000
        pyplot.plot(timeArray, wave_data,color='red')
        for chunk in self.chunk_time_list:
            if(len(chunk)!=1):
                data_for_plotting = wave_data[chunk[0] * sample_rate:chunk[1] * sample_rate]
                time_array_for_plot=timeArray[chunk[0] * sample_rate:chunk[1] * sample_rate]
                pyplot.plot(time_array_for_plot,data_for_plotting,color="green")
            else:
                time_array_for_plot=timeArray[chunk[0] * sample_rate:]
                data_for_plotting = wave_data[chunk[0] * sample_rate:]
                pyplot.plot(time_array_for_plot,data_for_plotting,color="green")
        pyplot.show()


if(__name__=="__main__"):
    utils.utils.save_to_file("webrtc_vad.txt", webercvad.send_to_google_api("out_test.wav", "ru_RU"))
