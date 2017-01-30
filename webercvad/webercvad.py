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
import copy
from matplotlib.pyplot import cm

import webrtcvad


class webercvad:
    def __init__(self):
        self.chunk_time_list = []
        self.max_len=0
        self.current_filename=""

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

    def parse_audio(self,aggression, audio,sample_rate,frame_duration_ms=30, padding_duration_ms=300):
        #audio, sample_rate = self.read_wave(path)
        vad = webrtcvad.Vad(int(aggression))
        frames = self.frame_generator(frame_duration_ms, audio, sample_rate)
        frames = list(frames)
        segments = self.vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames)

        for i, segment in enumerate(segments):
            # path = 'temp_web/chunk-%002d%002d.wav' % (i,aggression)
            # print(' Writing %s' % (path,))
            # self.write_wave(path, segment, sample_rate)
            # self.write_wave("temp_web"+os.sep+str(i)+"out.wav",segment,sample_rate)
            pass

        print('Success', len(self.chunk_time_list), 'chunks')

        return self.chunk_time_list

    def rolling_window(self,a, size):
        shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



    def split_by_silence_points(self,audio,sample_rate,duration_of_silence,duration,silence_level=10):
        """
        Splitting audio in the mosts silent points. Work only with 2 bytes frames.
        :param audio: audio data
        :param sample_rate: rate of audio data
        :param duration: duration after splitting
        :return: list of points to splitting
        """
        audio_array=np.frombuffer(audio,dtype=np.int16)

        audio_array = audio_array / (2. ** 15)

        timeArray = np.arange(0, audio_array.shape[0], 1)
        timeArray = (timeArray / sample_rate) * 1000
        fig=pyplot.figure()
        fig.add_subplot()
        pyplot.plot(timeArray, audio_array)


        audio_array.setflags(write=1)
        silence_level=audio_array.max()*silence_level/100
        assert len(audio_array)==len(audio)/2
        audio_array[np.abs(audio_array)<silence_level]=0
        audio_array[audio_array!=0]=1

        len_of_seq=duration_of_silence*sample_rate
        #len_of_seq=100
        sequency=np.zeros(len_of_seq)
        points_filter=np.all(self.rolling_window(audio_array,len_of_seq) == sequency)
        print(audio[points_filter])


    def parse_audio_with_increasing_aggression(self,start_agression,path,duration,min_legal_duration=1,frame_duration_ms=30, padding_duration_ms=300,max_agression=3):
        self.current_filename=path
        audio, sample_rate = self.read_wave(path)
        self.parse_audio(start_agression,audio,sample_rate,frame_duration_ms=frame_duration_ms, padding_duration_ms=padding_duration_ms)
        # first_durations=[];
        temp_list=copy.deepcopy(self.chunk_time_list)
        temp_list[-1].append(self.max_len)
        # for chunk in temp_list:
        #     first_durations.append(chunk[1]-chunk[0])
        # first_mean=np.sum(first_durations) / len(temp_list)
        #self.built_plot()
        for aggression in range(start_agression,max_agression+1):
            i=0
            while(i<len(temp_list)):
                chunk=temp_list[i]
                if((chunk[1]-chunk[0])>duration):
                    self.chunk_time_list=[]
                    new_list=self.parse_audio(aggression,audio[int(chunk[0]*sample_rate):int(chunk[1]*sample_rate)],sample_rate,frame_duration_ms=frame_duration_ms, padding_duration_ms=padding_duration_ms)
                    if(len(new_list)!=0):
                        # print("index of old file",i)
                        # if(len(new_list[-1])==1):
                        #     print(chunk)
                        #     print(new_list[-1][0]+chunk[0],"error",chunk[1]-chunk[0])
                        new_list[-1].append(chunk[1]-chunk[0])
                        # print("old len -",(chunk[1]-chunk[0]))
                        # print("now we have ",len(new_list))
                        temp_list.remove(chunk)
                        for ch in range(len(new_list)):
                            #print("new len of ",ch, (new_list[ch][1]  - new_list[ch][0] ))
                            new_list[ch][0] += chunk[0]
                            new_list[ch][1] += chunk[0]
                            temp_list.insert(i+ch,new_list[ch])
                i+=1

        #print("\r\nAfter all splitting we have ",len(temp_list)," chunks")
        # self.built_plot()
        # pyplot.show()
        # second_durations=[]
        # for chunk in temp_list:
        #     second_durations.append(chunk[1]-chunk[0])
        # second_mean=np.sum(second_durations)/len(temp_list)
        #print("Means differ ",first_mean,second_mean)

        # durations = []
        # for chunk in temp_list:
        #     durations.append(chunk[1]-chunk[0])
        self.split_by_silence_points(audio,sample_rate,0.5,duration)
        for chunk in temp_list:
            if(chunk[1]-chunk[0]<min_legal_duration): temp_list.remove(chunk)
        self.chunk_time_list = temp_list
        self.built_plot()
        pyplot.show()

        return temp_list

    def send_to_google_api(self,filename, languageCode, agressivity=1,duration=15):
        """
        Send to google api using api_wrapper
        :param filename: name of input file
        :param languageCode: a BCP-47 language tag
        :param agressivity: agressivity of algorithm
        :return: list of recognized strings
        """
        rate, wave_data = scipy.io.wavfile.read(filename)
        self.max_len=len(wave_data)/rate
        #labels = self.parse_audio(agressivity, filename)
        labels=self.parse_audio_with_increasing_aggression(agressivity,filename,duration)
        aw = api_wrapper.api_wrapper()
        result = []
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

    def built_plot(self,path=""):
        if path=="":
            path=self.current_filename
        fig=pyplot.figure()
        fig.add_subplot()
        #pyplot.plot()
        sample_rate,wave_data=scipy.io.wavfile.read(path)
        wave_data = wave_data / (2. ** 15)
        # choose one chanel
        timeArray = np.arange(0, wave_data.shape[0], 1)
        timeArray = (timeArray / sample_rate) * 1000
        pyplot.plot(timeArray, wave_data,color='red')
        rainb=cm.rainbow(np.linspace(0, 1, 1.2 * len(self.chunk_time_list)))[:len(self.chunk_time_list)]
        np.random.shuffle(rainb)
        color = iter(rainb)

        for chunk in self.chunk_time_list:
            c = next(color)
            if(len(chunk)!=1):
                data_for_plotting = wave_data[chunk[0] * sample_rate:chunk[1] * sample_rate]
                time_array_for_plot=timeArray[chunk[0] * sample_rate:chunk[1] * sample_rate]

                pyplot.plot(time_array_for_plot,data_for_plotting,color=c)
            else:
                time_array_for_plot=timeArray[chunk[0] * sample_rate:]
                data_for_plotting = wave_data[chunk[0] * sample_rate:]
                pyplot.plot(time_array_for_plot,data_for_plotting,color=c)



if(__name__=="__main__"):
    utils.utils.save_to_file("webrtc_vad.txt", webercvad.send_to_google_api("out_test.wav", "ru_RU"))
