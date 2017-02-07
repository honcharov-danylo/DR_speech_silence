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
import pandas as pd

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

    def parse_audio_from_file(self,agression,path,frame_duration_ms=30, padding_duration_ms=300):
        self.current_filename=path
        audio, sample_rate = self.read_wave(path)
        res=self.parse_audio(agression,audio,sample_rate,frame_duration_ms=frame_duration_ms, padding_duration_ms=padding_duration_ms)
        self.chunk_time_list=[]
        return res


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
            #self.write_wave("temp_web"+os.sep+str(i)+"out.wav",segment,sample_rate)
            pass

        print('Success', len(self.chunk_time_list), 'chunks')

        return self.chunk_time_list

    def write_wave_by_list(self,chunk_list):
        audio, sample_rate = self.read_wave(self.current_filename)
        for chunk in chunk_list:
            path = 'temp_web/chunk-%002d.wav' % chunk_list.index(chunk)
            print(' Writing %s' % (path,))
            self.write_wave(path,audio[int(chunk[0]*sample_rate):int(chunk[1]*sample_rate)], sample_rate)
            #self.write_wave("temp_web"+os.sep+str(i)+"out.wav",segment,sample_rate)


    def rolling_window(self,a, size):
        shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def rolling_sum(self,a, n):
        ret = np.cumsum(a, axis=0, dtype=int)
        #ret = np.cumsum(a, axis=1, dtype=int)
        #ret[:, n:] = ret[:, n:] - ret[:, :-n]
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:]

    def find_silent_point(self,audio,sample_rate,duration_of_silence):
        """
        Find silent point. Work only with frames with size 2 bytes.
        :param audio:audio data
        :param sample_rate: rate of audio data
        :param duration_of_silence:
        :return: time of silent point
        """
        audio_array = np.frombuffer(audio, dtype=np.int16)
        assert len(audio_array) == len(audio) / 2

        audio_array = audio_array / (2. ** 15)

        audio_array.setflags(write=1)
        binary_array=np.ones(len(audio_array))
        silence_level=0
        while(sum(binary_array)>(len(binary_array)/5)):
            silence_level+=1
            binary_array = audio_array.copy()
            silence_level_value = audio_array.max() * silence_level / 100
            binary_array[np.abs(binary_array) < silence_level_value] = 0
            binary_array[binary_array != 0] = 1

        # timeArray = np.arange(0, audio_array.shape[0], 1)
        # timeArray = (timeArray / sample_rate) * 1000
        # fig = pyplot.figure()
        # fig.add_subplot()
        # pyplot.plot(timeArray, binary_array)
        # pyplot.show()
        len_of_seq = duration_of_silence * sample_rate
        # len_of_seq=100
        # sequency=np.zeros(len_of_seq)
        # points_filter=np.all(self.rolling_window(audio_array,len_of_seq) == sequency)

        # print(audio[points_filter])
        sums_array = self.rolling_sum(binary_array, n=len_of_seq)
        # if(duration_of_silence>=0.01):
        #     return self.find_silent_point(audio[int(np.argmin(sums_array)*2)-(int(np.argmin(sums_array)*2)%2):int((np.argmin(sums_array) + len_of_seq)*2)-int(int((np.argmin(sums_array) + len_of_seq)*2)%2)],sample_rate,duration_of_silence=duration_of_silence*0.9)
        #sums_array=pd.rolling_sum(pd.S)
        #print("sum",sums_array.min())
        #min_v=sums_array.min()
        #print(np.where(sums_array==min_v))
        #return (np.where(sums_array==min_v) + len_of_seq) / sample_rate
        #return (audio_array[np.argmin(sums_array):np.argmin(sums_array)+len_of_seq].argmin()) / sample_rate
        return (np.argmin(sums_array) + len_of_seq) / sample_rate

    def split_by_silence_points(self,audio,sample_rate,duration_of_silence,duration):
        """
        Splitting audio in the most silent points. Work only with 2 bytes frames.
        :param audio: audio data
        :param sample_rate: rate of audio data
        :param duration: duration after splitting
        :return: list of points to splitting
        """
        points=[]
        point=self.find_silent_point(audio,sample_rate,duration_of_silence)
        #print(point)
        points.append([0,point])
        #print("append",[0,point])
        ind = int(point * sample_rate) - (int(point * sample_rate) % 2)
        print("coord",ind)
        if(len(audio[ind:])/sample_rate>duration):
            p=self.split_by_silence_points(audio[ind:],sample_rate,duration_of_silence,duration)
            for elem in p:
                elem[0]+=point
                elem[1]+=point
            points.extend(p)
            print("extend",p)
            return points
        if(len(audio[:ind])/sample_rate>duration):
            points.extend(self.split_by_silence_points(audio[:ind], sample_rate, duration_of_silence, duration))
            print("extend",self.split_by_silence_points(audio[:ind], sample_rate, duration_of_silence, duration))
            return points
        else:
            points.append([point, len(audio) / sample_rate])
            #print("append", [point, len(audio) / sample_rate])
            #print("marker final")
        return points


    def parse_audio_with_increasing_aggression(self, start_aggression, path, duration, min_legal_duration=1, frame_duration_ms=30, padding_duration_ms=300, max_agression=3):
        """

        :param start_aggression:
        :param path:
        :param duration:
        :param min_legal_duration:
        :param frame_duration_ms:
        :param padding_duration_ms:
        :param max_agression:
        :return:
        """
        self.current_filename=path
        audio, sample_rate = self.read_wave(path)
        self.parse_audio(start_aggression, audio, sample_rate, frame_duration_ms=frame_duration_ms, padding_duration_ms=padding_duration_ms)
        #self.built_plot(name="agression0")
        # first_durations=[];
        temp_list=copy.deepcopy(self.chunk_time_list)
        if(len(temp_list[-1])==1):
            temp_list[-1].append(self.max_len)

        # for chunk in temp_list:
        #     first_durations.append(chunk[1]-chunk[0])
        # first_mean=np.sum(first_durations) / len(temp_list)
        #self.built_plot()
        print(temp_list)
        for aggression in range(start_aggression+1, max_agression+1):
            i=0
            while(i<len(temp_list)):
                chunk=temp_list[i]
                if((chunk[1]-chunk[0])>duration):
                    #print(chunk,chunk[1]-chunk[0])
                    self.chunk_time_list=[]
                    new_list=self.parse_audio(aggression,audio[int(chunk[0]*sample_rate):int(chunk[1]*sample_rate)],sample_rate,frame_duration_ms=frame_duration_ms, padding_duration_ms=padding_duration_ms)
                    if(len(new_list)!=0):
                        # print("index of old file",i)
                        # if(len(new_list[-1])==1):
                        #     print(chunk)
                        #     print(new_list[-1][0]+chunk[0],"error",chunk[1]-chunk[0])
                        if(len(new_list[-1])==1):
                            new_list[-1].append(chunk[1]-chunk[0])
                        # print("old len -",(chunk[1]-chunk[0]))
                        # print("now we have ",len(new_list))
                        temp_list.remove(chunk)
                        for ch in range(len(new_list)):
                            #print("new len of ",ch, (new_list[ch][1]  - new_list[ch][0] ))
                            new_list[ch][0] += chunk[0]
                            new_list[ch][1] += chunk[0]
                            temp_list.insert(i+ch,new_list[ch])
                        print("new list",chunk, new_list)
                        #i+=len(new_list)-1
                i+=1
            print(aggression,temp_list)
        #print("\r\nAfter all splitting we have ",len(temp_list)," chunks")
        # self.built_plot()
        # pyplot.show()
        # second_durations=[]
        # for chunk in temp_list:
        #     second_durations.append(chunk[1]-chunk[0])
        # second_mean=np.sum(second_durations)/len(temp_list)
        #print("Means differ ",first_mean,second_mean)

        durations = []
        list_betw=copy.deepcopy(temp_list)
        for i in range(len(temp_list)):
            chunk=temp_list[i]
            durations.append(chunk[1]-chunk[0])
            if((chunk[1]-chunk[0])>duration):
                print(chunk)
                new_chunk_array=self.split_by_silence_points(audio[int(chunk[0] * sample_rate+int((chunk[0] * sample_rate)%2)):int(chunk[1] * sample_rate)-int(int(chunk[1] * sample_rate)%2)], sample_rate, 1,
                                           duration)
                temp_list.remove(chunk)
                for ch in range(len(new_chunk_array)):
                    new_chunk_array[ch][0]+=chunk[0]
                    new_chunk_array[ch][1]+=chunk[0]
                    temp_list.insert(i+ch,new_chunk_array[ch])
                    print(new_chunk_array[ch])


        #self.split_by_silence_points(audio[int(268*sample_rate):int(285*sample_rate)],sample_rate,1,duration)
        self.chunk_time_list=[]
        for chunk in temp_list:
            if(chunk[1]-chunk[0]<min_legal_duration): temp_list.remove(chunk)
        self.chunk_time_list = temp_list

        #self.built_plot()

        #print(len(temp_list), temp_list)
        self.chunk_time_list=list_betw
        #self.built_plot(name="betw")
        #pyplot.show()
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
        #return []
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

    def built_plot(self,path="",name=""):
        if path=="":
            path=self.current_filename


        if name != "":
            fig = pyplot.figure(name,figsize=(10,5))
        else:
            fig=pyplot.figure(figsize=(10,5))
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

