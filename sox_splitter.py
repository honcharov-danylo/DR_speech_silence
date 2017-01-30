import sox
import shutil
import os
import glob
import api_wrapper
import re
import json
import matplotlib.pyplot as pyplot
import scipy.io.wavfile
from matplotlib.pyplot import cm
import numpy as np

class sox_splitter:
    def __init__(self):
        self.points={}
        self.ordered_files=[]
        self.current_filename=""

    def split_file_by_silence(self,path,max_duration=20,threshold=1):
        """
        :param path: path to input file
        :param max_duration: max duration of file which not be splitted
        :param threshold: default=1,start value of threshold
        :return: list of ordered files, splitted by silence
        """
        self.current_filename=path
        if(os.path.exists("temp")):
            shutil.rmtree("temp")
        os.makedirs("temp")
        filename, file_extension = os.path.splitext(path)
        shutil.copy(path,"temp"+os.sep+"audio"+file_extension)
        self.ordered_files.append("temp" + os.sep + "audio" + file_extension)
        return self.__split_file_by_silence("temp"+os.sep+"audio"+file_extension,max_duration,threshold)

    def __split_file_by_silence(self,path,max_duration,threshold=0.1):
        #sox.core.sox([path, "temp"+os.sep+str(threshold)+"out.wav","silence","1","0.5","1%","1","5.0",str(threshold)+"%",":","newfile",":","restart"])
        sox.core.sox([path, "temp" + os.sep + str(threshold) + "out.wav", "silence","-l", "0", "0.5", "5.0",
                      str(threshold) + "%", ":", "newfile", ":", "restart"])

        os.remove(path)

        #statistics
        oldindex=self.ordered_files.index(path)
        self.ordered_files.remove(path)
        list_of_new_files=sorted(glob.glob("temp"+os.sep+str(threshold)+"*"))

        for i in range(len(list_of_new_files)):
            self.ordered_files.insert(oldindex+i,list_of_new_files[i])
        #end statistics

        files=os.listdir("temp")
        for file in files:
            if (os.path.exists("temp" + os.sep + file)):
                if (sox.file_info.duration("temp" + os.sep + file) > max_duration):
                    self.__split_file_by_silence("temp" + os.sep + file, max_duration, threshold + 1)
        return self.ordered_files

    def get_points(self):
        cumsum=0
        points={}
        for file in self.ordered_files:
            cumsum+=sox.file_info.duration(file)
            name=os.path.basename(file)
            points[cumsum]=int(name[0:name.find("out")])
        print(points)
        return points

    def send_to_google_api(self,filename,langCode="en_US"):
        """
        Send to google api using api_wrapper
        :param filename: name of input file
        :param languageCode: a BCP-47 language tag
        :return: list of recognized strings
        """
        files = self.split_file_by_silence(filename)
        print(self.ordered_files)
        # utils.analysis.plot(path_to_file,ss.get_points())
        self.built_plot()
        pyplot.show()
        aw = api_wrapper.api_wrapper.api_wrapper()
        result=[]
        for file in files:
            res = aw.send_file(file,languageCode=langCode)
            #print(res)
            regexp = re.findall(r"\":\s*\"[^\"]*", json.dumps(res))
            for line in regexp:
                line=bytes(line, "utf-8").decode("unicode_escape")
                result.append(line[4:])
                print(line[4:])
        return result

    def built_plot(self,path="",name=""):
        if path=="":
            path=self.current_filename


        if name != "":
            fig = pyplot.figure(name)
        else:
            fig=pyplot.figure(figsize=(10,5))
        fig.add_subplot()

        #pyplot.plot()
        sample_rate,wave_data=scipy.io.wavfile.read(path)
        wave_data = wave_data / (2. ** 15)
        # choose one chanel
        timeArray = np.arange(0, wave_data.shape[0], 1)
        timeArray = (timeArray / sample_rate) * 1000

        points = self.get_points()
        chunk_time_list = []
        chunk_time_list.append([0, list(points.keys())[0]])
        for p in points:
            try:
                chunk_time_list.append([p, points[list(points.keys()).index(p) + 1]])
            except:
                chunk_time_list.append([p,len(wave_data)/sample_rate])


        pyplot.plot(timeArray, wave_data,color='red')
        rainb=cm.rainbow(np.linspace(0, 1, 1.2 * len(chunk_time_list)))[:len(chunk_time_list)]
        np.random.shuffle(rainb)
        color = iter(rainb)


        for chunk in chunk_time_list:
            c = next(color)
            if(len(chunk)!=1):
                data_for_plotting = wave_data[chunk[0] * sample_rate:chunk[1] * sample_rate]
                time_array_for_plot=timeArray[chunk[0] * sample_rate:chunk[1] * sample_rate]

                pyplot.plot(time_array_for_plot,data_for_plotting,color=c)
            else:
                time_array_for_plot=timeArray[chunk[0] * sample_rate:]
                data_for_plotting = wave_data[chunk[0] * sample_rate:]
                pyplot.plot(time_array_for_plot,data_for_plotting,color=c)

        for p in points:
            pyplot.text(p*1000,0,points[p],color='black')
