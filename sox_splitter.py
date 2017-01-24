import sox
import shutil
import os

class sox_splitter:
    def split_file_by_silence(self,path,max_duration=20,threshold=1):
        shutil.rmtree("temp")
        os.makedirs("temp")
        filename, file_extension = os.path.splitext(path)
        shutil.copy(path,"temp"+os.sep+"audio."+file_extension)
        return self.__split_file_by_silence("temp"+os.sep+"audio."+file_extension,max_duration,threshold)

    def __split_file_by_silence(self,path,max_duration,threshold=1):
        sox.core.sox([path, "temp"+os.sep+str(threshold)+"out.wav","silence","1","0.5","1%","1","5.0",str(threshold)+"%",":","newfile",":","restart"])
        os.remove(path)
        files=os.listdir("temp")
        for file in files:
            if(os.path.exists("temp"+os.sep+file)):
                if(sox.file_info.duration("temp"+os.sep+file)>max_duration):
                    self.__split_file_by_silence("temp"+os.sep+file,max_duration,threshold+1)
        return os.listdir("temp")
        #return

if(__name__=="__main__"):
    ss=sox_splitter()
    files=ss.split_file_by_silence("data/003NTWY_U1_CL.mp3")
