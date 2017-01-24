import sox_splitter
from api_wrapper import api_wrapper
import os


def main(path_to_file):
    ss=sox_splitter.sox_splitter()
    files=ss.split_file_by_silence(path_to_file)
    aw=api_wrapper.api_wrapper()
    for file in files:
        aw.send_file("temp"+os.sep+file)



if(__name__=="__main__"):
    main("data"+os.sep+"007NTWY_U2_RC.mp3")
