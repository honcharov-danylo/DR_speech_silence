import base64
import json
import os
import sox


from googleapiclient import discovery
import httplib2
from oauth2client.client import GoogleCredentials


class api_wrapper:
    def __init__(self):
        #Looks like something terrible. Not use this in production. Ever. EVER!
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.getcwd()+os.sep+"api_wrapper"+os.sep+"speech-recognition-7284f98b659f.json"


    #taken from official documentation
    def get_speech_service(self):
        DISCOVERY_URL = ('https://{api}.googleapis.com/$discovery/rest?'
                         'version={apiVersion}')
        credentials = GoogleCredentials.get_application_default().create_scoped(
            ['https://www.googleapis.com/auth/cloud-platform'])
        http = httplib2.Http()
        credentials.authorize(http)

        return discovery.build(
            'speech', 'v1beta1', http=http, discoveryServiceUrl=DISCOVERY_URL)


    def send_file(self,speech_file,languageCode= 'en-US'):
        """Transcribe the given audio file.

        Args:
            speech_content: the name of the audio file.
        """
        #languageCode='ru-RU'
        tfm = sox.Transformer()
        tfm.remix(num_output_channels=1)
        tfm.build(speech_file, "temp"+os.sep+"result.wav")
        speech_file="temp"+os.sep+"result.wav"
        rate=sox.file_info.sample_rate(speech_file)
        with open(speech_file, 'rb') as speech:
           speech_content = speech.read()
        return self.send_data(speech_content,rate,languageCode)

    def send_data(self,speech_content,rate,languageCode= 'en-US'):
        """Transcribe the given audio data.

        Args:
            speech_content: audio data.
        """
        speech_content=base64.b64encode(speech_content)
        service = self.get_speech_service()
        service_request = service.speech().syncrecognize(
            body={
                'config': {
                    'encoding': 'LINEAR16',  # raw 16-bit signed LE samples
                    'sampleRate': 32000,
                    'languageCode': languageCode,  # a BCP-47 language tag
                },
                'audio': {
                    'content': speech_content.decode('UTF-8')
                }
            })
        response = service_request.execute()
        return response
        #print(response['results'])
        #print(json.dumps(response))
        #return json.dumps(response)

if(__name__=="__main__"):
    file="../data/short.mp3"
    aw=api_wrapper()
    tfm = sox.Transformer()
    tfm.remix(num_output_channels=1)
    tfm.build(file,'../temp/audio.wav')
    aw.send_file('../temp/audio.wav')