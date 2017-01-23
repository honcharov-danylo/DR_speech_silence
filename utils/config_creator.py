import utils.config_wrapper

class config_creator:
    def __init__(self):
        settings={}
        settings['api_key']=input("Type your api key: ")
        settings['data_path']=input("Type path to your data: ")
        cw=utils.config_wrapper.config_wrapper()
        cw.make_new_config(settings)

if __name__=="__main__" :
    print("Making new configuration:")
    cc=config_creator()
