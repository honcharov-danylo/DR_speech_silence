import configparser

class config_wrapper:
    def __init__(self):
        self.config=configparser.RawConfigParser()

    def make_new_config(self, dict_of_config_entries):
        self.config.add_section('Google_api_section')
        self.config.set('Google_api_section',"api_key", dict_of_config_entries['api_key'])
        self.config.add_section('Settings')
        self.config.set('Settings',"data_path", dict_of_config_entries['data_path'])
        with open('../settings.cfg', 'w') as configfile:
            self.config.write(configfile)

    def read_from_file(self):
        self.config.read('../settings.cfg')
        dict_of_config_values={}
        dict_of_config_values['api_key']=self.config.get('Google_api_section',"api_key")
        dict_of_config_values['data_path']=self.config.get('Settings',"data_path")
        return dict_of_config_values