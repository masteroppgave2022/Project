import os
import logging
import subprocess
import configparser
import multiprocessing
import data.request as req
# import data.preprocess as pp

if __name__ == '__main__':
    logging.basicConfig(filename='main_log.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s',  datefmt='%m/%d/%Y %H:%M:%S')
    parser_main = configparser.ConfigParser()
    parser_main.read('main_config.ini')

    if parser_main.getboolean('main','download'):
        user = parser_main['download']['username']
        pw = parser_main['download']['password']


        search_configs = 'data/search_configs/'
        parser_locations = configparser.ConfigParser()
        parser_locations.read(search_configs+'LOCATIONS.ini')
        configs = []
        for f in os.listdir(search_configs):
            loc = f.split('.')[0]
            if loc.lower() == 'locations': continue
            if parser_locations.getboolean('download',loc):
                configs.append(search_configs+f)
        
        request = req.requestDownload(username=user,password=pw,search_configs=configs)






    if parser_main.getboolean('main', 'preprocess'):
        """ Processing pipeline to be implemented here """
        pass


    

