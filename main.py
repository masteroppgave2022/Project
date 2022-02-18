import os
import logging
import subprocess
import multiprocessing
import data.request as req
# import data.preprocess as pp

if __name__ == '__main__':
    config = 'main_config.cfg'
    logging.basicConfig(filename='main_log.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s',  datefmt='%m/%d/%Y %H:%M:%S')

    with open(config) as cfg:
        for line in cfg:
            parameters = line.strip().split(': ')
            if parameters[0] == 'download':
                download = True if parameters[1].lower() == 'true' else False
            elif parameters[0] == 'preprocess':
                preprocess = True if parameters[1].lower() == 'true' else False

    if download:
        search_configs = 'data/search_configs/'
        all_configs = [search_configs+f for f in os.listdir(search_configs)]
        print(all_configs)

