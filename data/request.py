import asf_search as asf
import geopandas as gpd
from shapely.geometry import box
from datetime import date
import os
import sys

"""
Download object, prompting user for download settings.

TODO: Finish DOCSTRING
TODO: Clean up
TODO: ROI input method (shapefile method is hopefully temporary, but it works)
"""

class searchConstants(object):
    """ Constant list for parsing search config file """
    constants = {
    "Sentinel-1": asf.PLATFORM.SENTINEL1,
    "GRDHD": asf.PRODUCT_TYPE.GRD_HD,
    "GRDMD": asf.PRODUCT_TYPE.GRD_MD,
    "SLC": asf.PRODUCT_TYPE.SLC,
    "IW": asf.BEAMMODE.IW,
    "EW": asf.BEAMMODE.EW,
    "Dual_HH": asf.POLARIZATION.HH_HV,
    "Dual_VV": asf.POLARIZATION.VV_VH,
    "HH": asf.POLARIZATION.HH,
    "VV": asf.POLARIZATION.VV,
    "Ascending": asf.FLIGHT_DIRECTION.ASCENDING,
    "Descending": asf.FLIGHT_DIRECTION.DESCENDING
    }


class requestDownload(searchConstants):

    """ TODO: Create Docstring for help() method """

    def __init__(self,username=None,password=None,search_configs=None,search_mode='roi'):
        self.search_mode = search_mode
        self.username = username
        self.password = password
        self.search_configs = search_configs
        self.authenticate_session()

        self.search_params = {
            "intersectsWith": None,
            "platform": None,
            "processingLevel": None,
            "beamMode": None,
            "polarization": None,
            "flightDirection": None,
            "start": None,
            "end": None,
            "maxResults": None
        }
        self.search()
        
    def authenticate_session(self):
        try:
            self.session = asf.ASFSession().auth_with_creds(self.username,self.password)
            print("[INFO] ASF session validated successfully.")
        except Exception as e:
            print(f"USER AUTHENTICATION FAILED: {e}\n\n\
            Initiate download request as: requestDownload(<username>,<password>,<search_mode>)")

    def search(self):
        print(f"Search mode set to {self.search_mode.upper()}.")
        if not self.search_mode.upper() == 'ROI':
            raise Exception('Script currently only supports requests by Region of Interest (ROI).')
        
        scenes_to_download = []
        for search_config in self.search_configs:
            self.parse_search_config(search_config=search_config)
            print(f"Search parameters:\n{self.search_params}\n")
            scenes_to_download.extend(self.append_search_results())
        if len(scenes_to_download) != 0:
            download = input(f"Total of {len(scenes_to_download)} images found, proceed to download [y/n]? ")
            if download.lower() == 'y':
                results = asf.product_search(scenes_to_download)
                self.download(results=results)
        else:
            print(f"[ABORTING] Total of {len(scenes_to_download)} images found.")


    def parse_search_config(self, search_config):
        with open(search_config) as cfile:
            for line in cfile:
                parameters = line.strip().split(': ')
                if parameters[1] == 'None':
                    self.search_params[parameters[0]] = None
                elif parameters[1] in self.constants.keys():
                    self.search_params[parameters[0]] = self.constants[parameters[1]]
                elif parameters[0] == 'start' or parameters[0] == 'end':
                    date_list = [int(d) for d in parameters[1].split('-')]
                    self.search_params[parameters[0]] = date(date_list[0],date_list[1],date_list[2])
                elif parameters[0] == 'shapefile':
                    gdf = gpd.read_file(parameters[1])
                    bounds = gdf.total_bounds
                    gdf_bounds = gpd.GeoSeries([box(*bounds)])
                    self.search_params['intersectsWith'] = gdf_bounds.to_wkt().values.tolist()[0]
                else:
                    self.search_params[parameters[0]] = int(parameters[1])

    def set_region_of_interest(self):
        """
        After extensive research on folium.py, it turns out that's a one way street from javascript to python.
        Thus, displaying maps using folium is possible, but interacting and fetching coordinate data from the map
        is not as easy. This function may or may not change during my free time... that is, whether or not I'll bother
        to make a leaflet.js ---> geoJSON map in JS. For now: generate shapefiles and give path to it in config file.

        CONSIDER THIS A PLACEHOLDER FUNCTION FOR NOW.
        """
        pass
                
    def append_search_results(self):
        results = asf.search(intersectsWith=self.search_params['intersectsWith'],
                            platform=self.search_params['platform'],
                            beamMode=self.search_params['beamMode'],
                            processingLevel=self.search_params['processingLevel'],
                            polarization=self.search_params['polarization'],
                            flightDirection=self.search_params['flightDirection'],
                            start=self.search_params['start'],
                            end=self.search_params['end'],
                            maxResults=self.search_params['maxResults'])
        print(f"[INFO] Search complete, images found: {len(results)}.")
        metadata = results.geojson()
        # Check for duplicates in search results
        existing_rasters = [os.path.split(f)[1] for f in os.listdir('data/unprocessed_downloads/')]
        scenes_to_download = []
        for result in metadata['features']:
            if result['properties']['fileName'] in existing_rasters:
                print("Raster already downloaded ------> SKIPPING")
            else:
                scenes_to_download.append(result['properties']['fileID'])
        return scenes_to_download


    def download(self, results):
        metadata = results.geojson()
        print(f"Downloading the following results:\n{metadata}")
        results.download(
        path='data/unprocessed_downloads/',
        session=self.session,
        processes=8 
        )
        
        
        

if __name__=='__main__':
    """ Just for testing purposes: """
    request = requestDownload("mikaelvagen", "Masteroppgave2022", search_configs=["data/search_configs/gaula_kvål_06_2021.cfg"])



















    