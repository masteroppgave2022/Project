import asf_search as asf
import geopandas as gpd
from shapely.geometry import box
from datetime import date

"""
Download object, prompting user for download settings.

TODO: Finish DOCSTRING
TODO: requestDownload.modify_search()
TODO: Dually accessible directory for data storage
TODO: ROI input method
"""

class requestDownload():

    """ TODO: Create Docstring for help() method """

    def __init__(self,username=None,password=None,search_mode='roi'):
        self.search_mode = search_mode
        self.username = username
        self.password = password
        self.authenticate_session()
        self.request_search()

    def authenticate_session(self):
        try:
            self.session = asf.ASFSession().auth_with_creds(self.username,self.password)
            print("[INFO] ASF session validated successfully.")
        except Exception as e:
            print(f"USER AUTHENTICATION FAILED: {e}\n\n\
            Initiate download request as: requestDownload(<username>,<password>,<search_mode>)")

    def request_search(self):
        print(f"Search mode set to {self.search_mode}.")
        if not self.search_mode.upper() == 'ROI':
            print('Script currently only supports requests by Region of Interest (ROI).')
        
        default_search = {
            self.platform: asf.PLATFORM.SENTINEL1,
            self.processingLevel: asf.PRODUCT_TYPE.GRD_HD,
            self.beamMode: asf.BEAMMODE.IW,
            self.polarization: asf.POLARIZATION.DUAL_HH,
            self.flightDirection: None,
            self.lookDirection: None,
            self.start: date(2022,1,1),
            self.end: date(2022,1,31),
            self.maxResults: 2
        }

        print(f"Search parameters:\n{default_search}")
        search = input("Proceed to search [s], modify parameters [m], or cancel [anything else]? ")

        if search.lower() == 's':
            self.search()
        elif search.lower() == "m":
            self.modify_search()
        else:
            raise Exception("Search and download cancelled by user.")
    
    def search(self):
        results = asf.search(platform=self.platform,processingLevel=self.processingLevel,
                            beamMode=self.beamMode,polarization=self.polarization,
                            flightDirection=self.flightDirection,lookDirection=self.lookDirection,
                            start=self.start,end=self.end,maxResults=self.maxResults)
        print(f"[INFO] Search complete, total images found: {len(results)}.")
        download = input("Download images [y/n]?")
        if download.lower() == 'y': self.download(results)

    def modify_search(self, search_params):
        pass

    def download(self, results):
        metadata = results.geojson()
        print(f"Downloading the following results:\n{metadata}")
        results.download(
            path='S1_lvl1_products/', # Temporary
            session=self.session,
            processes=2 # May be able to increase this?
        )


if __name__=='__main__':
    """ Just for testing purposes: """
    request = requestDownload("mikaelvagen", "Drage127")

    