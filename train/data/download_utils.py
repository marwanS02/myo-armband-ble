import os
import shutil
import zipfile
import time
"""
The shutil library in Python is a standard utility module that provides a range 
of high-level operations on files and collections of files.
"""

import urllib
"""
used for opening and reading URLs, handling URL requests, parsing URLs, and managing 
data from the internet.
"""

def download_url(url, root, filename):
    """
    Download a file with given filename from a given url to a given directory
    url: where to download from
    root: directory where to download to
    filename: filename under which the file should be saved
    """
    file_path = os.path.join(root, url[url.rfind('/')+1:])

    os.makedirs(root, exist_ok=True)
    """
    When set to True, it allows the creation of the directory to succeed even 
    If the directory already exists, the function call will simply pass 
    without making any changes to the directory or its contents. 
    If set to False (the default), an OSError 
    will be raised if the target directory already exists
    """
    final_file_path = os.path.join(root, filename)
    if not os.path.exists(final_file_path):  
     # Check if the path to the location where the file will be installed 
     # already exists if it doesn't => need to download
     print('Downloading ' + url + 'to' + file_path)

     urllib.request.urlretrieve(url,file_path)
     # Rename the downloaded file to the desired filename
     os.rename(file_path, final_file_path)
     #The urllib.request.urlretrieve function is a part of Python's urllib 
     #library, which is used for retrieving data from URLs. This function is 
     #particularly useful for downloading files from the internet and saving 
     #them to a local file system.

def download_dataset(url, root, filename, force_download=False):
    """
    Ensures that the dataset is available in a clean directory, handles existing
    data appropriately, and provides flexibility to force re-downloads when 
    necessary.
    """
    
    #check if the root directory exists or if the folder is empty.
    if not os.path.exists(root) or not os.listdir(root) or force_download:
        #To download from the url check if the directory exists and delete it to 
        # recreate it with download_url if directory doesn"t exist start download_url
        if os.path.exists(os.path.join(root,filename)):
            shutil.rmtree(os.path.join(root,filename))
            #this function is used to delete directory content (files and subdirectories)
        download_url(url,root,filename)
        # Check if the file ends with .zip and extract if necessary
        if url[url.rfind('/')+1:].endswith(".zip"):
            os.rename(os.path.join(root,filename),os.path.join(root,filename+'.zip'))
            # Remove the ".zip" from the filename for extraction purposes
            zip_file_path = os.path.join(root, filename+'.zip')
            os.makedirs(os.path.join(root, filename), exist_ok=True) 
            extract_zip(zip_file_path, os.path.join(root, filename))

def extract_zip(file_path, extract_to='.'):
    """
    Extracts a zip file to a specified directory.
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
