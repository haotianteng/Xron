#Initialize xron package, need to run this when first time runnning xron
import os
import wget
import xron
import zipfile
MODEL_URL="https://xronmodel.s3.us-east-1.amazonaws.com/models.zip"
MODEL_PATH=xron.__path__[0]+"/models"
def get_models(args):
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    print("Downloading models...")
    wget.download(MODEL_URL, out=MODEL_PATH)
    print("\nExtracting models...")
    with zipfile.ZipFile(MODEL_PATH+"/models.zip", 'r') as zip_ref:
        zip_ref.extractall(MODEL_PATH)
    os.remove(MODEL_PATH+"/models.zip")
    print("Done!")



    