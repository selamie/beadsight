from datetime import datetime
import shutil
import os

START_TIME = datetime.now()

#CLUSTER:

# FIXED 
DATA_TYPE = ""
CKPT_DIR = '/media/selamg/DATA/beadsight/data/checkpoints/'
#for pretrained clip head
BEADSIGHT_WEIGHTS_PATH = '/media/selamg/DATA/beadsight/data/clipmodels/test_ref/1/epoch_0_beadsight_encoder.pth'
IMAGE_WEIGHTS_PATH = '/media/selamg/DATA/beadsight/data/clipmodels/test_ref/1/epoch_0_vision_encoder.pth'
DATA_DIR = "/media/selamg/DATA/beadsight/data/processed_data"
CODE_START_DIR = '/media/selamg/DATA/beadsight/HardwareTeleop' #temp fix for testing -- pc is a recursive directory lol 
ENC_TYPE = 'clip'  # weights above not actually used here
DEVICE_STR = 'cuda:0'
PRED_HORIZON = 20
ABLATE_BEAD = False 
BEAD_ONLY = False


#SAVING RATE
EPOCHS = 10 #3500
VAL_EVERY = 2 #10

# REMEMBER TO EDIT TRAINING PARAMS(!!)

######### shutil moves code ###############

assert  not(ABLATE_BEAD == True and BEAD_ONLY == True)

print(f"{START_TIME}__STARTING TASK: {DATA_TYPE} WITH {ENC_TYPE} CUDA {DEVICE_STR} HORIZON {PRED_HORIZON} FOR {EPOCHS} EPOCHS")

if ABLATE_BEAD:
    print("ABLATING BEADSIGHT")

if BEAD_ONLY:
    print("BEAD ONLY ABLATING IMAGES")

now_time = START_TIME.strftime("%H-%M-%S_%Y-%m-%d")

CODE_DIR = CKPT_DIR+'/code_'+now_time+'_'+ ENC_TYPE + DATA_TYPE


thisfile = os.path.abspath(__file__)
thisfile_name = os.path.basename(__file__)


os.makedirs(CODE_DIR,exist_ok=True)


shutil.copytree(CODE_START_DIR, CODE_DIR+'/diffusion_plugging')
shutil.copyfile(thisfile, CODE_DIR+'/'+thisfile_name)
