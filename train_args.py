from datetime import datetime
import shutil
import os

START_TIME = datetime.now()



#NODE 3 clip:
DATA_TYPE = "drawer_withSupporting" # only put dataset type here, the rest is handled below
CKPT_DIR = '/home/selam/beadsight_data/drawer_checkpoints/'
#for pretrained clip head
BEADSIGHT_WEIGHTS_PATH = '/home/selam/model_weights/drawer_supporting_epoch_1499_beadsight_encoder.pth'
IMAGE_WEIGHTS_PATH = '/home/selam/model_weights/drawer_supporting_epoch_1499_vision_encoder.pth'
# BEADSIGHT_WEIGHTS_PATH = '/home/selam/model_weights/original_usb_wts/epoch_1499_beadsight_encoder.pth'
# IMAGE_WEIGHTS_PATH = '/home/selam/model_weights/original_usb_wts/epoch_1499_vision_encoder.pth'
#not used for resnet
DATA_DIR = "/home/selam/processed_drawer/"
CODE_START_DIR = '/home/selam/beadsight' 
ENC_TYPE = 'clip' 
DEVICE_STR = 'cuda:0'
PRED_HORIZON = 20
ABLATE_BEAD = True
FREEZE_BEAD = False

BEAD_ONLY = False #not gonna mess with this

# # local testing:
# DATA_TYPE = "_augmentedFrozen"
# CKPT_DIR = '/media/selamg/DATA/beadsight/data/'
# #for pretrained clip head
# BEADSIGHT_WEIGHTS_PATH = '/media/selamg/DATA/beadsight/data/weights/clip_weights_aug/epoch_1499_beadsight_encoder.pth'
# IMAGE_WEIGHTS_PATH = '/media/selamg/DATA/beadsight/data/weights/clip_weights_aug/epoch_1499_vision_encoder.pth'
# DATA_DIR = "/media/selamg/DATA/beadsight/data/processed_data"
# CODE_START_DIR = '/media/selamg/DATA/beadsight/HardwareTeleop' #throwaway 
# ENC_TYPE = 'clip' 
# DEVICE_STR = 'cuda:0'
# PRED_HORIZON = 20
# ABLATE_BEAD = False 
# BEAD_ONLY = False
# FREEZE_BEAD = True



#SAVING RATE
#TODO:
EPOCHS = 3500
VAL_EVERY = 10

# REMEMBER TO EDIT TRAINING PARAMS(!!)

######### shutil moves code ###############

assert  not(ABLATE_BEAD == True and BEAD_ONLY == True)
assert  not(ABLATE_BEAD == True and FREEZE_BEAD == True)
assert  not(FREEZE_BEAD == True and BEAD_ONLY == True)

print(f"{START_TIME}__STARTING TASK: {DATA_TYPE} WITH {ENC_TYPE} CUDA {DEVICE_STR} HORIZON {PRED_HORIZON} FOR {EPOCHS} EPOCHS")


now_time = START_TIME.strftime("%H-%M-%S_%Y-%m-%d")
CODE_DIR = CKPT_DIR+'/code_'+now_time+'_'+ ENC_TYPE + '_'+DATA_TYPE

if ABLATE_BEAD:
    print("ABLATING BEADSIGHT")
    DATA_TYPE = DATA_TYPE + '_ablate'
    CODE_DIR = CKPT_DIR+'/code_'+now_time+'_'+ ENC_TYPE + '_'+DATA_TYPE

if BEAD_ONLY:
    print("BEAD ONLY ABLATING IMAGES")
    DATA_TYPE = DATA_TYPE + '_beadOnly'
    CODE_DIR = CKPT_DIR+'/code_'+now_time+'_'+ ENC_TYPE +'_'+ DATA_TYPE


if FREEZE_BEAD:
    print("FREEZING BEADSIGHT ENCODER")
    DATA_TYPE = DATA_TYPE + '_freeze'
    CODE_DIR = CKPT_DIR+'/code_'+now_time+'_'+ ENC_TYPE +'_'+ DATA_TYPE


thisfile = os.path.abspath(__file__)
thisfile_name = os.path.basename(__file__)


os.makedirs(CODE_DIR,exist_ok=True)


shutil.copytree(CODE_START_DIR, CODE_DIR+'/beadsight')
shutil.copyfile(thisfile, CODE_DIR+'/'+thisfile_name)
