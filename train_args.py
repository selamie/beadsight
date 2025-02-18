from datetime import datetime
import shutil
import os

START_TIME = datetime.now()



#NODE 3 clip:
DATA_TYPE = "drawer" # only put dataset type here, the rest is handled below
CKPT_DIR = '/home/selam/beadsight_data/drawer_notFrozen_checkpoints/'
#for pretrained clip head
BEADSIGHT_WEIGHTS_PATH = '/home/selam/model_weights/usb_epoch_1499_beadsight_encoder.pth'
IMAGE_WEIGHTS_PATH = '/home/selam/model_weights/usb_epoch_1499_vision_encoder.pth'
#not used for resnet, beadsight wts not used for eef pretraining
DATA_DIR = "/home/selam/processed_drawer/"
CODE_START_DIR = '/home/selam/beadsight' 
ENC_TYPE = 'resnet18' 
DEVICE_STR = 'cuda:0'
PRED_HORIZON = 20
ABLATE_BEAD = False
FREEZE_BEAD = False
PRETRAINED_VISION = False
EEF_WEIGHTS_PATH = '/home/selam/model_weights/eef/epoch_1499_eef_encoder.pth'

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

if PRETRAINED_VISION == False:
    EEF_WEIGHTS_PATH = IMAGE_WEIGHTS_PATH

# REMEMBER TO EDIT TRAINING PARAMS(!!)

######### shutil moves code ###############

assert  not(ABLATE_BEAD == True and BEAD_ONLY == True)
assert  not(ABLATE_BEAD == True and FREEZE_BEAD == True)
assert  not(FREEZE_BEAD == True and BEAD_ONLY == True)
if PRETRAINED_VISION: 
    ABLATE_BEAD = True #there's not really a situation we'd be using randomly init beadsight right...
    assert not(BEAD_ONLY == True or FREEZE_BEAD == True)
    assert not(EEF_WEIGHTS_PATH == IMAGE_WEIGHTS_PATH)
elif not PRETRAINED_VISION:
    assert EEF_WEIGHTS_PATH == IMAGE_WEIGHTS_PATH, "eef weights should be same as image weights unless using pretrained eef encoder"


print(f"{START_TIME}__STARTING TASK: {DATA_TYPE} WITH {ENC_TYPE} CUDA {DEVICE_STR} HORIZON {PRED_HORIZON} FOR {EPOCHS} EPOCHS")


now_time = START_TIME.strftime("%H-%M-%S_%Y-%m-%d")
CODE_DIR = CKPT_DIR+'/code_'+now_time+'_'+ ENC_TYPE + '_'+DATA_TYPE

if ABLATE_BEAD:
    if not PRETRAINED_VISION:
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

if PRETRAINED_VISION: 
    print("USING EEF PRETRAINED ENCODER, ABLATE BEAD BY DEFAULT")
    DATA_TYPE = DATA_TYPE + '_eef_pretrained'
    CODE_DIR = CKPT_DIR+'/code_'+now_time+'_'+ ENC_TYPE +'_'+ DATA_TYPE



thisfile = os.path.abspath(__file__)
thisfile_name = os.path.basename(__file__)


os.makedirs(CODE_DIR,exist_ok=True)


shutil.copytree(CODE_START_DIR, CODE_DIR+'/beadsight')
shutil.copyfile(thisfile, CODE_DIR+'/'+thisfile_name)
