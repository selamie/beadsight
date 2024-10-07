import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--freeze", help="freeze or unfreeze beadsight", type=bool, required=True)
parser.add_argument("--ablateBead", help="ablate beadsight if True, use beadsight if false", type=bool, required=True)
parser.add_argument("--encType", help="", choices=['resnet','clip'], required=True)
parser.add_argument("--beadOnly", help="uses beadsight only if true, otherwise false", type=bool, required=False)

args = parser.parse_args()


#I have done this in a way where I have to really overhaul shit to get this to work....