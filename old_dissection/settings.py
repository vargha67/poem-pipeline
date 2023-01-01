
# initial settings: 
GPU = True                                  # running on GPU is highly suggested
TEST_MODE = False                           # turning on the testmode means the code will run on a small dataset.
CLEAN = True                                # set to "True" if you want to clean the temporary large files after generating result
QUANTILE = 0.01                             # the threshold used for activation
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
SCORE_THRESHOLD = 0.04                      # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                                # how many process is used for tallying (Experiments show that 1 is the fastest)
CATAGORIES = ["object", "part", "material", "color"] # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
OUTPUT_FOLDER = "identification_results" 	# result will be stored in this folder

MODEL = None
DATASET = None
MODEL_FILE = 'zoo/model.pth'
DATA_DIRECTORY = 'dataset/broden1_224'
MODEL_PARALLEL = False
NUM_CLASSES = 2
FEATURE_NAMES = None
IMG_SIZE = 224

WORKERS = 12
BATCH_SIZE = 128
TALLY_BATCH_SIZE = 16
TALLY_AHEAD = 4
INDEX_FILE = 'index.csv'


def init_settings():
	print('DATA_DIRECTORY:', DATA_DIRECTORY)
	global FEATURE_NAMES

	if MODEL == 'custom':
		FEATURE_NAMES = ['conv_layer6']

	if MODEL == 'alexnet':
		global DATA_DIRECTORY
		global IMG_SIZE
		DATA_DIRECTORY = 'dataset/broden1_227'
		IMG_SIZE = 227
		
	if TEST_MODE:
		global WORKERS
		global BATCH_SIZE
		global TALLY_BATCH_SIZE
		global TALLY_AHEAD
		global INDEX_FILE
		global OUTPUT_FOLDER
		
		WORKERS = 1
		BATCH_SIZE = 4
		TALLY_BATCH_SIZE = 2
		TALLY_AHEAD = 1
		INDEX_FILE = 'index_sm.csv'
		OUTPUT_FOLDER += "_test"

		
