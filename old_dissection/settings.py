
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


def update_settings(model_name, dataset_name, num_classes, target_layer, activation_high_thresh, 
					min_iou, result_path, model_file_path, dataset_path):
	global MODEL
	global DATASET
	global NUM_CLASSES
	global FEATURE_NAMES
	global QUANTILE
	global SEG_THRESHOLD
	global SCORE_THRESHOLD
	global OUTPUT_FOLDER
	global MODEL_FILE
	global DATA_DIRECTORY
	global IMG_SIZE
	global WORKERS
	global BATCH_SIZE
	global TALLY_BATCH_SIZE
	global TALLY_AHEAD
	global INDEX_FILE

	MODEL = model_name
    DATASET = dataset_name
    NUM_CLASSES = num_classes
    FEATURE_NAMES = [target_layer]
    QUANTILE = 1.0 - activation_high_thresh
    SEG_THRESHOLD = min_iou
    SCORE_THRESHOLD = min_iou
    OUTPUT_FOLDER = result_path
    MODEL_FILE = model_file_path
    DATA_DIRECTORY = dataset_path

	if MODEL == 'custom':
		FEATURE_NAMES = ['conv_layer6']

	if MODEL == 'alexnet':
		DATA_DIRECTORY = 'dataset/broden1_227'
		IMG_SIZE = 227
		
	if TEST_MODE:
		WORKERS = 1
		BATCH_SIZE = 4
		TALLY_BATCH_SIZE = 2
		TALLY_AHEAD = 1
		INDEX_FILE = 'index_sm.csv'
		OUTPUT_FOLDER += "_test"
