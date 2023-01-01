import configs
from old_dissection import settings
from old_dissection import main



def concept_identification (dataset_path, model_file_path, result_path):
    print(configs.old_process)
    settings.MODEL = configs.model_name
    settings.DATASET = configs.dataset_name
    settings.NUM_CLASSES = configs.num_classes
    settings.FEATURE_NAMES = [configs.target_layer]
    settings.QUANTILE = 1.0 - configs.activation_high_thresh
    settings.SEG_THRESHOLD = configs.min_iou
    settings.SCORE_THRESHOLD = configs.min_iou
    settings.OUTPUT_FOLDER = result_path
    settings.MODEL_FILE = model_file_path
    settings.DATA_DIRECTORY = dataset_path

    settings.init_settings()
    main.run()

