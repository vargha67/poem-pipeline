import configs
# from old_dissection import settings   # This kind of import does not reflect changes to settings in other codes like feature_operation.py
import settings
import main
import os, shutil



def concept_identification (broden_dataset_path, model_file_path, identification_result_path):
    if os.path.exists(identification_result_path):
        shutil.rmtree(identification_result_path)
    os.makedirs(identification_result_path)

    settings.update_settings(configs.model_name, configs.dataset_name, configs.num_classes, configs.target_layer, 
                             configs.activation_high_thresh, configs.min_iou, identification_result_path, 
                             model_file_path, broden_dataset_path)

    main.run()

