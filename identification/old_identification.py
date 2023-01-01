import configs
# from old_dissection import settings   # This kind of import does not reflect changes to settings in other codes like feature_operation.py
import settings
import main



def concept_identification (dataset_path, model_file_path, result_path):
    settings.update_settings(configs.model_name, configs.dataset_name, configs.num_classes, configs.target_layer, 
                             configs.activation_high_thresh, configs.min_iou, result_path, model_file_path, dataset_path)

    main.run()

