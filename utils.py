import sys


def save_imported_packages(packages_path):
    # Saving imported packages and their versions: 
    modules_info = []

    for module in sys.modules:
        if len(module.split('.')) > 1:   # ignoring subpackages
            continue

        try:
            modules_info.append((module, sys.modules[module].__version__))
        except:
            try:
                if type(sys.modules[module].version) is str:
                    modules_info.append((module, sys.modules[module].version))
                else:
                    modules_info.append((module, sys.modules[module].version()))
            except:
                try:
                    modules_info.append((module, sys.modules[module].VERSION))
                except:
                    pass

    modules_info.sort(key=lambda x: x[0])
    with open(packages_path, 'w') as f:
        for m in modules_info:
            f.write('{} {}\n'.format(m[0], m[1]))



def extract_class_titles (ds_name, binning_classes):
	ctitles = {}
	name_parts = ds_name.split('_')
	if len(name_parts) <= 1:
		return ctitles
	
	n_classes = len(name_parts[1:])
	for i,p in enumerate(name_parts[1:]):
		ctitles[i] = p
		if binning_classes:
			ctitles[i + n_classes] = 'maybe ' + p
		
	return ctitles

