# POEM: Pattern-Oriented Explanations of CNN Models

This project includes the source codes of the POEM pipeline, which can be used to explain image classifier CNN models 
using patterns of semantic concepts. The following research paper explains POEM's pipeline details and 
the related quantitative and qualitative experiments: 

```
Vargha Dadvar, Lukasz Golab, and Divesh Srivastava. 2023. 
POEM: Pattern-Oriented Explanations of Convolutional Neural Networks. 
Proc. VLDB Endow. 16, 11 (July 2023), 3192–3200. 
https://doi.org/10.14778/3611479.3611518
```

The POEM's web-based demo system, which is based on an older version of the pipeline and 
only shows patterns without any inactivated (No) concepts, is introduced in the following paper: 

```
Vargha Dadvar, Lukasz Golab, and Divesh Srivastava. 2022. 
POEM: Pattern-Oriented Explanations of CNN Models. 
Proc. VLDB Endow. 15, 12 (August 2022), 3618–3621. 
https://doi.org/10.14778/3554821.3554858
```

## Running Instructions: 

The project can be run on a Unix-based system with an Nvidia CUDA-enabled GPU by executing the following commands. 
The configs including the target CNN model and dataset can be changed in the `configs` file. 
Note that running the entire pipeline can take a few hours depending on the model depth, dataset size, pattern mining methods, and other configs.

```
git clone https://github.com/vargha67/poem-pipeline
cd poem-pipeline
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

However, in order to reduce the chance of environment and GPU errors, 
we recommend running in a [Google Colab](https://colab.research.google.com/) environment.
A Jupyter notebook called `run_poem_pipeline` is provided, which can be uploaded and run in Google Colab. 

The notebook clones the project, installs the requirements, changes the default configs if needed, 
runs the pipeline, and finally saves the results back to a user-provided Google Drive directory. 

The main configs that can be set in the notebook include the pipeline steps to run, the target CNN model and dataset, 
whether to use the older version of network dissection (CNN2DT) or POEM, the pattern mining methods used, 
and the minimum support values to test for pattern mining. 
Other configs can be set in the `configs` file in the project itself. 

To save the results in a Google Drive directory, user's Google Drive should be connected to the Colab environment. 
Also the `drive_base_results_path` variable should be set based on the user's desired results path in Google Drive. 
In case of running a subset of the pipeline steps, the results path should include the outputs from the other steps, 
so that they are copied by the notebook to the project before running. 

Current CNN models that can be set in the configs include: 

* ResNet-18
* ResNet-50
* VGG-16

Current target datasets available include: 

* Places dataset subset including bedroom, kitchen, and living room classes
* Places dataset subset including coffeeshop and restaurant classes
* ImageNet dataset subset including minivan and pickup classes
* ImageNet dataset subset including laptop and mobile classes

The details and Google Drive paths to other CNN models and datasets can be set in the configs 
by adding entries to the `model_settings` and `dataset_settings` configs. 

