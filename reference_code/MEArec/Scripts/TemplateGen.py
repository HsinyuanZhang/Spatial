import MEArec as mr
import MEAutility as mu
import yaml
from pprint import pprint
import matplotlib.pylab as plt

default_info, mearec_home = mr.get_default_config()
pprint(default_info)

# define cell_models folder
cell_folder = default_info['cell_models_folder']
template_params = mr.get_default_templates_params()
pprint(template_params)

# the templates are not saved, but the intracellular simulations are saved in 'templates_folder'
tempgen = mr.gen_templates(cell_models_folder=cell_folder, params=template_params, n_jobs=2)

mr.save_template_generator(tempgen, filename='D:/University/SpatialSpikeSorting_GitLab/MEArec/Templates/WIREDOR32X32_36u_templates.h5')


