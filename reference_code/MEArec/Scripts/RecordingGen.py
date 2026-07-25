import MEArec as mr
import MEAutility as mu
import yaml
from pprint import pprint
import matplotlib.pylab as plt

recordings_params = mr.get_default_recordings_params()

runSet = 0
for i in range(runSet,runSet + 25):
    recordings_params['seeds']['spiketrains'] = i

    recgen = mr.gen_recordings(templates='D:/University/SpatialSpikeSorting_GitLab/MEArec/Templates/WIREDOR32X32_36u_templates.h5', params=recordings_params, n_jobs= 2)
    mr.save_recording_generator(recgen, filename='D:/University/SpatialSpikeSorting_GitLab/MEArec/Recording/WIREDOR32X32_recordings_A150to500_N22_500C_D6s_36um_' + str(i) +'.h5')
