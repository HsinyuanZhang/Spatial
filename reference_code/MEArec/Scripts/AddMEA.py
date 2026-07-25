import MEArec as mr
import MEAutility as MEA
import yaml
from pprint import pprint
import matplotlib.pylab as plt


pprint(MEA.return_mea_list())
MEA.add_mea('../Configuration/WIREDOR32X32_36u.yaml')


wiredor_MEA = MEA.return_mea('WIREDOR32X32_36u')
plt.plot(wiredor_MEA.positions[:, 0], wiredor_MEA.positions[:, 1], 'b*')
plt.axis('equal')
plt.show()

pprint(MEA.return_mea_list())




