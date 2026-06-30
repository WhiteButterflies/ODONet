import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
# dataset_name = 'nfs' # choosen from 'uav', 'nfs', 'lasot_extension_subset', 'lasot', 'lasot_lang', 'otb99_lang', 'tnl2k'
# dataset_name = 'lasot'
dataset_name = 'lasot_extension_subset'
# dataset_name = 'vasttrack'
# dataset_name = 'otb'
# dataset_name = 'tnl2k'
# dataset_name = 'uav'

if dataset_name == 'otb':
    trackers.extend(trackerlist(name='odonet_v1', parameter_name='odonet_b384', dataset_name=dataset_name,
                                run_ids=None, display_name='odonet_b384'))
elif dataset_name == 'lasot':
    trackers.extend(trackerlist(name='odonet_v1', parameter_name='odonet_b384', dataset_name=dataset_name,
                                run_ids=None, display_name='odonet_b384'))
elif dataset_name =="lasot_extension_subset":
    trackers.extend(trackerlist(name='odonet_v1', parameter_name='odonet_b384', dataset_name=dataset_name,
                                run_ids=None, display_name='odonet_b384'))

elif dataset_name =='tnl2k':
    
    trackers.extend(trackerlist(name='odonet_v1', parameter_name='odonet_b384', dataset_name=dataset_name,
                                run_ids=None, display_name='odonet_b384'))
elif dataset_name =='uav':
    trackers.extend(trackerlist(name='odonet_v1', parameter_name='odonet_b384', dataset_name=dataset_name,
                                run_ids=None, display_name='odonet_b384'))
elif dataset_name =='vasttrack':
    trackers.extend(trackerlist(name='odonet_v1', parameter_name='odonet_b384', dataset_name=dataset_name,
                                run_ids=None, display_name='odonet_b384'))


dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
              force_evaluation=True)