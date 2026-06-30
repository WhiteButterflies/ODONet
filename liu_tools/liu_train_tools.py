import os
import numpy as np
from collections import OrderedDict
def _check_resultsFile_exist(seq: dict, tracker: dict):
    """Checks if the results file exists."""
    if seq["dataset"] in ['trackingnet', 'got10k', 'lasot', 'lasot_extension_subset', 'otb', 'uav', 'nfs', 'tnl2k']:
        base_results_path = os.path.join(tracker["results_dir"], seq["dataset"], seq["name"])
    else:
        base_results_path = os.path.join(tracker["results_dir"], seq["name"])

    def check_bb(file):
        return os.path.exists(file)

    def check_time(file):
        return os.path.exists(file)

    def check_score(file):
        return os.path.exists(file)
    # Single-object mode
    bbox_file = '{}.txt'.format(base_results_path)
    if not check_bb(bbox_file):
        return False
    return True

def _save_tracker_output(seq: dict, tracker: dict, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker["results_dir"]):
        print("create tracking result dir:", tracker["results_dir"])
        os.makedirs(tracker["results_dir"])
    if seq["dataset"] in ['trackingnet', 'got10k', 'lasot', 'lasot_extension_subset', 'otb', 'uav', 'nfs', 'tnl2k']:
        if not os.path.exists(os.path.join(tracker["results_dir"], seq["dataset"])):
            os.makedirs(os.path.join(tracker["results_dir"], seq["dataset"]))
    '''2021.1.5 create new folder for these three datasets'''
    if seq["dataset"] in ['trackingnet', 'got10k', 'lasot', 'lasot_extension_subset', 'otb', 'uav', 'nfs', 'tnl2k']:
        base_results_path = os.path.join(tracker["results_dir"], seq["dataset"], seq["name"])
    else:
        base_results_path = os.path.join(tracker["results_dir"], seq["name"])

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)

