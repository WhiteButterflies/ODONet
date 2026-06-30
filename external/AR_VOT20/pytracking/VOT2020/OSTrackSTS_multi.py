import sys
import os
filepath = os.path.abspath(__file__)
AR_dir = os.path.join(os.path.dirname(filepath), "..", "..")
sys.path.append(AR_dir)
stark_dir = os.path.join(AR_dir, "..", "..")
stark_lib_dir = os.path.join(stark_dir, "lib")
sys.path.append(stark_dir)
sys.path.append(stark_lib_dir)

# from pytracking.VOT2020.stark_alpha_seg_class import run_vot_exp
from stark_alpha_seg_class import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
run_vot_exp('ostrack', 'vitb_256_mae_ce_32x4_ep400_topk1.0', 'baseline_plus_got_lasot', 0.40, VIS=False, trt='false')
