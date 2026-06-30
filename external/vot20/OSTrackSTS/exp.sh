vot evaluate --workspace . OSTrackSTS_base OSTrackSTS_multi OSTrackSTS_multi_dconv  OSTrackSTS_sam_multi_dconv
vot analysis --workspace . OSTrackSTS OSTrackSTS_multi OSTrackSTS_multi_dconv --format html
vot analysis --workspace . ODONet --format html
vot analysis --workspace . ODONet_B384 --format html
vot analysis --workspace . ODONet_B384_old --format html
