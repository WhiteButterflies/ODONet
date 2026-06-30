from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data2/lqh/data/got10k_lmdb'
    settings.got10k_path = '/data2/lqh/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/data2/lqh/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data2/lqh/data/lasot_lmdb'
    settings.lasot_path = '/data2/lqh/data/lasot'
    settings.lasotlang_path = '/data2/lqh/data/lasot'
    settings.network_path = '/data2/lqh/workspace_pycharm/ODONet/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data2/lqh/data/nfs'
    settings.otb_path = '/data2/lqh/data/OTB2015'
    settings.otblang_path = '/data2/lqh/data/otb_lang'
    settings.prj_dir = '/data2/lqh/workspace_pycharm/ODONet'
    settings.result_plot_path = '/data2/lqh/workspace_pycharm/ODONet/test/result_plots'
    settings.results_path = '/data2/lqh/workspace_pycharm/ODONet/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data2/lqh/workspace_pycharm/ODONet'
    settings.segmentation_path = '/data2/lqh/workspace_pycharm/ODONet/test/segmentation_results'
    settings.tc128_path = '/data2/lqh/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data2/lqh/data/TNL2K'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data2/lqh/data/trackingnet'
    settings.uav_path = '/data2/lqh/data/UAV123'
    settings.vot_path = '/data2/lqh/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

