import yaml

import torch
from modelling.vq_model import VQ_models



def build_tokenizer(vq_config,
                    vq_ckpt):
    
    with open(vq_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config_name = vq_config.split('/')[-2]
    exp_index = int(config_name.split('-')[0][3:])
    if exp_index < 37 and config['vq_model'].startswith('SoftVQ'):
        config['vq_model'] = config['vq_model'] + '-Legacy'
    
    enc_img_res = config.get('enc_img_res', False)
    enc_img_align = config.get('enc_img_align', 'global')
    num_codebooks = config.get('num_codebooks', 1)
    enc_img_res_v2 = config.get('enc_img_res_v2', False)
    enc_img_layer_indices = config.get('enc_img_layer_indices', [])
    dec_img_res = config.get('dec_img_res', False)
    dec_img_layer_indices = config.get('dec_img_layer_indices', [])
    
    # residual
    residual = config.get('residual', False)
    repa = config.get('repa', False)
    repa_model = config.get('repa_model', 'None')
    repa_patch_size = config.get('repa_patch_size', 16)
    repa_proj_dim = config.get('repa_proj_dim', 1024)
    repa_align = config.get('repa_align', 'global')
    

    vae = VQ_models[config['vq_model']](
        image_size=config['image_size'],
        codebook_size=config['codebook_size'],
        codebook_embed_dim=config['codebook_embed_dim'],
        enc_type=config['enc_type'],
        encoder_model=config['encoder_model'],
        dec_type=config['dec_type'],
        decoder_model=config['decoder_model'],
        num_latent_tokens=config['num_latent_tokens'],
        enc_tuning_method=config['encoder_tuning_method'],
        dec_tuning_method=config['decoder_tuning_method'],
        enc_patch_size=config['encoder_patch_size'],
        dec_patch_size=config['decoder_patch_size'],
        tau=config['tau'] if 'tau' in config else 1.0,
        enc_img_align=enc_img_align,
        enc_img_res=enc_img_res,
        enc_img_res_v2=enc_img_res_v2,
        enc_img_layer_indices=enc_img_layer_indices,
        num_codebooks=num_codebooks,
        dec_img_res=dec_img_res,
        dec_img_layer_indices=dec_img_layer_indices,
        residual=residual,
        repa=repa,
        repa_model=repa_model,
        repa_patch_size=repa_patch_size,
        repa_proj_dim=repa_proj_dim,
        repa_align=repa_align
    )

    # vq_model.to(device)
    # vq_model.eval()
    checkpoint = torch.load(vq_ckpt, map_location="cpu")
    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    keys = vae.load_state_dict(model_weight, strict=False)
    print(keys)
    

    vq_1d = False
    if config_name == 'exp000-cnn_llamagen_vq16_v8196d16':
        vq_mean, vq_std = -0.00193, 0.24993
        dit_input_size = 16
    elif config_name == 'exp001-cnn_llamagen_vq16_v16384d8':
        vq_mean, vq_std = 0.023302, 0.352785
        dit_input_size = 16
    elif config_name == 'exp002-cnn_llamagen_kl16_d16':
        vq_mean, vq_std = 0.004139, 1.026751
        dit_input_size = 16
    elif config_name == 'exp006-cnn_llamagen_vq16_v16384d8_dino_disc_w0.02':
        vq_mean, vq_std = 0.018639, 0.353062
        dit_input_size = 16
    elif config_name == 'exp005-cnn_llamagen_vq16_v16384d8_dino_disc_var':
        vq_mean, vq_std = 0.019001, 0.353043
        dit_input_size = 16
    elif config_name == 'exp019-vit_dinos_full_vq256_v16384d16':
        vq_mean, vq_std = -0.000197, 0.25
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp021-vit_dinos_full_kl256_d16':
        vq_mean, vq_std = 0.664058, 6.377685
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp024-vit_dinos_full_softvq256_v16384d16':
        vq_mean, vq_std = 0.000229, 0.151825
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp026-vit_dinos_full_softvq128_v16384d32':
        vq_mean, vq_std = 0.000605, 0.07915
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp029-vit_dinos_full_softvq256_v16384d16_repa':
        vq_mean, vq_std = 0.000016, 0.151583
        dit_input_size = config['num_latent_tokens']
        vq_1d = True 
    elif config_name == 'exp020-vit_dinos_full_vq128_v16384d32':
        vq_mean, vq_std = 0.000103, 0.176777
        dit_input_size = config['num_latent_tokens']
        vq_1d = True 
    elif config_name == 'exp021-vit_dinos_full_vq32_v16384d128':
        vq_mean, vq_std = -0.000456, 0.088388
        dit_input_size = config['num_latent_tokens']
        vq_1d = True 
    elif config_name == 'exp022-vit_dinos_full_kl128_d32':
        vq_mean, vq_std = -0.061042, 6.328941
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp023-vit_dinos_full_kl32_d128':
        vq_mean, vq_std = -0.128278, 2.459388
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp027-vit_dinos_full_softvq64_v16384d64':
        vq_mean, vq_std = 0.000221, 0.038764
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp028-vit_dinos_full_softvq32_v16384d128':
        vq_mean, vq_std = 0.000859, 0.036317
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp033-vit_dinob_full_softvq32_v16384d128':
        vq_mean, vq_std = -0.000445, 0.035826
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp034-vit_dinob_full_softvq64_v16384d64':
        vq_mean, vq_std = 0.00019, 0.032813
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp038-vit_dinos_full_softvqleg64_v16384d64':
        vq_mean, vq_std = 0.0003863436, 0.035706423
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp040-vit_dinos_full_softvq64_v16384d64':
        vq_mean, vq_std = 0.0017585702, 0.028160747
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp041-vit_dinos_full_softvq64_v16384d64_novqloss':
        vq_mean, vq_std = -0.00015628517, 0.047457933
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp042-vit_dinos_full_softvq64_v16384d64_nocomloss':
        vq_mean, vq_std = -0.00020806103, 0.03576384
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp043-vit_dinos_full_softvq64_v16384d64_noboth':
        vq_mean, vq_std = 0.00028580736, 0.028316298
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp045-vit_dinos_full_softvq64_v16384d64_nocomloss_repa':
        vq_mean, vq_std = 0.0004393749, 0.037167482
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp046-vit_dinos_full_softvq64_v16384d64_nocomloss_repa_avg1d':
        vq_mean, vq_std = -6.9363334e-05, 0.029894955
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp047-vit_dinos_full_softvq64_v16384d64_nocomloss_repa_avg1d_shffule':
        vq_mean, vq_std = 0.0005302097, 0.0355743
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp049-vit_dinos_full_softvq64_v16384d64_nocomloss_encimgres_avg1d':
        vq_mean, vq_std = 0.0020379436, 0.037771754
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp050-vit_dinos_full_softvq64_v16384d64_nocomloss_encimgres_avg1d_shffule':
        vq_mean, vq_std = -0.00036217086, 0.028036779
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp051-vit_dinos_full_softvq64_v16384d64_nocomloss_enc_repa_avg1d':
        vq_mean, vq_std = -8.65E-05, 0.028589882
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp052-vit_dinos_full_softvq64_n4v16384d64_nocomloss':
        vq_mean, vq_std = -0.000567266, 0.03159521
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp055-vit_dinos_full_softvq64_n4v8192d64_enc_repa_avg1d':
        vq_mean, vq_std = -0.001577422, 0.029091477
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp075-vit_dinob_full_softvq32_n4v8192d128':
        # TODO: temporary values
        vq_mean, vq_std = 2.3551744e-05, 0.011036889
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
        checkpoint = torch.load(vq_ckpt, map_location="cpu")
        model_weight = checkpoint["model"]
        keys = vae.load_state_dict(model_weight, strict=False)
        print(keys)
    elif config_name == 'exp074-vit_dinob_full_softvq64_n4v8192d64':
        # TODO: temporary values
        vq_mean, vq_std = 0.001376948, 0.027370881
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
        checkpoint = torch.load(vq_ckpt, map_location="cpu")
        model_weight = checkpoint["model"]
        keys = vae.load_state_dict(model_weight, strict=False)
        print(keys)
    elif config_name == 'exp071-vit_dinos_full_softvq64_n4v8192d64':
        # TODO: temporary values
        vq_mean, vq_std = -0.0007923665, 0.028401684
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp074-vit_dinob_full_softvq64_n4v8192d64_global':
        vq_mean, vq_std = -0.00014397933, 0.03498383
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp059-vit_dinos_full_softvq64_n64v256d64':
        vq_mean, vq_std = -0.00039463455, 0.017090624
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp074-vit_dinob_full_softvq64_n4v8192d64_enc':
        # 0.5675
        vq_mean, vq_std = 0.0005967469, 0.029659314
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
        checkpoint = torch.load(vq_ckpt, map_location="cpu")
        model_weight = checkpoint["model"]
        keys = vae.load_state_dict(model_weight, strict=False)
        print(keys)
    elif config_name == 'exp074-vit_dinob_full_softvq64_n4v8192d64_enc_repa_global':
        # 0.5675
        vq_mean, vq_std = -0.00012682067, 0.032536615
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
        checkpoint = torch.load(vq_ckpt, map_location="cpu")
        model_weight = checkpoint["model"]
        keys = vae.load_state_dict(model_weight, strict=False)
        print(keys)
    elif config_name == 'exp074-vit_dinob_full_softvq64_n4v8192d64_encv2_repa_global':
        # 0.703220
        vq_mean, vq_std =  -0.0009537882, 0.030718027
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
        checkpoint = torch.load(vq_ckpt, map_location="cpu")
        model_weight = checkpoint["model"]
        keys = vae.load_state_dict(model_weight, strict=False)
        print(keys)
    elif config_name == 'exp042-vit_dinob_full_softvq64_n4v8192d64_encv2deep_repa_global':
        vq_mean, vq_std =  -0.0015510367, 0.031602673
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
        checkpoint = torch.load(vq_ckpt, map_location="cpu")
        model_weight = checkpoint["model"]
        keys = vae.load_state_dict(model_weight, strict=False)
        print(keys)
    elif config_name == 'exp074-vit_dinob_full_softvq64_n4v8192d64_encv2_dec_repa_global':
        vq_mean, vq_std =   -2.61e-05, 0.027315257
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
        checkpoint = torch.load(vq_ckpt, map_location="cpu")
        model_weight = checkpoint["model"]
        keys = vae.load_state_dict(model_weight, strict=False)
        print(keys)
    elif config_name == 'exp074-vit_dinob_full_softvq64_n4v8192d64_encv2deep_decdeep_repa_global':
        vq_mean, vq_std =   0.0015506828, 0.028120134
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
        checkpoint = torch.load(vq_ckpt, map_location="cpu")
        model_weight = checkpoint["model"]
        keys = vae.load_state_dict(model_weight, strict=False)
        print(keys)
    elif config_name == 'exp074-vit_dinob_full_softvq64_n4v8192d64_encv2_repa_repeat':
        vq_mean, vq_std =  -0.0012404232, 0.029286876
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
        checkpoint = torch.load(vq_ckpt, map_location="cpu")
        model_weight = checkpoint["model"]
        keys = vae.load_state_dict(model_weight, strict=False)
        print(keys)
        
        
        
    elif config_name == 'exp090-vit_dinos_full_softvq64_n4v8192d32':
        vq_mean, vq_std = -0.0009900038, 0.06745294
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp091-vit_dinob_full_softvq64_n4v8192d32':
        vq_mean, vq_std = -0.003192751, 0.07235257
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp092-vit_dinol_full_softvq64_n4v8192d32':
        vq_mean, vq_std = 0.0009714666, 0.07259091
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp093-vit_dinobl_full_softvq64_n4v8192d32':
        vq_mean, vq_std = -0.00077168207, 0.07157949
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp094-vit_dinos_full_softvq32_n4v8192d64':
        vq_mean, vq_std = 0.00020753492, 0.029780312
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp095-vit_dinob_full_softvq32_n4v8192d64':
        vq_mean, vq_std = -0.0001770356, 0.029009325
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp096-vit_dinol_full_softvq32_n4v8192d64':
        vq_mean, vq_std = 0.00060894643, 0.029030401
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp097-vit_dinobl_full_softvq32_n4v8192d64':
        vq_mean, vq_std = 0.0010870857, 0.028672216
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp112-vit_dinobl_full_softvq64_n4v8192d32_512':
        vq_mean, vq_std = -0.0021498345, 0.073663734
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp113-vit_dinol_full_softvq32_n4v8192d64_512':
        vq_mean, vq_std = -0.0011922447, 0.028908715
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    
    elif config_name == 'exp116-vit_dinos_full_vq64_n4v8192d32':
        vq_mean, vq_std = 0.000893734, 0.17677507
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp117-vit_dinos_full_vq32_n4v8192d64':
        vq_mean, vq_std = 0.0010069222, 0.124996305
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp118-vit_dinos_full_vq128_n4v8192d16':
        vq_mean, vq_std = -0.000997954, 0.24999803
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp119-vit_dinos_full_vq256_n4v8192d8':
        vq_mean, vq_std = -0.004624111, 0.35352302
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp121-vit_dinos_full_ae128_n4v8192d16':
        vq_mean, vq_std = 0.015747057, 2.7646668
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp122-vit_dinos_full_ae64_n4v8192d32':
        vq_mean, vq_std = -8.47E-05, 5.659972
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp123-vit_dinos_full_ae32_n4v8192d64':
        vq_mean, vq_std = -0.11781523, 3.691254
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp125-vit_dinos_full_softvqres64_v8192d32':
        vq_mean, vq_std = -0.0011922447, 0.028908715
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp126-vit_dinos_full_softvqres64_n4v8192d32':
        vq_mean, vq_std = -0.0011922447, 0.028908715
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp127-vit_clipb_full_softvq64_n4v8192d32':
        vq_mean, vq_std = -0.003618149, 0.06822964
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp128-vit_eva02b_full_softvq64_n4v8192d32':
        vq_mean, vq_std = -0.003901229,  0.07412359
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp129-vit_dinos_full_softvq128_n4v8192d16':
        vq_mean, vq_std = None
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp130-vit_dinos_full_softvq256_n4v8192d8':
        vq_mean, vq_std = None
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp131-vit_dinob_full_softvq64_n4v8192d32_scratch':
        vq_mean, vq_std = None
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    elif config_name == 'exp132-vit_dinob_full_softvq64_n4v8192d32_scratch_no_align':
        vq_mean, vq_std = None
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    else:
        raise NotImplementedError
    
    
    return vae, config['vq_model'], config['codebook_embed_dim'], dit_input_size, vq_mean, vq_std, vq_1d