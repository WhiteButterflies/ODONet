class Train_Prev_Processing(BaseProcessing):
    """ The processing class used for training SeqTrack. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', multi_modal_language=False, settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.multi_modal_language = multi_modal_language

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=True) #搜索区域旋转

        s_list = ['template', 'search']

        for s in s_list:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data
            if s =='search':
                '''no jittered anno center pos'''
                crops_center, boxes_center, att_mask_center, mask_crops_center,boxes_extract_center = prutils.jittered_center_crop(data[s + '_images'], data[s + '_anno'],
                                                                                  data[s + '_anno'], self.search_area_factor[s],
                                                                                  self.output_sz[s], masks=data[s + '_masks'])
                '''added LK 稀疏光流'''
                lk_center_points = prutils.get_sparse_flow_keypoints_in_boxes(crops_center,boxes_center,max_corners=150,max_points_per_box=5)

                # Apply transforms 不会修改bbbox 可以视为上一帧的中心坐标 取消旋转
                data[s + '_center_images'], data[s + '_center_anno'], data[s + '_center_masks'] = self.transform[s](
                    image=crops_center, bbox=boxes_center, mask=mask_crops_center, joint=False, new_roll=False)

                # added by liu for boxes：裁切后的GT  box_extract:中心裁切的坐标，并非GT
                crops, boxes, att_mask, mask_crops, boxes_extract = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                                                 data[s + '_anno'], self.search_area_factor[s],
                                                                                                 self.output_sz[s], masks=data[s + '_masks'])

                '''apply LK Transform'''
                data[s + '_LK_points'] = prutils.transform_keypoints_with_boxes(lk_center_points,boxes_extract_center,boxes_extract)

                # Apply transforms 不会修改bbbox 暂时取消旋转
                data[s + '_images'], data[s + '_anno'], data[s + '_masks'] = self.transform[s](
                    image=crops, bbox=boxes, mask=mask_crops, joint=False, new_roll=True)
                data[s+'_prev_anno'] = boxes_extract

                original_boxes ,new_boxes,points = torch.cat([item.unsqueeze(0) for item in boxes_extract], dim=0), torch.cat([item.unsqueeze(0) for item in data[s + '_anno']], dim=0), torch.cat([item.unsqueeze(0) for item in data[s + '_LK_points']], dim=0)

                flipped = prutils.compare_tensor_lists(boxes,data[s + '_anno'])
                flipped = torch.tensor(flipped,dtype=torch.bool, device=original_boxes.device)
                offset,is_flipped = prutils.compute_point_offsets(original_boxes ,new_boxes,points,is_flipped=flipped)
                data[s + '_LK_points_offset'], data[s + 'is_flipped'] =torch.unbind(offset,dim=0), torch.unbind(is_flipped,dim=0)
            elif s =='template':
                #added by liu for boxes：裁切后的GT  box_extract:中心裁切的坐标，并非GT
                crops, boxes, att_mask, mask_crops,boxes_extract = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                                  data[s + '_anno'], self.search_area_factor[s],
                                                                                  self.output_sz[s], masks=data[s + '_masks'])
                # Apply transforms 不会修改bbbox
                data[s + '_images'], data[s + '_anno'], data[s + '_masks'] = self.transform[s](
                    image=crops, bbox=boxes, mask=mask_crops, joint=False)
            else:
                raise ValueError(f"Unexpected frame type: {s}")



            '''added by liu for 记录中心裁切坐标（非GT）'''
            # data[s+'_extract_anno'] = boxes_extract

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Process NLP  grounding_nl_token_ids, grounding_nl_token_masks
        if self.multi_modal_language:
            if 'nl_token_ids' in data:
                data['nl_token_ids'] = torch.tensor(data['nl_token_ids'])
                data['nl_token_masks'] = torch.tensor(data['nl_token_masks'])
            else:
                data['nl_token_ids'] = torch.zeros(self.settings.max_query_len, dtype=torch.long)
                data['nl_token_masks'] = torch.zeros(self.settings.max_query_len, dtype=torch.long)
            # if (data['nl_token_masks'] == 0).all():
            #     data['valid'] = False
            #     print('nl_token_masks is error')
            #     return data
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
