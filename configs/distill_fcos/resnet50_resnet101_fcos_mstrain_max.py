_base_ = [
    '../fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py'
]

model = dict(
    type='Distilling_Fcos',
    
    distill = dict(
        teacher_cfg='./configs/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py',
        teacher_model_path='/workspace/S/duzhixing/workspace/model/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth',
        
        distill_warm_step=500,
        distill_feat_weight=0.01,
        distill_cls_weight=0.05,
        # distill_bbox_weight=0.002,
        
        stu_feature_adap=dict(
            type='ADAP',
            in_channels=256,
            out_channels=256,
            num=5,
            kernel=3
        ),
    )
)

# # optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[16, 22])
# total_epochs = 24


seed=520

find_unused_parameters=True



