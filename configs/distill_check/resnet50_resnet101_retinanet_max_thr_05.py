_base_ = [
    '../retinanet/retinanet_r50_fpn_2x_coco.py'
]

model = dict(
    type='Distilling_Max_Thr',

    distill = dict(
        teacher_cfg='./configs/retinanet/retinanet_r101_fpn_1x_coco.py',
        teacher_model_path='/lustre/S/duzhixing/workspace/model/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth',
        
        distill_warm_step=500,
        distill_feat_weight=0.005,
        # distill_cls_weight= 0,
        # distill_bbox_weight= 0,
        mask_thr=0.05,
        stu_feature_adap=dict(
            type='ADAP',
            in_channels=256,
            out_channels=256,
            num=5,
            kernel=3
        ),
    )
)

seed=520
# find_unused_parameters=True
