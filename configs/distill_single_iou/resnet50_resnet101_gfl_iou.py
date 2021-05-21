_base_ = [
    '../gfl/gfl_r50_fpn_1x_coco.py'
]

model = dict(
    type='Distilling_Single_IOU',

    distill = dict(
        teacher_cfg='./configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py',
        teacher_model_path='/lustre/S/duzhixing//workspace/model/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth',
        
        distill_warm_step=500,
        distill_feat_weight=0.1,
        # distill_cls_weight=0.02,
        # distill_bbox_weight=0.2, 
        anchor_num=1,
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
