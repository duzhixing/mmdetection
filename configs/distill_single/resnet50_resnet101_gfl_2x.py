_base_ = [
    '../gfl/gfl_r50_fpn_mstrain_2x_coco.py'
]

model = dict(
    type='Distilling_Max',

    distill = dict(
        teacher_cfg='./configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py',
        teacher_model_path='/lustre/S/duzhixing//workspace/model/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth',
        
        distill_warm_step=500,
        distill_feat_weight=0.1,
        distill_cls_weight=0.05,
        distill_bbox_weight=0.002,
        
        stu_feature_adap=dict(
            type='ADAP',
            in_channels=256,
            out_channels=256,
            num=5,
            kernel=3
        ),
    )
)

# load_from="/lustre/S/duzhixing/workspace/mmdet/result_coco/norelu/resnet50_resnet101_gfl_single_score_decay_coco/epoch_12.pth"

seed=520
find_unused_parameters=True
