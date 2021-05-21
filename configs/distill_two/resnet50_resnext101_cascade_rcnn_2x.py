_base_ = [
    '../cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco.py'
]

model = dict(
    type='Distilling_Two',

    distill = dict(
        teacher_cfg='./configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py',
        teacher_model_path='/workspace/S/duzhixing/workspace/model/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth',
        
        distill_warm_step=500,
        distill_feat_weight=0.0015,
        distill_cls_weight=0.1,
        # distill_bbox_weight=0.5,
        
        stu_feature_adap=dict(
            type='ADAP',
            in_channels=256,
            out_channels=256,
            num=5,
            kernel=3
        ),
    )
)

# load_from="/lustre/S/duzhixing/workspace/model/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"


seed=520
# find_unused_parameters=True
