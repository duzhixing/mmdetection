_base_ = [
    '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
]

model = dict(
    type='Distilling_Two',

    distill = dict(
        teacher_cfg='./configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py',
        teacher_model_path='/lustre/S/duzhixing/workspace/model/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth',
        
        distill_warm_step=500,
        distill_feat_weight=0.002,
        distill_cls_weight=0.05,
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

# load_from="/workspace/S/duzhixing/workspace/model/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"


seed=520
find_unused_parameters=True
