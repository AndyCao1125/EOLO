from .yolov1 import YOLOv1
from .yolov2 import YOLOv2
from .yolov3 import YOLOv3
from .yolov4 import YOLOv4
from .yolo_tiny import YOLOTiny
from .yolo_nano import YOLONano
from .eyolo_tiny import EYOLOTiny


# build YOLO detector
def build_model(args, cfg, device, num_classes=80, trainable=False):
    
    if args.model == 'yolov1':
        print('Build YOLOv1 ...')
        model = YOLOv1(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov2':
        print('Build YOLOv2 ...')
        model = YOLOv2(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'E-yolo-tiny':
        print(f'Build {args.model} ...')
        model = EYOLOTiny(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample,
                        fusion_method=args.fusion_method,
                        )
    elif args.model == 'yolov3':
        print('Build YOLOv3 ...')
        model = YOLOv3(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample,
                        in_channel = args.in_channel,)
    elif args.model == 'yolov3_spp':
        print('Build YOLOv3 with SPP ...')
        model = YOLOv3(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov3_de':
        print('Build YOLOv3 with DilatedEncoder ...')
        model = YOLOv3(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov4':
        print('Build YOLOv4 ...')
        model = YOLOv4(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolo_tiny':
        print('Build YOLO-Tiny ...')
        model = YOLOTiny(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample,
                        in_channel = args.in_channel)
    elif args.model == 'yolo_nano':
        print('Build YOLO-Nano ...')
        model = YOLONano(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
        
    
    return model
