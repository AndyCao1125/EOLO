## eolo-tiny underexposure
python test_eyolo.py -d voc \
               --cuda \
               -m E-yolo-tiny  \
               --weight weights/voc/E-yolo-tiny/Eyolov-tiny_VOC_Underexposure_0.2_random42_1gpu_32bs_50epoch_noalign_AFNet_symetric_fusion_maxavgcat_temp_final/E-yolo-tiny_50_60.61.pth \
               --img_size 320 \
               --root path/to/dataset/ \
               --save_name Eyolov-tiny_VOC_Underexposure_0.2_random42_results\
               --data_type Exposure_Event\
               --fusion_method AFNet_symetric_fusion_maxavgcat\
               --exposure_factor Underexposure_0.2_random42 \
               --visual_threshold 0.4