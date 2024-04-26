import numpy as np
import cv2
import os
import copy
import matplotlib.pyplot as plt
from PIL import Image
os.environ["OPENCV_IO_ENABLE_OPENEXR"]='1'
from tqdm import tqdm

## Global variable
OFFSET = 3      ## optical flow
THRESHOLD= 20   ## event camera threshold
SCALE = 0.3     ## gradient scale
EVENT_MASK = 50 ## mask some unimportant event
# EXPOSE = 1.0    ## exposure factor    1.0 for noexposure
EXPOSE = 0.2    ## exposure factor    0.2 for underexposure
# EXPOSE = 5.0    ## exposure factor    5.0 for overexposure

np.random.seed(42)  ## Random seed

def exposure_img(img_path, hdr_path, exposure_factor):
    '''
    This code reads the image using OpenCV and converts it to the HSV color space, 
    which separates the color information (Hue and Saturation) from the intensity 
    information (Value). It then adjusts the exposure by directly scaling the pixel 
    intensities in the Value channel according to the provided exposure factor. 
    An exposure factor between [0.0, 1.0] will create an underexposed image, while 
    a factor greater than 1.0 will create an overexposed image. 
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_image = hsv_image.astype('float32')
    hsv_image[..., 2] = hsv_image[..., 2] * exposure_factor    ## over/under-exposure means the lightness of the image is extremely high/low

    cv2.imwrite(hdr_path, hsv_image)


def hdr2rgb(exr_path, clip_uint8=True):
    '''
    This code transforms the HDR image to RGB image for visualization.
    '''
    hsv_img = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  ## EXR to HSV

    if clip_uint8:
        hsv_img[..., 2] = np.clip(hsv_img[..., 2], 0, 255)
        hsv_img = hsv_img.astype('uint8')

    # Convert the image from HSV to RGB
    rgb_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)  ## the range of pixel value is not in range(0,255)

    return rgb_image

def hdr2event(hdr_path, show_exposure=False, display=False, random_optical_flow=False):
    '''
    This code is the core part of the proposed randomized optical flow-based event synthesis.
    Only a single RGB/HDR image is required to generate the corresponding event frames.

    Ref: https://arxiv.org/pdf/2309.09297
    '''

    def normalize_image(image):
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image = (image - min_val) * (255 / (max_val - min_val + 1e-3))
        return normalized_image.astype(np.uint8)

    def compute_gradient(image):

        # Normalize the image
        normalized_image = normalize_image(image)  ## Norm the image

        # Convert to grayscale
        grayscale_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY) ## transform HDR image into grayscale

        # Apply Sobel operator
        sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)

        return sobel_x, sobel_y

    def display_event(input_event):
        event_frame = input_event.copy()
        positive_pixel = event_frame[..., 0] > event_frame[..., 1]
        negetive_pixel = event_frame[..., 0] < event_frame[..., 1]

        # event_frame[positive_pixel] = [255, 0, 0]
        # event_frame[negetive_pixel] = [0, 0, 255]
        event_frame[positive_pixel] = [255, 0]
        event_frame[negetive_pixel] = [0, 255]

        event_frame = np.concatenate( (event_frame, np.zeros((event_frame.shape[0],event_frame.shape[1],2)) ),axis=-1) ## stack 0 to channel 3
        
        ## positive event is red, negative event is blue
        event_frame = np.transpose(event_frame, (0, 1, 2))[:, :, [0, 2, 1]]   ##RGB format
        return event_frame.astype(np.uint8)

    if show_exposure:    ## get the exposure image with uint8
        clip_exposure_img = hdr2rgb(hdr_path)

    else:
        clip_exposure_img = None
    
    ## read EXR by default
    row_hdr_img = cv2.imread(hdr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    hdr_rgb_image = cv2.cvtColor(row_hdr_img, cv2.COLOR_HSV2RGB)  ## Change HSV to RGB & the range of pixel value may not in range(0,255)

    ## read JPG
    # row_exp_img = cv2.imread(hdr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # hdr_rgb_image = cv2.cvtColor(row_exp_img, cv2.COLOR_BGR2RGB)  ## Change BGR to RGB & the range of pixel value is in range(0,255)

    norm_img = normalize_image(hdr_rgb_image)
    grad_x, grad_y = compute_gradient(norm_img)
    grad = np.stack([grad_x, grad_y], axis=-1) * SCALE

    if random_optical_flow:
        theta = np.random.uniform(-np.pi, np.pi)
        v = np.array([np.cos(theta), np.sin(theta)]) * OFFSET * np.sqrt(2)  ## randomized optical flow
    else:
        v = np.ones_like(grad) * OFFSET          ## optical flow
    increment = np.sum(grad * v, axis=-1)    ## delta_light (H,W)
    o_increment = copy.deepcopy(increment)

    ## mask some unimportant event
    mask = (increment < EVENT_MASK) * (increment > -EVENT_MASK)
    increment[mask] = 0

    event = np.full((norm_img.shape[0],norm_img.shape[1], 2), 0, int)  ## event has shape (H,W,2)
    px, py = np.where(increment > 0)
    nx, ny = np.where(increment < 0)

    event[px, py, 0] += (o_increment[px, py] / THRESHOLD).astype(int)   ## count the positive event
    event[nx, ny, 1] += (-o_increment[nx, ny] / THRESHOLD).astype(int)  ## count the negative event


    if display:
        event_based_frame = display_event(event)
    else:
        event_based_frame = None

    return event, event_based_frame, clip_exposure_img


if __name__ == "__main__":

    '''
    This is an example code of generating E-VOC datasets.

    The resulting dataset will have the following data structure:

    VOC2007
    |---Event                      ## Raw Event (.npy)
       |---{event_type}, e.g.,'Underexposure_0.2_random42'
           |---XXXX.npy
           |...
    |---EventFrameImages           ## Event Frame (.jpg)
        |---{event_type}
           |---XXXX.jpg
           |...
    |---ExposureImages             ## Exposure RGB image for visulization (.jpg), clip into [0,255] from HDR image
        |---{event_type}
           |---XXXX.jpg
           |...
    |---HDRImages                  ## Exposure Images (.exr)
        |---{event_type}
           |---XXXX.exr
           |...
    |---Annotations                
    |---JPEGImages
    |---ImageSets
    |---SegmentationClass
    |---SegmentationObject

    where the Event, EventFrameImages, ExposureImages and HDRImages are newly generated.
    '''


    ##-----------------VOC dataset-----------------##
    voc_dir = '/home/lsf_node01/dataset/VOC_dataset/VOCdevkit/VOC2007'
    # voc_dir = '/home/lsf_node01/dataset/VOC_dataset/VOCdevkit/VOC2012'

    jpg_list = os.listdir(os.path.join(voc_dir, 'JPEGImages'))
    jpg_list.sort()
    random_optical_flow = True

    if isinstance(EXPOSE,float):
        if EXPOSE < 1.0:
            if random_optical_flow:
                event_type = 'Underexposure_' + str(EXPOSE) + '_random42'
            else:
                event_type = 'Underexposure_' + str(EXPOSE)
        elif EXPOSE >= 1.0:
            if random_optical_flow:
                event_type = 'Overexposure_' + str(EXPOSE) + '_random42'
            else:
                event_type = 'Overexposure_' + str(EXPOSE) 
    
    print(f'--------Start to generate {event_type} dataset in {voc_dir}--------')

    exposure_jpg_dir = os.path.join(voc_dir, 'ExposureImages', event_type)
    hdr_dir = os.path.join(voc_dir, 'HDRImages', event_type)
    event_dir = os.path.join(voc_dir, 'Event', event_type)
    event_frame_dir = os.path.join(voc_dir, 'EventFrameImages', event_type)

    for dir in [hdr_dir, event_dir, event_frame_dir, exposure_jpg_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for img in tqdm(jpg_list):
        img_path = os.path.join(voc_dir, 'JPEGImages', img)

        hdr_path = os.path.join(hdr_dir, os.path.splitext(os.path.basename(img))[0] + '.exr')
        event_path = os.path.join(event_dir, os.path.splitext(os.path.basename(img))[0] + '.npy')
        event_frame_path = os.path.join(event_frame_dir, os.path.splitext(os.path.basename(img))[0] + '.jpg')
        exposure_img_path = os.path.join(exposure_jpg_dir, img)

        exposure_img(img_path, hdr_path, exposure_factor=EXPOSE)
        event, event_based_frame, clip_exposure_img = hdr2event(hdr_path, show_exposure=True, display=True, random_optical_flow=random_optical_flow)
        np.save(event_path, event)
        cv2.imwrite(event_frame_path, cv2.cvtColor(event_based_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(exposure_img_path, cv2.cvtColor(clip_exposure_img, cv2.COLOR_RGB2BGR))

    


   