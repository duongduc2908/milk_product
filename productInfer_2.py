import cv2 
import numpy as np
import os 
from PIL import Image

from mmocr1.getmodel import MMOCR

from pan.predict import Pytorch_model

from textClassify.product_classifier_infer import ClassifierInfer

from CRAFTpytorch.inference import load_model, extract_wordbox

from textRecognition.inferer import TextRecogInferer, default_args, text_recog


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# get 4 points of rectangle from poly
def get_box_from_poly(pts):
    pts = pts.reshape((-1,2)).astype(int)
    x,y,w,h = cv2.boundingRect(pts)
    area = w*h
    return np.array([x,y,x+w,y+h]), area

def ensemble(pts1, pts2):
    # take pts2 first then pts1
    iou_thres = 0.5
    pts = []
    
    # box2 = []
    # for poly in pts2:
    #     box2.append(get_box_from_poly(poly))
    
    # rm = []
    # for count, poly in enumerate(pts1):
    #     box = get_box_from_poly(poly)
    #     iou_score = np.array([bb_intersection_over_union(box, box2i) for box2i in box2])
    #     if np.any(iou_score) > iou_thres:
    #         rm.append(count)
    # pts1 = np.delete(pts1, rm, axis=0)
    
    if len(pts1) == 0:
        pts = pts2
    elif len(pts2) == 0:
        pts = pts1
    elif len(pts1)==0 and len(pts2)==0: pts = []
    else:
        pts = np.vstack((pts1, pts2))   
    
    rm = []
    for i in range(len(pts)):
        for count, poly in enumerate(pts[i+1:]):
            j = count +i+1
            boxi, areai = get_box_from_poly(pts[i])
            boxj, areaj = get_box_from_poly(poly)
            iou_score = bb_intersection_over_union(boxi, boxj)
            if iou_score > iou_thres:
                if areai > areaj: rm.append(j)
                else: rm.append(i)
                
    pts = np.delete(pts, rm, axis=0)
    return pts


def convert_boxPoint2Yolo(pts, imgShape):
    yolo = []
    height = imgShape[0]
    width = imgShape[1]
    for poly in pts:
        x,y,w,h = cv2.boundingRect(poly.astype(int))
        x = (x+w/2)/width
        y = (y+h/2)/height
        yolo.append(np.array([x,y,w/width, h/height]))
    return yolo

def remove_small_text(pts):
    new_pts = []
    for poly in pts:
        x,y,w,h = cv2.boundingRect(poly.astype(int))
        if h > 10 and w > 10:
            new_pts.append(poly)
    return np.array(new_pts)

def sort_pts(pts, max_pts):
    heights = []    # sort by height
    for poly in pts:
        x,y,w,h = cv2.boundingRect(poly.astype(int))
        heights.append(h)
    new_poly = pts[np.argsort(heights)][::-1]
    return new_poly[:max_pts]

def expand_number2(boxes):
    new_boxes = []
    for box in boxes:
        rrect = cv2.minAreaRect(box.astype('float32'))
        temp = (rrect[1][0], rrect[1][1])
        if temp[0]>temp[1]: temp = (temp[0]*1.2, temp[1])
        else: temp = (temp[0], temp[1]*1.2)
        new_rrect = (rrect[0], temp, rrect[2])
        new_box = cv2.boxPoints(new_rrect)
        new_box[np.where(new_box<0)] = 0
        new_boxes.append(new_box)
        
    return new_boxes

def textSpotting(detect1, detect2, recog, img, max_word=16):
    # mmocr_det_res = detect2.readtext(img=[img.copy()])
    # pts_mmocr = np.array([np.array(pts[:8]).reshape((-1,2)) for pts in mmocr_det_res[0]['boundary_result']])
    # pts_mmocr[np.where(pts_mmocr < 0)] = 0

    word_boxes = extract_wordbox(detect2, img)


    
    preds, pts_pan, t = detect1.predict(img=img.copy())
    pts_pan[np.where(pts_pan < 0)] = 0
    # pts_pan = expand_box(pts_pan)

    pts = ensemble(word_boxes, pts_pan) 
    # remove small text
    # pts = remove_small_text(pts.copy())
    # sort and take up to max_word poly
    pts = sort_pts(pts, max_word)
    yolo = convert_boxPoint2Yolo(pts, img.shape)
    
    crop_imgs = [crop_with_padding(img, poly) for poly in pts]
    
    result_recog = recog.readtext(img=crop_imgs.copy(), batch_mode=True, single_batch_size=max_word)
    
    temp = {'boxPoint': None, 'boxYolo': None, 'text': None, 'text_score': None}
    result = []
    for count, poly in enumerate(pts):
        temp1 = temp.copy()
        temp1['boxPoint'] = poly
        temp1['boxYolo'] = yolo[count]
        temp1['text'] = result_recog[count]['text']
        temp1['text_score'] = result_recog[count]['score']
        result.append(temp1)
        
    ######### refine text value of number 3, expand box for case of 2 with 20%
    for count, res in enumerate(result):
        if res['text'] == '2':
            new_pts = [res['boxPoint']]
            new_pts = expand_number2(new_pts)
            result[count]['boxPoint'] = new_pts[0]

            new_img = [crop_with_padding(img, poly) for poly in new_pts]
            new_res = recog.readtext(img=new_img.copy(), batch_mode=False)
            result[count]['text'] = new_res[0]['text']

    return result

def textClassify(model1, model2, model3, pre_result):
    text = ' '.join([res['text'] for res in pre_result])
    level1 = model1.product_predict([text])
    level2 = model2.product_predict([text])
    level3 = model3.product_predict([text])
    has_age = model1.check_has_age(text)
    
    pre_result.append((level1, level2, level3, has_age))
    return pre_result
    
def crop_with_padding(img, pts):
    pts_ = pts.reshape((-1, 2)).astype(int)

    rect = cv2.boundingRect(pts_)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts_ = pts_ - pts_.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts_], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst

    return dst2

def word2line(result, img):
    temp = {'center': None, 'text': None}
    new_res = []
    zero_mask = np.zeros(img.shape[:2]).astype('uint8')
    zero_mask_copy = zero_mask.copy()
    for res in result:
        x,y,w,h = cv2.boundingRect(res['boxes'].astype(int))
        zero_mask[y+int(0.2*h):y+int(0.8*h), x:x+w] = 125
        # zero_mask = cv2.polylines(zero_mask, [res['boxes'].astype(int)], True, 255, -1)

        center = np.array([x+0.5*w, y+0.5*h]).astype(int)
        # print(cv2.pointPolygonTest(res['boxes'].astype(int),tuple(center),False))
        item = temp.copy()
        item['center'] = center
        item['text'] = res['text']
        new_res.append(item)

    kernel = np.ones((1, 20), np.uint8)    
    zero_mask = cv2.dilate(zero_mask, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(zero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # zero_mask_copy = cv2.drawContours(zero_mask_copy, contours, -1, 255, 2)
    # cv2.imrite('mask.jpg', zero_mask_copy)

    temp = {'contour': None, 'text': None, 'box': None}
    final_res = []  
    for contour in contours:
        box = cv2.boundingRect(contour.astype(int))
        item = temp.copy()
        item['box'] = np.array(box)
        item['contour'] = contour

        text_with_center = []
        temp1 = {'center': None, 'text': None}
        for pt in new_res:
            if cv2.pointPolygonTest(contour,tuple(pt['center']),False) > 0:
                item1 = temp1.copy()
                item1['text'] = pt['text']
                item1['center'] = pt['center']
                text_with_center.append(item1)
        
        text_with_center = np.array(text_with_center)
        only_center = [it['center'][0] for it in text_with_center]
        text_with_center = text_with_center[np.argsort(only_center)]
        
        item['text'] = ' '.join([text['text'] for text in text_with_center])
        final_res.append(item)

    return final_res

if __name__ == "__main__":
    ################## TextSpoting for product's image###############
    # Load models into memory
    detect_model = load_model('CRAFTpytorch/craft_mlt_25k.pth')
    mmocr_recog = MMOCR(det=None, recog='SAR', config_dir='mmocr1/configs/')

    model_path = 'pan/pretrain/pannet_wordlevel.pth'
    pan_detect = Pytorch_model(model_path, gpu_id=0)

    classifyModel_level1 = ClassifierInfer(path='textClassify/checkpoints/product/product_classifier_level1_f192.15.pkl')
    classifyModel_level2 = ClassifierInfer(path='textClassify/checkpoints/product/product_classifier_level2_f19306.pkl')
    classifyModel_level3 = ClassifierInfer(path='textClassify/checkpoints/product/product_classifier_level3_f1_8117.pkl')
    
    # inference
    img = cv2.imread('TextSpottingAndTextClassify/img_1.jpg')
    
    result = textSpotting(pan_detect, detect_model, mmocr_recog, img)

    result = textClassify(classifyModel_level1, classifyModel_level2, classifyModel_level3, result)
    
    #save txt
    
    print(result)
    #save image
    
    ################## TextSpoting for banner's image###############

    # load model recog
    model_path = 'textRecognition/best_accuracy.pth'
    opt = default_args(model_path)
    inferer = TextRecogInferer(opt)
    
    # detect text
    # cho nay o truyen vao img va boxes
    # img = cv2.polylines(img, np.array(boxes).astype(int), True, (255,255,255), -1)
    
    word_boxs = extract_wordbox(detect_model, img)
    # recog
    result = text_recog(img, word_boxs, inferer)