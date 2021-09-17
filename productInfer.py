# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
from PIL import ImageFont, ImageDraw, Image
import argparse
import sys
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.ensemble_labels import display_image,meger_label_branch,word2line,Ensemble,Analyzer
import numpy as np
from spellchecker import SpellChecker

from utils.general import xywh2xyxy,clip_coords
from textSpotting import textSpottingInfer
from classifyText import textClassifyInfer
from classifyImage import objectClasssifyInfer 


craft_detect, model_recognition = textSpottingInfer.load_model_1()
mmocr_recog,pan_detect,classifyModel_level1,mapping_checkpoints = textClassifyInfer.load_model()
chinh_model,model_step,labels_end,labels_branch,dict_middle,dict_step= objectClasssifyInfer.load_model()

spell = SpellChecker(language=None,)  # loads default word frequency list
spell.word_frequency.load_text_file('corpus.txt')

with open('keywords.txt', 'r') as f:
    lines = f.readlines()
keywords = []
for line in lines:
    line = line.strip().split()
    keywords += line


def run(weights=['models/weights/binh_new_best.pt', 'models/weights/sua_new_best_2.pt'],  # model.pt path(s)
        # file/dir/URL/glob, 0 for webcam
        source='/content/drive/MyDrive/Colab Notebooks/data_test_end',
        imgsz=640,  # inference size (pixels)
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        # save results to project/name
        project='/content/drive/MyDrive/Colab Notebooks/results',
        name='data_test_end',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):

    save_img = not nosave and not source.endswith(
        '.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, graph_def = (
        suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model_binh_sua = attempt_load(
            weights[0], map_location=device)  # load FP32 model
        model_sua = attempt_load(
            weights[1], map_location=device)  # load FP32 model

        stride_binh = int(model_binh_sua.stride.max())  # model stride
        names_binh = model_binh_sua.module.names if hasattr(
            model_binh_sua, 'module') else model_binh_sua.names  # get class names
        names_sua = model_sua.module.names if hasattr(
            model_sua, 'module') else model_sua.names  # get class names
        if half:
            model_binh_sua.half()  # to FP1
            names_sua.half()  # to FP1
        if classify:  # second-stage classifier
            model_binh_sua_c = load_classifier(
                name='resnet50', n=2)  # initialize
            model_binh_sua_c.load_state_dict(torch.load(
                'resnet50.pt', map_location=device)['model']).to(device).eval()

            model_sua_c = load_classifier(name='resnet50', n=2)  # initialize
            model_sua_c.load_state_dict(torch.load('resnet50.pt', map_location=device)[
                                        'model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    imgsz_binh = check_img_size(imgsz, s=stride_binh)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz_binh, stride=stride_binh)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz_binh, stride=stride_binh)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model_binh_sua(torch.zeros(1, 3, imgsz_binh, imgsz_binh).to(
            device).type_as(next(model_binh_sua.parameters())))  # run once
        model_sua(torch.zeros(1, 3, imgsz_binh, imgsz_binh).to(
            device).type_as(next(model_sua.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if pt:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        elif onnx:
            img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(
                save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred_binh = model_binh_sua(
                img, augment=augment, visualize=visualize)[0]
            pred_sua = model_sua(img, augment=augment, visualize=visualize)[0]

        # NMS
        pred_binh = non_max_suppression(
            pred_binh, 0.4, iou_thres, classes, agnostic_nms, max_det=max_det)
        pred_sua = non_max_suppression(
            pred_sua, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()
        
        # Second-stage classifier (optional)
        if classify:
            pred_binh = apply_classifier(
                pred_binh, model_binh_sua_c, img, im0s)
            pred_sua = apply_classifier(pred_sua, model_sua_c, img, im0s)

        # Process predictions binh sua

        for i, (det_binh, det_sua) in enumerate(zip(pred_binh, pred_sua)):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy()  # for save_crop
            with open(txt_path + '.txt', 'a') as f:
                txt_path_end = save_path.replace(
                    '.jpg', '_return.txt').replace('.png', '_return.txt')
                with open(txt_path_end, "w") as file_label:
                    if len(det_binh):
                        # Rescale boxes from img_size to im0 size
                        det_binh[:, :4] = scale_coords(
                            img.shape[2:], det_binh[:, :4], im0.shape).round()

                        # Print results
                        for c in det_binh[:, -1].unique():
                            # detections per class
                            n = (det_binh[:, -1] == c).sum()
                            # add to string
                            s += f"{n} {names_binh[int(c)]}{'s' * (n > 1)}, "

                        # Write results
                        for *xyxy, conf, cls in reversed(det_binh):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(
                                    1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                # label format
                                line = (
                                    cls, *xywh, conf) if save_conf else (cls, *xywh)
                                f.write(('%g ' * len(line)).rstrip() %
                                        line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                # To classification
                                c = int(cls)  # integer class
                                label = None if hide_labels else (
                                    names_binh[c] if hide_conf else f'{names_binh[c]} {conf:.2f}')
                                im0 = plot_one_box(xyxy, im0, label=label, color=colors(
                                    0, True), line_width=line_thickness)
                                # if save_crop:
                                #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names_binh[c] / f'{p.stem}.jpg', BGR=True)

                    list_bbox = []
                    if len(det_sua):
                        # Rescale boxes from img_size to im0 size
                        det_sua[:, :4] = scale_coords(
                            img.shape[2:], det_sua[:, :4], im0.shape).round()

                        # Print results
                        for c in det_sua[:, -1].unique():
                            # detections per class
                            n = (det_sua[:, -1] == c).sum()
                            # add to string
                            s += f"{n} {names_sua[int(c)]}{'s' * (n > 1)}, "
                        # Write results
                        crop_st = 0 
                        for *xyxy, conf, cls in reversed(det_sua):
                            list_bbox.append(xyxy)
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(
                                    1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                # label format
                                line = (
                                    4, *xywh, conf) if save_conf else (4, *xywh)
                                f.write(('%g ' * len(line)).rstrip() %
                                        line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                # Crop Box
                                BGR = True
                                xyxy_test = torch.tensor(xyxy).view(-1, 4)
                                b = xyxy2xywh(xyxy_test)  # boxes
                                b[:, 2:] = b[:, 2:] * 1.02 + \
                                    10  # box wh * gain + pad
                                xyxy_crop = xywh2xyxy(b).long()
                                clip_coords(xyxy_crop, imc.shape)
                                crop = imc[int(xyxy_crop[0, 1]):int(xyxy_crop[0, 3]), int(
                                    xyxy_crop[0, 0]):int(xyxy_crop[0, 2]), ::(1 if BGR else -1)]
                                # End crop
                                # To classification

                                height_crop,width_crop,_ =  crop.shape
                                if height_crop > 65 and  width_crop > 50:
                                  crop_st+=1
                                  jpng_crop_end = save_path.replace('.jpg', '_{0}.jpg'.format(crop_st)).replace('.png', '_{0}.png'.format(crop_st))
                                  txt_crop_end = save_path.replace('.jpg', '_{0}.txt'.format(crop_st)).replace('.png', '_{0}.txt'.format(crop_st))
                                  cv2.imwrite(jpng_crop_end,crop)

                                  file_txt_xrop = open(txt_crop_end,"w") 
                                  output_brand, sc1 = objectClasssifyInfer.predict(
                                      chinh_model, crop, return_features=False)

                                  result_text_spotting = textClassifyInfer.spotting_text(
                                      pan_detect, craft_detect, mmocr_recog, crop)

                                  result = textClassifyInfer.predict(result_text_spotting.copy(
                                  ), classifyModel_level1, classifyModel_level3=None, branch=True)

                                  branch_0 = result[-1][0][0].replace(" ", "_")
                                  text_list = []
                                  for i in result[:-1]:
                                      text = i['text'].lower().replace(' ', '_')
                                      text_list.append(text)
                                  test_keyword = False
                                  for text_ in text_list:
                                      if text_ in keywords:
                                          test_keyword = True
                                          break
                                  c = (list(labels_branch.keys())
                                      [output_brand].strip())
                                  if test_keyword == True:
                                      output_final_branch = result[-1][0][0]
                                  elif len(text_list) == 0 and sc1 < 0.98:
                                      output_final_branch = 'Unknow'
                                  elif len(text_list) >= 4:
                                      branch_0 = c
                                      if sc1 > 0.95:
                                          output_final_branch = c
                                      else:
                                          output_final_branch = 'Unknow'
                                  else:
                                      if sc1 > 0.93:
                                          output_final_branch = c
                                      else:
                                          output_final_branch = 'Unknow'
                                  output_final_branch = output_final_branch.replace(" ", "_")
                                  label = output_final_branch

                                  check_list = False
                                  output_merge, _ = objectClasssifyInfer.predict_merge_model(
                                      model_step, crop)
                                  if len(dict_middle[str(output_merge)]) > 1:
                                      check_list = True
                                  name_merge = dict_middle[str(
                                      output_merge)][-1]
                                  brand_merge = name_merge.split("/")[0]
                                  file_txt_xrop.writelines(
                                      "============RESULT CHINH====================\n")
                                  file_txt_xrop.writelines(name_merge+"\n")

                                  temp_step = None
                                  if output_final_branch in ["f99foods", "heinz", "bubs_australia", "megmilksnowbrand", "meiji"]:
                                      pass
                                  else:
                                      if output_final_branch in dict_model.keys():
                                          classifyModel_level3 = dict_model[output_final_branch]                                          
                                          result_2 = textClassifyInfer.predict(
                                              result_text_spotting, classifyModel_level1, classifyModel_level3, step=True, added_text=''.replace(' ', '_'))
                                          temp_step = result_2[-1][0].replace(
                                              " ", "_")
                                          brand_text = meger_label_branch(
                                              labels_end, 2, temp_step)
                                          file_txt_xrop.writelines(
                                              "============RESULT THANH====================\n")
                                          file_txt_xrop.writelines(
                                              result_2[-1][0].replace(" ", "_")+"\n")
                                      else:
                                          print(output_final_branch)

                                  esem = Ensemble(
                                      output_final_branch, output_merge, temp_step, dict_middle, dict_step, text_list)
                                  label = esem.run()  
                                  file_txt_xrop.writelines("============TEXT===============\n")
                                  file_txt_xrop.writelines([i+"\n" for i in text_list])
                                  file_txt_xrop.writelines("============Ti le===============\n")
                                  file_txt_xrop.writelines(str(width_crop / height_crop)+"\n")
                                  file_txt_xrop.writelines("============RESULTS END===============\n")
                                  file_txt_xrop.writelines(label)
                                # display_image(crop)
                                  if width_crop / height_crop < 0.495: #(4/7):
                                    label = output_final_branch
                                else:
                                  label = "size nho ({0} x {1})".format(width_crop,height_crop)
                                file_label.writelines(
                                    "=================== NHÃN SẢN PHẨM ====================\n")
                                file_label.writelines(label + "\n")
                                im0 = plot_one_box(xyxy, im0, label=label, color=colors(
                                    0, True), line_width=line_thickness)
                        result_text = textSpottingInfer.predict(
                            imc, craft_detect, model_recognition)
                        final_res = word2line(result_text, imc)
                        list_text = []
                        for res in final_res:
                            x1, y1, w, h = res['box']
                            # bbox = res['boxes']
                            # res_b = textSpottingInfer.get_box_from_poly(bbox)
                            x2 = x1+w
                            y2 = y1+h
                            # c1, c2 = (int(res_b[0]), int(res_b[1])), (int(res_b[2]), int(res_b[3]))
                            text = res['text']
                            # tokens = text.split()
                            # text_end = ''
                            # for token in tokens:
                            #     text_end=text_end+spell.correction(token)+" "
                            list_text.append((text))
                            c1, c2 = (x1, y1), (x2, y2)
                            cv2.rectangle(im0, c1, c2, colors(
                                0, True), thickness=line_thickness, lineType=cv2.LINE_AA)
                        list_text = [x+'\n' for x in list_text]

                        file_label.writelines(
                            "=================== NỘI DUNG TEXT ====================\n")
                        file_label.writelines(list_text)
                    # End model recognition
                    # Save results (image with detections)

                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path[i] != save_path:  # new video
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    # release previous video writer
                                    vid_writer[i].release()
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(
                                        cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(
                                        cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer[i] = cv2.VideoWriter(
                                    save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer[i].write(im0)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=['models/weights/binh_new_best.pt','models/weights/sua_new_best.pt'], help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/content/milk_classification/data_test', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--project', default='./results', help='save results to project/name')
    parser.add_argument('--name', default='test', help='save results to project/name')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(
    source=opt.source,
    project=opt.project,
    name = opt.name
    )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

