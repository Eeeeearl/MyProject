import time
import glob
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
import torch
import os

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized


class LoadImg(object):
    def __init__(self, imgs, img_size=640, stride=32):
        self.imgs = imgs
        self.imgsz = img_size
        self.stride = stride
        self.nf = len(imgs)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        img0 = self.imgs[self.count]  # 传入的是RGB
        img0 = img0[:, :, ::-1]       # RGB to BGR
        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        self.count += 1
        return img, img0


class MyDetector(object):
    def __init__(self):
        # model.pt path(s)
        # self.weights = 'runs/train/重新标注后训练600轮/weights/best.pt'
        # self.weights = 'runs/train/data823_300_0.003/weights/last.pt'
        self.weights = 'runs/train/data901_300_0.003/weights/best.pt'
        self.source = '../sd_data/images/val'             # help='source' file/folder, 0 for webcam
        self.img_size = 1280                     # inference size (pixels)
        self.conf_thres = 0.4                  # object confidence threshold
        self.iou_thres = 0.5                    # IOU threshold for NMS
        self.max_det = 1000                     # maximum number of detections per image
        self.device = ''                        # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = True                    # display results
        self.save_txt = True                    # save results to *.txt
        self.save_conf = False                  # save confidences in --save-txt labels
        self.save_crop = False                  # save cropped prediction boxes
        self.nosave = False                     # do not save images/videos
        self.classes = None                     # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False               # class-agnostic NMS
        self.augment = False                    # augmented inference
        self.update = False                     # update all models
        self.project = 'runs/detect'            # save results to project/name
        self.name = 'exp'                       # save results to project/name
        self.exist_ok = False                   # existing project/name ok, do not increment
        self.line_thickness = 3                 # bounding box thickness (pixels)
        self.hide_labels = False                # hide labels
        self.hide_conf = False                  # hide confidences
        self.half = False                       # use FP16 half-precision inference
        self.start()                            # load model for once

    # 开始
    def start(self):
        # Initialize
        set_logging()  # 日志初始化
        # 是GPU还是CPU
        self.device = select_device(self.device)
        print('--------------------device:', self.device)
        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device)
                       .type_as(next(self.model.parameters())))  # run once

    # 预处理
    def preprocess(self, img_source):
        # Set Dataloader
        self.dataset = LoadImg(img_source, img_size=self.img_size, stride=self.stride)

        imgs, im0s = [], []
        for img, im0 in self.dataset:
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            imgs.append(img)
            im0s.append(im0)
        imgs = np.array(imgs)
        imgs = torch.from_numpy(imgs).to(self.device)
        imgs = imgs.half() if self.half else imgs.float()  # uint8 to fp16/32
        return imgs, im0s

    # 检测
    def detect(self, imgs):
        t0 = time.time()
        # Set Dataloader
        imgs, im0s = self.preprocess(imgs)

        # Inference 推理
        # time_synchronized() 是为了可以正确计算时间结果
        t1 = time_synchronized()
        pred = self.model(imgs, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        t2 = time_synchronized()

        # Process detections
        pred = self.reprocess(pred, imgs, im0s)

        print(f'Done. ({time.time() - t0:.3f}s)', 'total:(%dpics)' % len(imgs),
              f'1pic:({(time.time() - t0) / len(imgs):.3f}s)')
        return pred

    # 再加工
    # Process detections
    def reprocess(self, pred, imgs, im0s):
        # Process detections
        result = []
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s[i].copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(imgs.shape[2:], det[:, :4], im0.shape).round()
                det = self.reform_det(det, self.names)
                result.append(det)
        return result

    # 更正输出
    def reform_det(self, det, names):
        det = det.cpu()
        det = det.numpy()
        new_det = []
        for d in det:
            d = list(d)
            d[0] = int(d[0])
            d[1] = int(d[1])
            d[2] = int(d[2])
            d[3] = int(d[3])
            d[-1] = names[int(d[-1])]
            new_det.append(d)
        return new_det


# 创建存放检测结果文件夹
def get_root_path():
    root = './runs/detect/images/exp'

    save_i = 1
    while os.path.exists(root):
        print('已存在', root)
        root = root[:24] + str(save_i)
        save_i = save_i + 1

    os.mkdir(root)
    print('进行创建', root)
    return root


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    '''
    在img中注入中文字
    :param img: cv2图片
    :param text: 标识文字
    :param left: x坐标
    :param top: y坐标
    :param textColor: Box框颜色
    :param textSize: 文字大小
    :return: cv2格式的图片
    '''

    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        # print('isinstance')
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    # return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    # imgs
    # path = glob.glob('./runs/detect/images/val/*.jpg')
    # path = glob.glob('E:/jingdi/workplace/code/yolo5_demo/test_data/*.jpg')

    # path = glob.glob('M:\\项目\\布匹检测\\DefectDetector\\image\\batch-20210914154628\\ng\\*.jpg')
    path = glob.glob('E:\\jingdi\\workplace\\code\\rui-9-10\\image\\test_image\\test01.jpg')
    print(path)
    # path = glob.glob('E:/code/yolo5_demo/灰度sd_data/images/val/*.jpg')
    imgs = []
    for p in path:
        p = Image.open(p).convert("RGB")
        img = np.array(p)

        # img = np.array(Image.open(p))

        imgs.append(img)
        # 展示下图片
        # im = Image.fromarray(img)
        # im.show()
        # break

    # # detect
    d = MyDetector()
    root_path = get_root_path()
    num = 1
    for i in range(len(imgs)):
        images = imgs[i:i+1]
        # print(images)
        pred = d.detect(images)
        print('num: {}, pred: {}'.format(i, pred))

        # show
        for j, det in enumerate(pred):
            img = imgs[i]
            for box in det:
                # print(box)
                # 依据检测结果，hole-破洞-蓝色 blot-污渍-红色 进行画不同颜色的框
                # img = imgs[i]
                # 调整添加文字信息内容
                add_txt = str(box[-1]) + str(box[-2])[:5]
                if box[-1] == 'hole':
                    img = cv2.rectangle(img, tuple(box[:2]), tuple(box[2:4]), (0, 0, 255), 3)  # R G B blue
                    img = cv2ImgAddText(img, add_txt, box[0], box[1] - 60, (255, 0, 0), 60)  # BGR
                elif box[-1] == 'stain':
                    img = cv2.rectangle(img, tuple(box[:2]), tuple(box[2:4]), (255, 0, 0), 3)  # red
                    img = cv2ImgAddText(img, add_txt, box[0], box[1] - 60, (255, 0, 0), 60)
                else:
                    pass
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imshow('img', img)
            nam = path[i].split('\\')[-1].replace('.jpg', '')
            print(nam)
            # img_name = '/detected_' + str(num) + '.jpg'
            img_name = '/' + nam + '.jpg'
            # 保存图片
            saved_path = root_path + img_name
            cv2.imwrite(saved_path, img)
            num = num + 1
            cv2.waitKey(1000)

        cv2.destroyAllWindows()



