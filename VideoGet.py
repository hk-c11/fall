import numpy as np
import cv2

def is_inside(o, i):# 判断o是否在i外的函数
   ox, oy, ow, oh = o
   ix, iy, iw, ih = i
   return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

def detect_person(image,capture):# 检测被追踪的人体的函数
    ret, img = capture.read()
    hog = cv2.HOGDescriptor()# 创建HOGDescriptor对象
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())# 使用OpenCV预训练好的人体识别检测器
    found, w = hog.detectMultiScale(img)# 构造了图像金字塔和滑窗,进行检测
    found_filtered = []# 存储检测目标
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
            else:
                found_filtered.append(r)
    # for person in found_filtered:
    return found_filtered # 返回检测到的人体的位置

def getVedio(path):
    global found_filtered, frame
    fcap = cv2.VideoCapture(path)
    while fcap.isOpened():
        # 3.获取每一帧图像
        ret, frame = fcap.read()
        found_filtered = detect_person(frame,fcap)
        print(len(found_filtered))
        if (len(found_filtered) != 0):
            break  # 跳出while 循环
    c, r, w, h = found_filtered[0]  # 此处仅跟踪了一个人体,实际应该考虑画面中有多个人的情况
    track_window = c, r, w, h  # 人体在画面中的位置范围
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换为HSV图
    mask = cv2.inRange(hsv_roi, np.array((0,30,60)),np.array((20,150,255)))  # 选取色彩范围
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])  # 计算Histogram
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # Histogram归一化
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)  # 准备CamShift
    while (1):
        ret, frame = fcap.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换HSV图
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)  # BackProject
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)  # Camshift方法-物体追踪
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.imshow('Tracked Person', img2)
        key = cv2.waitKey(1)  # 等待键盘输入,间隔1ms waits for a key event infinitely (when [delay]< 0 ) or for [delay] milliseconds,
        if key == 27:  # ESC键的ASCII码
            print("detect ESC")
            break  # 退出while循环
    fcap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    getVedio("E:\\GraduationProject\\Le2i Fall\\FallDataset\\Coffee_room_01\\Coffee_room_01\\Videos\\video (1).avi")
    # getVedio(0)
