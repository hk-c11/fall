import cv2


class Predict(object):
    def __init__(self):
        # 获取hog检测器对象
        self.hog = cv2.HOGDescriptor()
        # 设置检测人的默认检测器
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("模型加载完成！")

    # 检测i方框 包含o方框
    def is_inside(self, o, i):
        ox, oy, ow, oh = o
        ix, iy, iw, ih = i
        return ox > ix and ox + ow < ix + iw and oy + oh < iy + ih

    # 将人外面的方框画出来
    def draw_person(self, image, person):
        x, y, w, h = person
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    def detect_img(self, img):
        # 在图片中检测人，
        # 返回found列表 每个元素是一个(x, y, w, h)的矩形，w是每一个矩形的置信度
        if img is None:
            return
        found, w = self.hog.detectMultiScale(img)
        found_filtered = []
        # 如果方框有包含，只留下内部的小方块
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and self.is_inside(r, q):
                    break
                else:
                    found_filtered.append(r)
        # 将每一个方块画出来
        for person in found_filtered:
            self.draw_person(img, person)
        return img

    def detect_video(self, video_path):
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")

        while vid.isOpened():
            return_value, frame = vid.read()
            result = self.detect_img(frame)
            if result is not None:
                cv2.imshow("person detection", result)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    predict = Predict()
    video_path = "E:\\GraduationProject\\Le2i Fall\\FallDataset\\Coffee_room_01\\Coffee_room_01\\Videos\\video (17).avi"
    predict.detect_video(video_path=video_path)
