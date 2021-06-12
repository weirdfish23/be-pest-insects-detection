import torch
import cv2
from PIL import Image
import numpy as np
import boto3
from io import StringIO
import io


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

def add_labels(pil_img, df_results, colors):
    img1 = np.ascontiguousarray(pil_img)
    img1 = img1.copy()
    data = df_results.values
    for i in range(len(df_results)):
        x = data[i, :4]
        confidence = data[i, 4]
        class_n = data[i, 5]
        class_name = data[i, 6]
        plot_one_box(x, img1, label='{} {conf:.2f}'.format(class_name, conf=confidence), color=colors(int(class_n), True))
    return Image.fromarray(np.uint8(img1)).convert('RGB')

def make_detection(model, pil_img):
    results = model(pil_img)
    df_results = results.pandas().xyxy[0]
    img2 = add_labels(pil_img, df_results, colors)
    return img2, df_results

def save_on_s3(result_img, result_df, filename):
    url_result_img, url_result_csv = "", ""

    s3 = boto3.resource('s3')
    bucket = s3.Bucket('results-pest-detection')

    # save img
    key_img= 'detection_'+filename
    img_buffer = io.BytesIO()
    result_img.save(img_buffer, 'JPEG')
    aux = bucket.Object(key_img).put(Body=(bytes(img_buffer.getvalue())))
    url_result_img = "https://results-pest-detection.s3.amazonaws.com/%s" % (key_img)

    # save csv
    key_csv= 'detection_'+filename+'.csv'
    csv_buffer = StringIO()
    result_df.to_csv(csv_buffer)
    aux = bucket.Object(key_csv).put(Body=(bytes(csv_buffer.getvalue(), encoding='utf8')))
    url_result_csv = "https://results-pest-detection.s3.amazonaws.com/%s" % (key_csv)

    return url_result_img, url_result_csv
