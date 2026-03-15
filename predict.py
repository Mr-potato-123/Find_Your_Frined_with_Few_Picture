import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN

# ---------- 人脸检测器（单例） ----------
class FaceDetector:
    _instance = None
    def __new__(cls, device=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_detector(device)
        return cls._instance

    def _init_detector(self, device):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=device)

    def detect(self, img_rgb):
        boxes, probs = self.mtcnn.detect(img_rgb)
        if boxes is None:
            return np.empty((0, 4))
        return boxes

# ---------- 模型加载 ----------
def _load_model(model_path, device):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# ---------- 预处理 ----------
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def _preprocess_face(face_rgb):
    face_pil = Image.fromarray(face_rgb)
    return _transform(face_pil).unsqueeze(0)

# ---------- 自适应字体大小 ----------
def _get_font_scale(image_width, base_scale=0.6, base_width=800):
    scale = base_scale * (image_width / base_width)
    return max(0.4, min(2.0, scale))

# ---------- 预测接口 ----------
def predict(image_path, model_path='best_model.pth', output_path='output.jpg'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _load_model(model_path, device)
    detector = FaceDetector(device)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_height, img_width = img_bgr.shape[:2]

    boxes = detector.detect(img_rgb)
    faces = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face_rgb = img_rgb[y1:y2, x1:x2]
        if face_rgb.size == 0:
            continue
        face_tensor = _preprocess_face(face_rgb).to(device)
        with torch.no_grad():
            output = model(face_tensor)
            prob = F.softmax(output, dim=1)[0, 1].item()
        faces.append((box, prob))

    if faces:
        best_face = max(faces, key=lambda x: x[1])
        box, best_prob = best_face
        if best_prob > 0.5:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            font_scale = _get_font_scale(img_width)
            label = f'CYT {best_prob:.2f}'
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.rectangle(img_bgr, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
        else:
            print(f"最高置信度 {best_prob:.2f} 低于阈值0.5，未标记。")
    else:
        print("未检测到人脸。")

    cv2.imwrite(output_path, img_bgr)
    return output_path

# ---------- 命令行调用 ----------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Face recognition for CYT')
    parser.add_argument('--image', type=str,default="test.jpg" , help='输入图片路径')
    parser.add_argument('--model', type=str, default='best_model.pth', help='模型文件路径')
    parser.add_argument('--output', type=str, default='output.jpg', help='输出图片路径')
    args = parser.parse_args()
    predict(args.image, args.model, args.output)
    print(f"完成，结果已保存至 {args.output}")