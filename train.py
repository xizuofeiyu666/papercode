#这是提供 参考示例的 YOLOMM训练器使用方法
#按必须按照你的需求来更改而不是盲目去使用

from ultralytics import YOLOMM
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':

    model = YOLOMM(r'C:\Users\Administrator\Desktop\HGDC-Det\cfg\models\HGDCDet.yaml')
    model.train(data=r'C:\Users\Administrator\Desktop\MutilModel-114\ultralytics\datasets\M3FD.yaml',
                task='detect',
                imgsz=640,
                epochs=300,
                device='0',
                batch=16,
                workers=8,
                optimizer='SGD',
                amp=True,
                # resume=True,
                cache=False,
                project='runs12/train',
                name='HGDC-Det-M3FD',
                )