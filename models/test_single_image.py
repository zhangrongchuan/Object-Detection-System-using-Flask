#detect single image
from torchvision import transforms
from PIL import ImageDraw
import torch

def detect_single_image(image,model,decoder):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tensorImage=transforms.ToTensor()(image)
    tensorImage=tensorImage.to(device)
    model=model.to(device)
    tensorImage=torch.reshape(tensorImage,[1,3,256,256])

    f1,f2=model(tensorImage)
    result=filter_pred_bbox(f1,f2,decoder)[0]

    draw=ImageDraw.Draw(image)
    
    for i in result:
        box=[i[0],i[1],i[2],i[3]]
        draw.rectangle(box,outline="orange",width=2)

    image.save("static/pic/result.png")
    return result

def filter_pred_bbox(feature1,feature2,Decoder):
    #用于decode神经网络的输出，使得我们可以得到预测的目标的(x1,y1,x2,y2,confident),所以shape=(n,5)
    fea1=Decoder.decode_box(4,feature1)
    fea2=Decoder.decode_box(16,feature2)
    f1_=Decoder.remove_low_confident(fea1,threshold=0.98)
    f2_=Decoder.remove_low_confident(fea2,threshold=0.90)
    result1=Decoder.change_position_format(f1_)
    result2=Decoder.change_position_format(f2_)
    result1=Decoder.non_maximum_suppression(result1,threshold=0.4)
    result2=Decoder.non_maximum_suppression(result2,threshold=0.0)
    result=Decoder.concate_three_feature(result1,result2)
    result=Decoder.non_maximum_suppression(result,threshold=0.0)

    return result

if __name__=="__main__":
    detect_single_image()
