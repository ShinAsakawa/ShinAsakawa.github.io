from torchvision.ops import nms
from torchvision import transforms as torchtrans  
import PIL
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def apply_nms(orig_prediction, iou_thresh=0.3):
    
    # torchvision は保持すべき bbox のインデックスを返す
    keep = nms(
    #keep = torchvision.ops.nms(
        orig_prediction['boxes'], 
        orig_prediction['scores'], 
        iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    return final_prediction


def torch_to_pil(img):
    """torchtensor を PIL 画像に変換する関数"""
    return torchtrans.ToPILImage()(img).convert('RGB')


def plot_img_bbox(img:PIL.Image, 
                  target:dict, 
                  title=None,
                  figsize:tuple=(7,7)):
    
    """画像中のバウンディングボックスを可視化する関数"""
    # バウンディングボックスは以下のように定義されます: x-min y-min 幅 高さ
    # Bounding boxes are defined as follows: x-min y-min width height
    
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(figsize)
    ax.imshow(img)
    print(target)
    
    for box in target['boxes']:
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle(
            (x, y),
            width, 
            height,
            linewidth = 4,
            edgecolor = 'red',
            facecolor = 'none')

        # 画像上にバウンディングボックスを描画 
        # Draw the bounding box on top of the image
        ax.add_patch(rect)
        
    if title != None:
        ax.set_title(title)
    plt.show()
    
    
def _plot_img_bbox(img:PIL.Image,
                  target:dict,
                  title:str=None,
                  edgecolor:str="red",
                  linewidth:int=4,
                  figsize:tuple=(7,7),
                  verbose=False,
                  ):

    """画像中のバウンディングボックスを可視化する関数"""
    # バウンディングボックスは以下のように定義されます: x-min y-min 幅 高さ
    # Bounding boxes are defined as follows: x-min y-min width height

    print(target) if verbose else None

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(figsize)
    ax.imshow(img)

    for box in target['boxes']:
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth = linewidth,
            edgecolor = edgecolor,
            facecolor = 'none')

        # 画像上にバウンディングボックスを描画
        # Draw the bounding box on top of the image
        ax.add_patch(rect)

    if title != None:
        ax.set_title(title)
    #if verbose:
    plt.show()

    return img, ax    
