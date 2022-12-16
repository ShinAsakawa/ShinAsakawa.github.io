import torch
import numpy as np
import cv2
import typing

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import matplotlib.pyplot as plt
from termcolor import colored

DEFAULT_CANVAS_WIDTH  = 224
DEFAULT_CANVAS_HEIGHT = 224
DEFAULT_BGCOLOR = (255,255,255)  # 真っ白
DEFAULT_MODE = "RGB"
DEFAULT_ANCHOR = 'la'   # l:left, a:ancestor だったかな。
DEFAULT_FONTSIZE = int(DEFAULT_CANVAS_WIDTH / 16)

from glob import glob
import os
import IPython
isColab = 'google.colab' in str(IPython.get_ipython())

hira_chars = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'
kata_chars = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン'
digits = '０１２３４５６７８９'
alphabets = 'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
star = '★'
background = '<background>'
#symbols = [background, star, hira_chars, kata_chars, digits, alphabet]
#symbols = [background, star] + hira_chars + kata_chars + digits + alphabet
symbols = [background, star]
for _type in [hira_chars, kata_chars, digits, alphabets]:
    for c in _type:
        symbols.append(c)

#print(list(PIL.ImageColor.colormap))
colornames = ['black', 'blue', 'brown', 'cyan', 'green', 'magenta', 'orange', 'purple', 'red', 'yellow'] # , 'white']


def rad2deg(rad:float)->float:
    """1度が180/pi ラジアンなのでその変換"""
    return 180 * rad / np.pi


def deg2rad(deg:float)->float:
    """度をラジアンに変換する"""
    return np.pi * deg / 180

import torch
def imshow(img, title=None):
    """Imshow for Tensor.
    Pytorch 公式チュートリアルの transfer_learning_tutorial.ipyb を元に若干の修正を施した
    """
    
    if isinstance(img, torch.Tensor):
        img = inp.clone().numpy().transpose((1,2,0))
        
        # 平均を引いて，標準偏差で割っている変換を元に戻す
        mean = np.array([0.485, 0.456, 0.406])  # 平均
        std = np.array([0.229, 0.224, 0.225])   # 標準偏差
        img = std * img + mean                  
        img = np.clip(img, 0, 1)                # 外れ値を切り取り
        
    plt.imshow(iimg)
    plt.title(title) if title is not None else None
    plt.pause(0.001)  # pause a bit so that plots are updated


def PILImage2torch(img:PIL.Image,
                   width:int=DEFAULT_CANVAS_WIDTH,
                   height:int=DEFAULT_CANVAS_HEIGHT,
                  )->torch.Tensor:
    image = np.array(img)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_res = cv2.resize(img_rgb, (width, height), cv2.INTER_AREA)
    img = img_res / 255.0
    img = torch.Tensor(img).permute(2,0,1)
    return img

def draw_a_rotated_line(canvas:PIL.ImageDraw=None,
                        theta:float=0,
                        area:tuple=(0,0,224,224),
                        bgcolor:tuple=(255,255,255),
                        color:[tuple or str]='black',
                        line_width:int=4,
                        perturbation=False,
                        margin:int=4,
                       )->None:
    left, top, right, bottom = area
    
    width = right - left
    height = bottom - top
    
    # 描画領域を中心部 1/8 - 7/8 に限定
    left   += width >> 3
    top    += height >> 3
    right  -= width >> 3
    bottom -= height >> 3
    
    # 半径を計算
    rad_x = (right - left) >> 1
    rad_y = (bottom - top) >> 1
    
    # 中心を計算
    cx, cy = left + rad_x, top + rad_y
    
    if perturbation == True:
        off_x = np.random.randint(width>>2) - (width>>3)
        off_y = np.random.randint(height>>2) - (height>>3)
        cx += off_x
        cy += off_y
        
    r = rad_x if rad_x > rad_y else rad_y
    
    # 回転後のオフセット座標を計算
    offs_x = r * np.cos(theta)
    offs_y = r * np.sin(theta)
    
    _top    = int(cy + offs_y)
    _bottom = int(cy - offs_y)
    _left   = int(cx - offs_x)
    _right  = int(cx + offs_x)
    
    canvas.line(xy=(_left,_top,_right,_bottom),
                fill=color,
                width=line_width)
    
    # 領域箱を計算
    if _left > _right:
        _right, _left = _left, _right
    if _bottom < _top:
        _top, _bottom = _bottom, _top
    _top    -= margin
    _bottom += margin
    _left   -= margin
    _right  += margin
    bbox = (_left, _top, _right, _bottom)
    return {'bbox':bbox, 'color':color, 'line_width':line_width}



def make_div_areas(width:int=DEFAULT_CANVAS_WIDTH,
                   height:int=DEFAULT_CANVAS_HEIGHT,
                   div:int=2)->list:
    areas = []
    
    unit_w = width // div
    unit_h = height // div

    for _h  in range(div):
        top = unit_h * _h
        bottom = top + unit_h
        for _w in range(div):
            left  = unit_w * _w
            right = left + unit_w
            area = (left,top,right,bottom)
            areas.append(area)

    return areas



def make_a_canvas(width:int=DEFAULT_CANVAS_WIDTH,
                  height:int=DEFAULT_CANVAS_HEIGHT,
                  color:tuple=DEFAULT_BGCOLOR,
                  verbose:bool=False,
                  mode:str=DEFAULT_MODE,
                 )->PIL.Image.Image:
    '''刺激を描くためのキャンバスを作成して返す'''
    #print(f'画像サイズ width:{width}, height:{height}') if verbose else None
    img = Image.new(mode=mode, 
                    size=(width, height), 
                    color=color)  
    draw = ImageDraw.Draw(img)
    return img, draw
    

# def make_a_canvas(width:int=DEFAULT_CANVAS_WIDTH,
#                   height:int=DEFAULT_CANVAS_HEIGHT,
#                   color:tuple=(255,255,255), # 真っ白な (255,255,255) 画像
#                   mode:str="RGB",
#                   verbose:bool=False,
#                  ): # ->(PIL.Image.Image, int, int):
#     '''刺激を描くためのキャンバスを作成して返す'''

#     global DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT
#     global DEFAULT_FONTSIZE, FONTSIZE_RANGE
#     global X_RANGE, Y_RANGE, X0, Y0, DEFAULT_ANCHOR
    
#     # if width != DEFAULT_CANVAS_WIDTH:
#     #     #print(f'DEFAULT_CANVAS_WIDTH:{DEFAULT_CANVAS_WIDTH}')
#     #     #print(f'X_RANGE:{X_RANGE}')
#     #     #print(f'width:{width} type(width):{type(width)}')
#     #     set_default_canvas_values(width=width)
#     #     #print(f'DEFAULT_CANVAS_WIDTH:{DEFAULT_CANVAS_WIDTH}')
#     #     #print(f'X_RANGE in make_a_canvas:{X_RANGE}')
#     # if height != DEFAULT_CANVAS_HEIGHT:
#     #     set_default_canvas_values(height=height)
    
#     #print(f'画像サイズ width:{width} type(width):{type(width)}, height:{height} type(height):{type(height)}')
#     #print(f'画像サイズ width:{width}, height:{height}') if verbose else None
    
#     img = Image.new(mode=mode, 
#                     size=(width, height), 
#                     color=color)  
#     draw = ImageDraw.Draw(img)  # draw オブジェクト ある種のキャンバスを生成
#     return img, draw, width, height


# def set_new_canvas(width:int=DEFAULT_CANVAS_WIDTH,
#                    height:int=DEFAULT_CANVAS_HEIGHT):
#     canvas, draw, width, height = make_a_canvas(width=width, height=height)
    
#     return canvas, draw, width, height


X_RANGE = (DEFAULT_CANVAS_WIDTH >> 2) + (DEFAULT_CANVAS_WIDTH>>1)
Y_RANGE = (DEFAULT_CANVAS_HEIGHT >> 2) + (DEFAULT_CANVAS_HEIGHT>>1)
X_RANGE = (DEFAULT_CANVAS_WIDTH >> 4) * 13
Y_RANGE = (DEFAULT_CANVAS_HEIGHT >> 4) * 13
X_RANGE = (DEFAULT_CANVAS_WIDTH >> 4) * 13 + (DEFAULT_FONTSIZE>>1)
Y_RANGE = (DEFAULT_CANVAS_HEIGHT >> 4) * 13 + (DEFAULT_FONTSIZE>>1)
FONTSIZE_RANGE = DEFAULT_FONTSIZE >> 1
FONTSIZE_RANGE = (DEFAULT_FONTSIZE >> 2) * 3
X0 = (DEFAULT_CANVAS_WIDTH >> 3) - (DEFAULT_FONTSIZE >> 1)
Y0 = (DEFAULT_CANVAS_HEIGHT >> 3) - (DEFAULT_FONTSIZE >> 1)
X0 = (DEFAULT_CANVAS_WIDTH >> 4) - (DEFAULT_FONTSIZE >> 1)
Y0 = (DEFAULT_CANVAS_HEIGHT >> 4) - (DEFAULT_FONTSIZE >> 1)
#X0 = (DEFAULT_CANVAS_WIDTH >> 4) - (DEFAULT_FONTSIZE >> 2) * 3
#Y0 = (DEFAULT_CANVAS_HEIGHT >> 4) - (DEFAULT_FONTSIZE >> 2) * 3

def set_font(fontname='NotoSansJP-Bold',
             fontsize:int=None,
             default_font=None)->None:
    '''フォントを設定する'''
    
    global DEFAULT_FONTSIZE
    
    DEFAULT_FONTSIZE = fontsize if fontsize != DEFAULT_FONTSIZE else DEFAULT_FONTSIZE
    default_font = fontdata[fontname]['data']
    DEFAULT_FONT = ImageFont.truetype(default_font, size=DEFAULT_FONTSIZE)


#def set_default_canvas_values(width:int=224,
#                              height:int=224,
def set_default_canvas_values(width:int=DEFAULT_CANVAS_WIDTH,
                              height:int=DEFAULT_CANVAS_HEIGHT,
                              FONTSIZE:int=DEFAULT_FONTSIZE,
                              x_range:int=X_RANGE,
                              y_range:int=Y_RANGE,
                              fontsize:int=DEFAULT_FONTSIZE,
                              fontsize_range:int=FONTSIZE_RANGE,
                              x0:int=X0,
                              y0:int=Y0,
                              anchor:str=DEFAULT_ANCHOR)->None:

    global DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT
    global DEFAULT_FONTSIZE, FONTSIZE_RANGE
    global X_RANGE, Y_RANGE, X0, Y0, DEFAULT_ANCHOR

    X_RANGE = x_range if x_range != X_RANGE else X_RANGE
    Y_RANGE = y_range if y_range != Y_RANGE else Y_RANGE
    
    if width != DEFAULT_CANVAS_WIDTH:
        DEFAULT_CANVAS_WIDTH = width
        print(f'type(DEFAULT_FONTSIZE):{type(DEFAULT_FONTSIZE)}, type(X_RANGE):{type(X_RANGE)}')
        fs = int(DEFAULT_FONTSIZE)
        xr = int(X_RANGE)
        xr = int((width / 16.) * 13.0 + (fs/2.))
        #xr = int((width / (2 ** 4)) * 13 + (fs/2))
        #X_RANGE = int((width / (2 ** 4)) * 13 + (DEFAULT_FONTSIZE / 2))

    if height != None:
        DEFAULT_CANVAS_HEIGHT = height
        Y_RANGE = int((DEFAULT_CANVAS_HEIGHT / (2 ** 4)) * 13) + (DEFAULT_FONTSIZE /2 )
    
    DEFAULT_FONTSIZE = fontsize if fontsize != DEFAULT_FONTSIZE else DEFAULT_FONTSIZE
    FONTSIZE_RANGE = fontsize_range if fontsize_range != None else FONTSIZE_RANGE
    X0 = x0 if x0 != None else X0
    Y0 = y0 if y0 != None else Y0
    DEFAULT_ANCHOR = anchor if anchor != None else DEFAULT_ANCHOR
    
                       

def set_font(fontname='NotoSansJP-Bold',
             fontsize:int=None,
             default_font=None)->None:
    '''フォントを設定する'''
    
    global DEFAULT_FONTSIZE
    
    DEFAULT_FONTSIZE = fontsize if fontsize != DEFAULT_FONTSIZE else DEFAULT_FONTSIZE
    default_font = fontdata[fontname]['data']
    DEFAULT_FONT = ImageFont.truetype(default_font, size=DEFAULT_FONTSIZE)


def draw_text(draw:PIL.ImageDraw.ImageDraw=None,
              xy:tuple=(100,100),              # 主線の開始座標 x
              text:str=None,
              #font:PIL.ImageFont.FreeTypeFont=DEFAULT_FONT,
              fontsize:int=28,
              fill:tuple=(0,0,0),
              anchor:str=DEFAULT_ANCHOR,
              align:str=DEFAULT_ANCHOR,
              verbose=False,                    # 情報の表示
             )->tuple:

    print(f'xy:{xy}', f'font:{font}', f'fontsize:{fontsize}',
          f'text:{text}', f'fill:{fill}' ) if verbose else None

    draw.text(xy=xy, text=text, font=font, fill=fill, anchor=anchor, align=align)
    bbox = draw.textbbox(xy=xy,text=text,font=font, anchor=anchor, align=align)
    
    print(f'bbox:{bbox}') if verbose else None
    return bbox


def within_bboxes(bboxes:list,
                  bbox:tuple)->bool:
    """第 2 引数で与えられた bounding box の座標が，第一引数で与えら得た bounding boxes のリスト
    にある領域と重複していれば True していなければ False を返す"""
    
    left, top, right, bottom = bbox  # 第 2 引数の座標
    for _bbox in bboxes:
        
        # 判定用変数の初期化
        left_in, right_in, top_in, bottom_in = False, False, False, False
        width_in, height_in = False, False
        width_overlap, height_overlap = False, False
        
        _left, _top, _right, _bottom = _bbox
        
        if ((_left <= left ) and (left <= _right)):    left_in = True
        if ((_left <= right) and (right <= _right)):   right_in = True
        if ((top >= _top) and (top <= _bottom)):       top_in = True
        if ((bottom <= _bottom) and (bottom >= _top)): bottom_in = True
        if (left < _left) and (_right < right):        width_overlap = True
        if (top < _top) and (_bottom < bottom):        height_overlap = True
        if width_overlap and height_overlap:           return True

        if left_in and top_in:     return True         # 左上角が領域内
        if right_in and top_in:    return True         # 右上角が領域内
        if left_in and bottom_in:  return True         # 左下角が領域内
        if right_in and bottom_in: return True         # 右下角が領域内
    
        if left_in and right_in: width_in = True       # 横幅が領域内
        if top_in and bottom_in: height_in = True      # 縦長が領域内

        if width_in and height_in: return True         # 内側にある包含関係
    
        if width_in  and (top_in or bottom_in): return True
        if height_in and (left_in or right_in): return True
    
        if width_overlap and (top_in or bottom_in):  return True
        if height_overlap and (left_in or right_in): return True
    
    return False
            
    
def draw_strs(strs:list=symbols[1:],
              max_chars:int=len(symbols)-1,
              draw:PIL.ImageDraw.ImageDraw=None,
              fixed_fontsize:int=None,  # DEFAULT_FONTSIZE,
              fixed_color=None,
              anchor:str=DEFAULT_ANCHOR,
              x_range:int=X_RANGE,
              y_range:int=Y_RANGE,
              fontsize_range:int=FONTSIZE_RANGE,
              x0:int=X0,
              y0:int=Y0,
              verbose:bool=False): # ->tuple[PIL.ImageDraw.ImageDraw, dict, dict]:
              #verbose:bool=False)->tuple[PIL.ImageDraw.ImageDraw, dict, dict]:

    #global FONTSIZE_RANGE, DEFAULT_FONTSIZE, X_RANGE, Y_RANGE
    #global int_COLORS
    
    if draw == None:
        img, draw = make_a_canvas()
    bboxes = []
    #fontsize = np.random.randint(FONTSIZE_RANGE) + (DEFAULT_FONTSIZE - (FONTSIZE_RANGE>>1))
    var_font = False if fixed_fontsize != None else True
    var_color = False if fixed_color != None else True
    
    displayed, excluded = {}, {}
    for c in np.random.permutation(strs)[:max_chars]:
        fontsize = np.random.randint(fontsize_range) \
        + (DEFAULT_FONTSIZE - (fontsize_range>>1)) if var_font else fixed_fontsize
        color = int_COLORS[np.random.choice(len(int_COLORS))] if var_color else fixed_color
        
        x = np.random.randint(x_range) + x0
        y = np.random.randint(y_range) + y0
        xy = (x,y)
        text = c
        fontname = list(notofonts.keys())[np.random.choice(len(notofonts))]
        font_data = notofonts[fontname]['data']
        #print(f'text:{text} xy:{xy} font_name:{font_name}') if verbose else None
        _font = ImageFont.truetype(font=notofonts[fontname]['fname'], size=fontsize)
        bbox_cand = draw.textbbox(xy=xy, text=text, font=_font, anchor=anchor, align=anchor)

        if within_bboxes(bboxes, bbox_cand):
            excluded[c] = {'bbox':bbox, 
                           'color':color, 
                           'xy':xy, 
                           'fontname': fontname, 
                           'fontsize':fontsize,
                           'anchor': anchor,
                          }
        else:
            bbox = draw_text(draw=draw, xy=xy, text=text, font=_font, fill=color, 
                             anchor=anchor, align=anchor)
            displayed[c] = {'bbox':bbox, 'color':color, 'xy':xy, 'fontname': fontname,
                            'fontsize':fontsize, 'anchor':anchor}
            bboxes.append(bbox)
            draw.rectangle(bbox, outline="grey", width=2) if verbose else None
        
    return draw, displayed, excluded


def draw_strs_with_bboxes(draw:PIL.ImageDraw.ImageDraw=None,
                          bboxes:dict=None,
                         )->PIL.ImageDraw.ImageDraw:
    if draw == None:
        img, draw = make_a_canvas()
        
    for key, _bbox in bboxes.items():
        #_font = ImageFont.truetype(font=_bbox['fontname'], size=_bbox['fontsize'])
        fontname = _bbox['fontname']
        fontsize = _bbox['fontsize']
        _font = ImageFont.truetype(font=notofonts[fontname]['fname'], size=fontsize)
        #_font = ImageFont.truetype(font=_bbox['fontname'], size=_bbox['fontsize'])
        bbox = draw_text(draw=draw, xy=_bbox['xy'], text=key,  font=_font, 
                         fill=_bbox['color'], anchor=_bbox['anchor'], align=_bbox['anchor'])

    return draw

def calc_IoU_from2boxes(boxA, boxB):
    #def bb_intersection_over_union(boxA, boxB):
    """2 つのバウディングボックス間の IoU: Intersection of Union を計算
    各引数は，それぞれ４つの値をもつ tuple を仮定。
    4 つの値とは，左，上，右，下の座標値
    c.f.:https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    
    # 交差長方形の（x, y）座標を定める
    xA = max(boxA[0], boxB[0])  # A, B 左座標のうち大きい値
    yA = max(boxA[1], boxB[1])  # A, B 上座標のうち大きい値
    xB = min(boxA[2], boxB[2])  # A, B 右座標のうち小さい値
    yB = min(boxA[3], boxB[3])  # A, B 下座標のうち小さい値
    
    # 交差長方形領域の面積を計算
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # 両 bbox 領域の面積を計算
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # 交差領域の面積を　A と B の面積の合計（全面積）で割って，交差領域 IOU を計算
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou    

