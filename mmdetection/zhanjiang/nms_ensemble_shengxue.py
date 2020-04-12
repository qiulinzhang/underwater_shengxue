import pandas as pd
import os
import tqdm
import networkx as nx
import torch
import numpy as np
import json
from mmdet.ops.nms.nms_wrapper import nms, soft_nms
#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

underwater_classes = {'target': 1}
#underwater_classes = {'holothurian':1, 'echinus':2, 'scallop':3, 'starfish':4}
score_thr = 0.0001

res1 = pd.read_csv("./temp_results/dcn_r50_0407_vote_500_wb_shengxue.csv")
res1 = res1[res1["confidence"]>=score_thr]
print(res1.shape)

res2 = pd.read_csv("./temp_results/dcn_r50_0407_vote_800_wsf_wb_shengxue.csv")
res2 = res2[res2["confidence"]>=score_thr]
print(res2.shape)

res3 = pd.read_csv("./temp_results/dcn_r50_0406_vote_800_wb_shengxue.csv")
res3 = res3[res3["confidence"]>=score_thr]
print(res3.shape)
img_raw_info = json.load(open('/home/mtc206/new_ssd/zql/data/zhanjiang/shengxue/a-test-image/test_A_shengxue.json', "r"))

name2id = {}
for img in img_raw_info['images']:
    name2id[img['file_name'][:-4]+'.xml']=img['id']

deal_nms = pd.concat([res1, res2, res3])

# deal_nms = pd.concat([res1, res2, res3])

test_path = '/media/vip/Data2/wusaifei/Under_DEC/shengxue/mmdetection/data/coco/concat_train/test_A'  # 官方测试集图片路径

json_name = "./temp_results/test.bbox.json"
print(deal_nms.shape)
result = []
images = []
#lis = {}
jl = 0




for filename in tqdm.tqdm(deal_nms['image_id'].unique()):
    img_id = name2id[filename]
    base_dets = deal_nms[deal_nms['image_id']==filename]
    #lis[jl] = filename
    for defect_label, value in underwater_classes.items(): # 查找标签，对应标签进行融合
        base_dets_1 = base_dets[base_dets['name'] == defect_label]
        dets = torch.FloatTensor(np.array(base_dets_1[['xmin','ymin','xmax','ymax','confidence']])).cuda()

        iou_thr = 0.5

        # surpressed, inds = nms(dets, iou_thr)
        surpressed, inds = soft_nms(dets, iou_thr)

        for press in surpressed:
            x1, y1, x2, y2, score = press.cpu().numpy()[:]
            x1, y1, x2, y2 = round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)  # save 0.00
            result.append(
                {'image_id': img_id, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'category_id': int(value), 'score': float(score)})


#for k in lis.keys():
#    lis_img = {}
#    lis_img['image_id'] = lis[k]
#    lis_img['id'] = k + 1
#    images.append(lis_img)

submit = result
with open(json_name, 'w') as fp:
    # json.dump(result, fp, indent=4, separators=(',', ': '))
    json.dump(submit,fp, separators=(',', ': '))
