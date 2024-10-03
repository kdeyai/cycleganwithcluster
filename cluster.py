import torch
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import cv2
from PIL import Image
import os
import extcolors
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torchvision.io import read_image 
import torch.nn as nn
import PIL
import pickle 
from patchify import patchify, unpatchify
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



# # segcolors = []
# # count = 0
# # for i in os.listdir('/home/ishika/kaushik/art2real/images/segmentation'):
# #       im = PIL.Image.open("/home/ishika/kaushik/art2real/images/segmentation/"+i).convert('RGB')
# #       colors, pixel_count = extcolors.extract_from_image(im)
# #       for j in range(len(colors)):
# #         segcolors.append(colors[j][0]) 

# #       print(count)  
# #       count+=1  


# # segkeys = list(set(segcolors))
# # with open("test", "wb") as fp:  
# #      pickle.dump(segkeys, fp)

# with open("test", "rb") as fp: 
#      segkeys = pickle.load(fp)

# imgdict = {key: [] for key in segkeys}
# # # unfold = nn.Unfold(kernel_size=(16, 16), stride = 6)
# # # fold = nn.Fold(output_size = (256, 256), kernel_size=(16, 16), stride = 6)
# # count1 = 0
# count = 0
# for z in os.listdir('/home/ishika/kaushik/art2real/images/segmentation'):
#     img_seg = (Image.open('/home/ishika/kaushik/art2real/images/segmentation/'+z).convert('RGB')) 
#     img_segarr = np.asarray(img_seg)
#     patches_seg = patchify(img_segarr,(8, 8, 3), 5)
#     count+=1

#     img = (Image.open('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainB/'+z).convert('RGB')) 
#     img_arr = np.asarray(img) 
#     patches = patchify(img_arr,(8, 8, 3), 5)


#     for i in range(patches.shape[0]):
#         for j in range(patches.shape[1]):
#             im = Image.fromarray(patches_seg[i][j][0])
#             colors, pixel_count = extcolors.extract_from_image(im)
#             for j in range(len(colors)):
#                 if (colors[j][1]/ pixel_count) > 0.25:
#                     # if colors[j][0] not in imgdict:
#                     #     count1+=1
#                     #     imgdict[colors[j][0]] = []
#                     #     imgdict[colors[j][0]].append(patches[i][j][0])

#                     # else:
#                     if colors[j][0] not in imgdict:
#                         imgdict[colors[j][0]] = []
#                         imgdict[colors[j][0]].append(patches[i][j][0])
#                     else:
#                         imgdict[colors[j][0]].append(patches[i][j][0])


# # print(count1)
# # print(count)
# with open('saved_dictionary22.pkl', 'wb') as f:
#     pickle.dump(imgdict, f)
        
with open('saved_dictionary22.pkl', 'rb') as f:
    img_dict = pickle.load(f)
    # print('patches', patches.shape)
    # imageseg = read_image('/home/ishika/kaushik/art2real/images/segmentation/'+i)
    # imageseg = imageseg.to(torch.float32)
    # image = read_image('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainB/'+i)
    # image = image.to(torch.float32)

    # outputseg = unfold(imageseg)
    # outputimg = unfold(image)

    # # im = Image.fromarray((fold(outputseg).permute(1,2,0).cpu().numpy()).astype(np.uint8))
    # # im.save("img.png")

    # # im = Image.fromarray((imageseg.permute(1,2,0).cpu().numpy()).astype(np.uint8))
    # # im.save("img.png")
    # # break
    # listseg =  outputseg.permute(1,0).tolist()
    # listimg = outputimg.permute(1,0).tolist()
    # print(len(listseg[0]))
    # cnt = 0
    # for m in listseg:
    #     print(torch.Tensor(m).view(3, 16, 16).size())
    #     im = Image.fromarray((torch.Tensor(m).view(3, 16, 16).permute(1, 2 ,0).cpu().numpy()).astype(np.uint8))
    #     im.save("img.png")
    #     colors, pixel_count = extcolors.extract_from_path("img.png")
    #     for j in range(len(colors)):
    #           if (colors[j][1]/ pixel_count) > 0.25:
    #                imgdict[colors[j][0]].append(listimg[cnt])

    #     cnt+=1



final_dict = {key: [] for key in img_dict.keys()}

count = 0
for keyvalues, imgvalues in zip(img_dict.keys(), img_dict.values()):
    count+=len(imgvalues)

print(count)
for keyvalues, imgvalues in zip(img_dict.keys(), img_dict.values()):
    if len(imgvalues) <= 2:
        continue;
    else:
        print(len(imgvalues))
        M = np.stack(imgvalues).reshape(len(imgvalues), 8*8*3)
        print('numpy', M.shape)
        print(imgvalues[0].shape)

    minimum =  -99999
    opt = 0
    for v in range(2, 30):
        if v >= len(imgvalues):
            break;

        kmean = KMeans(n_clusters = v, random_state = 0, n_init = 'auto')
        kmean.fit_predict(M)
        if kmean.labels_.shape[0] > 1:
            print(kmean.labels_.shape[0],'h')
            score = silhouette_score(M, kmean.labels_, metric='euclidean')
            if score > minimum:
                minimum = score
                opt = v

    
    print(opt)
    kmean = KMeans(n_clusters = opt, random_state = 0, n_init = 'auto')
    kmeans = kmean.fit(M)
    
    final_dict[keyvalues] = kmeans.cluster_centers_



with open('saved_dictionary_centroidpatch22.pkl', 'wb') as f:
    pickle.dump(final_dict, f)
    
with open('saved_dictionary_centroidpatch22.pkl', 'rb') as f:
    final_dict = pickle.load(f)

# CHECKPOINT_PATH='sam_vit_h_4b8939.pth'

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# MODEL_TYPE = "vit_h"


# sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
# sam.eval()
# mask_generator = SamAutomaticMaskGenerator(sam)

# for i in os.listdir('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainB'):
#     image = cv2.imread('/home/ishika/kaushik/art2real/datasets/landscape2photo/trainB/'+i)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     output_mask = mask_generator.generate(image_rgb)
#     mask_annotator = sv.MaskAnnotator(opacity = 1, color_lookup = sv.ColorLookup.INDEX)
#     detections = sv.Detections.from_sam(output_mask)
#     annotated_image = mask_annotator.annotate(scene = image, detections = detections)
#     im = Image.fromarray(annotated_image)
#     im.save('datasets/landscape2photo/segmentation/'+i)


# # Generate segmentation mask
# output_mask = mask_generator.generate(image_rgb)


