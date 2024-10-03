import torch
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import cv2
from PIL import Image
import torchvision
import os
import extcolors
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torchvision.io import read_image 
import torch.nn as nn
import PIL
import pickle 
from patchify import patchify, unpatchify



class Clustering_loss():

    def __init__(self):
        self.CHECKPOINT_PATH='sam_vit_h_4b8939.pth'

        
        self.MODEL_TYPE = "vit_h"


        self.sam = sam_model_registry[self.MODEL_TYPE](checkpoint=self.CHECKPOINT_PATH).cuda()
        self.sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.mask_annotator = sv.MaskAnnotator(opacity = 1, color_lookup = sv.ColorLookup.INDEX)

        with open('saved_dictionary_centroid.pkl', 'rb') as f:
            self.final_dict = pickle.load(f)

        with open('saved_dictionary_centroidpatch2.pkl', 'rb') as f:
            self.final_dict2 = pickle.load(f)
        with open('saved_dictionary_centroidpatch3.pkl', 'rb') as f:
            self.final_dict3 = pickle.load(f)    





    def cal_loss(self, fake_B):

        fake_B_ = fake_B.clone()
        input_image = np.array(torchvision.transforms.functional.to_pil_image(fake_B_[0]))
        # input_image = fake_B.permute(0, 2, 3, 1).detach().cpu().numpy()[0].astype('uint8')
        output_mask = self.mask_generator.generate(input_image)
        detections = sv.Detections.from_sam(output_mask)
        annotated_image = self.mask_annotator.annotate(scene = input_image, detections = detections)
        im = Image.fromarray(annotated_image)
        colors, pixel_count = extcolors.extract_from_image(im)
        img_seg = np.asarray(im) 
        patches_seg = patchify(img_seg,(16, 16, 3), 6)

        im = Image.fromarray(input_image)
        img = np.asarray(im)
        patches = patchify(img, (16, 16, 3), 6)

        loss  = 0
        count = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                    im = Image.fromarray(patches_seg[i][j][0])
                    colors, pixel_count = extcolors.extract_from_image(im)
                    count+=1
                    for k in range(len(colors)):
                        if (colors[k][1]/ pixel_count) > 0.25:
                            if colors[k][0]  in self.final_dict:
                                   for m in self.final_dict[colors[k][0]]:
                                        patch_16 =  patches_seg[i][j][0].reshape(16*16*3)/255
                                        loss += np.mean(np.square(patch_16 - m/255))
                                       

                                # for m in self.final_dict.values():
                                #     for n in m:
                                #         patch_16 =  patches_seg[i][j][0].reshape(16*16*3)/255
                                #         loss+= np.mean(np.square(patch_16 - n/255))

                            # else:    

        patches_seg = patchify(img_seg,(8, 8, 3), 5)
        patches = patchify(img, (8, 8, 3), 5)
        loss1  = 0
        count1 = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                    im = Image.fromarray(patches_seg[i][j][0])
                    colors, pixel_count = extcolors.extract_from_image(im)
                    count1+=1

                    for k in range(len(colors)):
                        if (colors[k][1]/ pixel_count) > 0.25:
                            if colors[k][0]  in self.final_dict2:
                                   for m in self.final_dict2[colors[k][0]]:
                                        patch_16 =  patches_seg[i][j][0].reshape(8*8*3)/255
                                        loss1 += np.mean(np.square(patch_16 - m/255))  


                           

        patches_seg = patchify(img_seg,(4, 4, 3), 4)
        patches = patchify(img, (4, 4, 3), 4)
        loss2  = 0
        count2 = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                    im = Image.fromarray(patches_seg[i][j][0])
                    colors, pixel_count = extcolors.extract_from_image(im)
                    count2+=1
                    for k in range(len(colors)):
                        lossm = 99999
                        if (colors[k][1]/ pixel_count) > 0.25:
                            if colors[k][0]  in self.final_dict3:
                                   for m in self.final_dict3[colors[k][0]]:
                                        patch_16 =  patches_seg[i][j][0].reshape(4*4*3)/255
                                        loss2 += np.mean(np.square(patch_16 - m/255))  

                                            # print(lossm,'lossm3')

        print(loss/count + loss1/count1 + loss2/count2)
        return loss/count + loss1/count1 + loss2/count2                          