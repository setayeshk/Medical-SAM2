import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import nibabel as nib

from func_3d.utils import random_click, generate_bbox

class BRATS(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', seed=None, variation=0):

        # Set the data list for training
        
        self.name_list = os.listdir(data_path)
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None

        # برای اون دو تا فولدر csv 
        self.previous_data = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1  #1+ 0- -1padding modeling sam2base.py comment
        newsize = (self.img_size, self.img_size)
        print("modee :", self.mode)
        """Get the images"""
        name = self.name_list[index]
        # print("name: ", name )
        if "BraTS" in name:           

            img_path = os.path.join(self.data_path,name, name +"_t1ce.nii")
            mask_path = os.path.join(self.data_path, name, name + "_seg.nii")
        

            mask = nib.load(mask_path)
            data_seg_3d =  np.array(mask.get_fdata())
            
            
            img_3d = nib.load(img_path)
            img_3d =  np.array(img_3d.get_fdata())

            for i in range(data_seg_3d.shape[-1]):
                if np.sum(data_seg_3d[..., i]) > 0:
                    data_seg_3d = data_seg_3d[..., i:]
                    break
            starting_frame_nonzero = i

            for j in reversed(range(data_seg_3d.shape[-1])):
                if np.sum(data_seg_3d[..., j]) > 0:
                    data_seg_3d = data_seg_3d[..., :j+1]
                    break
            num_frame = data_seg_3d.shape[-1]

            # print("img shape :", img_3d.shape)
            # ا
            if self.video_length is None:
                video_length = int(num_frame / 4)
            else:
                video_length = self.video_length
            # video_length=20

            if num_frame > video_length and self.mode == 'Training':
                starting_frame = np.random.randint(0, num_frame - video_length + 1)
            else:
                starting_frame = 0
            img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
            mask_dict = {}
            point_label_dict = {}
            pt_dict = {}
            bbox_dict = {}
            # print("video lenght: ", video_length)
            for frame_index in range(starting_frame, starting_frame + video_length):
                # print("frame_index ", frame_index)
            
                img = img_3d[...,frame_index + starting_frame_nonzero]
                
                mask = data_seg_3d[..., frame_index]

                img = Image.fromarray(img).convert('RGB')
                obj_list = np.unique(mask[mask > 0])
                diff_obj_mask_dict = {}
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict = {}
                elif self.prompt == 'click':
                    diff_obj_pt_dict = {}
                    diff_obj_point_label_dict = {}
                else:
                    raise ValueError('Prompt not recognized')
                for obj in obj_list:
                    obj_mask = mask == obj
                    obj_mask = Image.fromarray(obj_mask)
                    obj_mask = obj_mask.resize(newsize)
                    
                    obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()

                    diff_obj_mask_dict[obj] = obj_mask

                    if self.prompt == 'click':
                        diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=None)
                    if self.prompt == 'bbox':
    
                        diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)

                img = img.resize(newsize) 

        
                img = torch.tensor(np.array(img)).permute(2, 0, 1)

                img_tensor[frame_index - starting_frame, :, :, :] = img

    
                mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
                if self.prompt == 'bbox':
                    bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
                elif self.prompt == 'click':
                    pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                    point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict

            print("image tensor :" , img_tensor.shape)
            image_meta_dict = {'filename_or_obj':name}
            if self.prompt == 'bbox':
                previous_data = {
                    'image':img_tensor,
                    'label': mask_dict,
                    'bbox': bbox_dict,
                    'image_meta_dict':image_meta_dict,
                }
            elif self.prompt == 'click':
                previous_data = {
                    'image':img_tensor,
                    'label': mask_dict,
                    'p_label':point_label_dict,
                    'pt':pt_dict,
                    'image_meta_dict':image_meta_dict,
                }
            return previous_data
        else:
            if self.previous_data is not None:
                return self.previous_data
            else:
                raise ValueError("No previous data available to repeat.")
