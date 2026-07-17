#!/usr/bin/env python
# coding: utf-8

# In[9]:


# image generator

import math
import pandas as pd
import torch
import os
import glob
import openpyxl
import torch
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import PIL
import os
import math



#encode the atoms present in the datset as points on a circle in R2
def atoms_encoding(atom):
    present_atoms = ["C","H","O","N","S","F","Cl","Br","I","P"]
    no_of_present_atoms = len(present_atoms)
    index = present_atoms.index(atom)
    return [math.cos(2*torch.pi*index/no_of_present_atoms),math.sin(2*torch.pi*index/no_of_present_atoms)]



max_point_mag = 10


#deleting all images in a directory
def delete_images_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff')):
            filepath = os.path.join(directory, filename)
            os.remove(filepath)



#rotate a point
def rotate(x,y,z):
    R_x = torch.tensor([
    [1, 0, 0],
    [0, torch.cos(x), -torch.sin(x)],
    [0, torch.sin(x), torch.cos(x)]
    ])
    R_y = torch.tensor([
    [torch.cos(y), 0, torch.sin(y)],
    [0, 1, 0],
    [-torch.sin(y), 0, torch.cos(y)]
    ])
    R_z = torch.tensor([
    [torch.cos(z), -torch.sin(z), 0],
    [torch.sin(z), torch.cos(z), 0],
    [0, 0, 1]
    ])
    return R_z @ R_y @ R_x


#project any 3d point to a 2d one
def project_to_isometric(t):
    return torch.stack([((3)**0.5)/2*(t[:,0]-t[:,1]), 0.5*(t[:,0]+t[:,1])-t[:,2]], dim=1)




#creating the grid of pixels
def process_data(data,zoom_val=0.05,plot = True,naming_index = 0,rotation_angles = torch.tensor([0.0, 0.0, 0.0]),rad=(0,1),format = "full",atom_type=1,metaball = False,res=400):
    datafr = data.copy()
    datafr['Atom_names'] = datafr['Atom_names'].map(atoms_encoding)
    datafr[['atom_cos','atom_sin']] = pd.DataFrame(
    datafr['Atom_names'].tolist(), index=datafr.index
    )
    datafr = datafr.drop(columns=['Atom_names'])
    data_tensor = torch.tensor(datafr.values, dtype=torch.float32)

    data_tensor[:,0:3] = data_tensor[:,0:3]@rotate(rotation_angles[0], rotation_angles[1], rotation_angles[2]).T
    projected = project_to_isometric(data_tensor)
    max_of_any_coord = torch.max(data_tensor[:,0:2].flatten())
    min_of_any_coord = torch.min(data_tensor[:,0:2].flatten())

    zoom = zoom_val * res
    projected = zoom * projected + (res/2)


    grid = torch.ones((3,res, res), dtype=torch.float32)

    distance_to_camera = torch.stack([data_tensor[:,0]-max_point_mag, data_tensor[:,1]-max_point_mag, data_tensor[:,2]-max_point_mag],dim=1)
    magnitude = torch.sqrt(torch.sum(distance_to_camera**2, dim=1))

    #colours
    red_channel = (data_tensor[:,3]-min(data_tensor[:,3]))/(max(data_tensor[:,3])-min(data_tensor[:,3]))
    green_channel = (data_tensor[:,4]-min(data_tensor[:,4]))/(max(data_tensor[:,4])-min(data_tensor[:,4]))
    blue_channel = (magnitude-min(magnitude))/(max(magnitude)-min(magnitude))
    colours = torch.stack([red_channel, green_channel, blue_channel], dim=1)

    distances_3d = torch.cdist(data_tensor[:, 0:3], data_tensor[:, 0:3])
    distances_3d.fill_diagonal_(float('inf'))
    min_3d_values, _ = torch.min(distances_3d, dim=1)

    sort_idx = torch.argsort(magnitude, descending=True)




    x = projected[:, 0].round().long()
    y = projected[:, 1].round().long()
    r = (min_3d_values/2) * zoom
    x = x[sort_idx]
    y = y[sort_idx]
    r = r*1.28
    r = r[sort_idx].round().long()
    colours = colours[sort_idx]
    magnitude = magnitude[sort_idx]




    if metaball == True:
        number_of_points = 100
        lin = torch.linspace(min_of_any_coord,max_of_any_coord,number_of_points)
        test = torch.cartesian_prod(lin,lin,lin)
        distances = torch.cdist(test,data_tensor[:,0:3],p=2)
        dis_inv = 1.0 / (distances + 1e-8)
        dis_sum = torch.sum(dis_inv, dim=1).flatten()


        grid_stamp = project_to_isometric(test)
        grid_stamp = zoom * grid_stamp + (res/2)
        grid_stamp = grid_stamp.round().long()

        mask = dis_sum > 8.5
        coords = grid_stamp[mask]
        valid = (coords[:,0] >= 0) & (coords[:,0] < res) & (coords[:,1] >= 0) & (coords[:,1] < res)
        coords = coords[valid]
        grid[:, coords[:,1], coords[:,0]] = 0.0

        cross = True
        if cross == True:
            cross_coords = coords.clamp(min = 1,max=res-2)
            grid[:, cross_coords[:,1], cross_coords[:,0]] = 0.0
            grid[:, cross_coords[:,1] + 1, cross_coords[:,0]] = 0.0
            grid[:, cross_coords[:,1] - 1, cross_coords[:,0]] = 0.0
            grid[:, cross_coords[:,1], cross_coords[:,0] - 1] = 0.0
            grid[:, cross_coords[:,1], cross_coords[:,0] + 1] = 0.0






    for i in range(len(x)):
        cx, cy, radius = x[i].item(), y[i].item(), r[i].item()
        color = colours[i]
        if rad[0] == 1:
            radius = rad[1]
        else:
            radius = r[i].item()


        #Calculate a bounding box around the circle
        x_min = max(0, cx - radius)
        x_max = min(res - 1, cx + radius)
        y_min = max(0, cy - radius)
        y_max = min(res - 1, cy + radius)

        if x_min >= x_max or y_min >= y_max:
            continue

        # Generate local pixel coordinates
        y_range = torch.arange(y_min, y_max + 1)
        x_range = torch.arange(x_min, x_max + 1)
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')

        #Create a boolean circle mask based on the distance formula
        circle_mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= radius**2
        #Paint the pixels across the color channels
        grid[:, y_grid[circle_mask], x_grid[circle_mask]] = color.unsqueeze(1)
    if atom_type == 1:
        for i in range(len(x)):
            cx, cy = x[i].item(), y[i].item()
            idx = sort_idx[i].item()
            color = torch.tensor([(data_tensor[idx,5]+1)/2, (data_tensor[idx,6]+1)/2, 0.])
            grid[:, cy-10:cy+10, cx-10:cx+10] = color[:, None, None]
    return grid












#Generate many images
def generate_images(data,number_of_images,speed,rad,atom_type,metaball,res=400,zoom=0.05):
    video_tensor = torch.zeros(number_of_images,3,res,res)
    for i in range(number_of_images):
        video_tensor[i,:,:,:] = process_data(data=data,plot=False,naming_index=i,rotation_angles = torch.tensor([i*0.01*speed,0,-0.01*i*speed]),rad=rad,atom_type=atom_type,metaball=metaball,res=res,zoom_val=zoom)
    return(video_tensor)






def generate_random_rotation_angles():
    u1 = torch.rand(1)
    u2 = torch.rand(1)
    u3 = torch.rand(1)

    theta = u1 * 2 * torch.pi
    phi = torch.acos(2 * u2 - 1)
    psi = u3 * 2 * torch.pi
    return torch.tensor([theta, phi, psi])












number_of_images_per_molecule = 20

def make_training_tensor(path_to_decoys, path_to_ligand, number_of_images_per_molecule,training_or_validation):

    total_items = len(os.listdir(path_to_decoys))+len(os.listdir(path_to_ligand))
    number_of_ligs = len(os.listdir(path_to_ligand))
    number_of_decoys = len(os.listdir(path_to_decoys))


    data_tensor = torch.zeros((total_items,number_of_images_per_molecule ,3,100,100), dtype=torch.float32)
    fileList_to_decoys = os.listdir(path_to_decoys)
    all_paths = [path_to_decoys, path_to_ligand]
    output_tensor = torch.zeros((total_items,number_of_images_per_molecule ,3,100,100), dtype=torch.float32)
    counter = 0
    for folder_path in all_paths:
        file_list = os.listdir(folder_path)
        for f in file_list:
            full_path = os.path.join(folder_path, f)
            print('Location:', full_path)
            print('File Name:', os.path.basename(full_path))
            if not full_path.endswith('.xlsx'):
                print(f"Skipping {full_path} as it is not an Excel file.")
                continue
            df = pd.read_excel(full_path, engine='openpyxl')
            for j in range(number_of_images_per_molecule):
                output_tensor[counter,j,:,:,:] = process_data(data = df,
                    plot = True,
                    naming_index = 0,
                    rotation_angles = generate_random_rotation_angles().clone().detach(),
                    rad=(0,1),
                    format = "full",
                    atom_type=0,
                    metaball = False,
                    res=100,
                    zoom_val=0.03
                )
            counter += 1

    labels = torch.cat([torch.zeros(number_of_decoys,number_of_images_per_molecule), torch.ones(number_of_ligs,number_of_images_per_molecule)],dim=0)
    if training_or_validation == 'training':
        output_tensor = output_tensor.squeeze(1)
        torch.save(labels, 'training_labels.pt')
        torch.save(output_tensor, 'training_tensor.pt')
        print(f"Training tensor shape: {output_tensor.shape}, Labels shape: {labels.shape}")
        print("Training data generated")
    else:
        output_tensor = output_tensor.squeeze(1)
        torch.save(labels, 'validation_labels.pt')
        torch.save(output_tensor, 'validation_tensor.pt')
        print(f"Validation tensor shape: {output_tensor.shape}, Labels shape: {labels.shape}")
        print("Validaiton data generated")


path_to_decoys_training = 'dude_test_train_split/train/decoys/'
path_to_ligand_training = 'dude_test_train_split/train/ligands/'

path_to_decoys_validation = 'dude_test_train_split/validation/decoys/'
path_to_ligand_validation = 'dude_test_train_split/validation/ligands/'


make_training_tensor(path_to_decoys_validation, path_to_ligand_validation, number_of_images_per_molecule,training_or_validation ="validation")












# In[7]:


create a test train split, insert the filenames into the function and download


# In[ ]:




