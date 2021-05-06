import numpy as np
import os
import torch
import struct
from torch.utils.data import Dataset
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
from utils import pc_normalize, random_select_points, shift_point_cloud, \
    jitter_point_cloud, generate_random_rotation_matrix, \
    generate_random_tranlation_vector, transform, random_crop


class ModelNet40Peng(Dataset):
    def __init__(self, root, npts, train):
        super(ModelNet40Peng, self).__init__()
        self.npts = npts
        self.train = train
        # files = [os.path.join(root, 'ply_data_train{}.h5'.format(i))
        #          for i in range(5)]
        # if not train:
        #     files = [os.path.join(root, 'ply_data_test{}.h5'.format(i))
        #              for i in range(2)]
        self.files = self.read_filename(root)
        # print(self.files[0])

    # def decode_h5(self, files):
    #     points, normal, label = [], [], []
    #     for file in files:
    #         f = h5py.File(file, 'r')
    #         cur_points = f['data'][:].astype(np.float32)
    #         cur_normal = f['normal'][:].astype(np.float32)
    #         cur_label = f['label'][:].astype(np.float32)
    #         points.append(cur_points)
    #         normal.append(cur_normal)
    #         label.append(cur_label)
    #     points = np.concatenate(points, axis=0)
    #     normal = np.concatenate(normal, axis=0)
    #     data = np.concatenate([points, normal], axis=-1).astype(np.float32)
    #     label = np.concatenate(label, axis=0)
    #     return data, label

    def read_filename(self, root):
        # files = os.listdir(path)
        # file_name = list()
        # for file in files:
            # if not os.path.isdir(path + file):  #判断该文件是否是一个文件夹
            # f_name = file.zfill(11)
            # new_name = ''.join(f_name)
            # print(new_name)
            # os.rename(path + '/' + file, path + '/' + new_name)
        if self.train is True:
            # print('root:' + str(root))
            path_ego = os.path.join(root, 'train/cloud_ego')
            path_coop = os.path.join(root, 'train/cloud_coop')
        else:
            path_ego = os.path.join(root, 'test/cloud_ego')
            path_coop = os.path.join(root, 'test/cloud_coop')
        files_ego = os.listdir(path_ego)
        dirpath_list_all = list()
        files_ego.sort()
        for file in files_ego:
            dirpath_list = list()
            dirpath_list.append(os.path.join(path_ego, file))
            file_name_num = str(file).split('.')[0]
            # print(path_coop, file_name_num)
            files_coop = os.listdir(path_coop + '/' + file_name_num)
            for file_coop in files_coop:
                dirpath_list.append(os.path.join(path_coop, file_name_num, file_coop))
            dirpath_list_all.append(dirpath_list)
        return dirpath_list_all

    def read_bin_velodyne(self, path):
        pc_list=[]
        with open(path,'rb') as f:
            content=f.read()
            pc_iter=struct.iter_unpack('ffff',content)
            for idx,point in enumerate(pc_iter):
                pc_list.append([point[0],point[1],point[2]])
        # print(len(pc_list))
        return np.asarray(pc_list,dtype=np.float32)

        
    def read_pointcloud_initially_filtered(self, dir_path_list, height_ground=-1.75, height_sky=5):
    # def read_pointcloud_initially_filtered(dir_path_list, height_ground=-50, height_sky=50):
        pcd_list = list()
        for i in range(len(dir_path_list)):
            # print(len(dir_path_list))
            point_cloud = self.read_bin_velodyne(dir_path_list[i])
            new_pc = list()
            for i in range(point_cloud.shape[0]):
                if point_cloud[i][2] > height_ground and point_cloud[i][2] < height_sky:
                    new_pc.append(point_cloud[i][:3])
                    # new_pc = np.asarray(new_pc,dtype=np.float32)
            # pcd = o3d.open3d.geometry.PointCloud()
            # pcd.points = o3d.open3d.utility.Vector3dVector(np.array(new_pc))
            new_pc = np.asarray(new_pc,dtype=np.float32)
            pcd_list.append(new_pc)
        return pcd_list

    def __getitem__(self, item):
        file = self.files[item]
        # print(item, len(file))
        # print(self.read_pointcloud_initially_filtered(file[0])[0])
        ref_cloud = self.read_pointcloud_initially_filtered([file[0]])[0]
        src_cloud = self.read_pointcloud_initially_filtered([file[1]])[0]
        ref_cloud = random_select_points(ref_cloud, m=self.npts)
        src_cloud = random_select_points(src_cloud, m=self.npts)
        R, t = generate_random_rotation_matrix(0, 0), \
               generate_random_tranlation_vector(-0.5, 0.5)
        ref_cloud = transform(ref_cloud, R, t)
        if self.train:
            ref_cloud = jitter_point_cloud(ref_cloud)
            src_cloud = jitter_point_cloud(src_cloud)
        return ref_cloud, src_cloud, R, t

    def __len__(self):
        return len(self.files)