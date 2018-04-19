
import os
import sys
import time
import math
import random
import collections

import numpy as np
import scipy.signal as ssignal
import scipy.io as sio
import pickle
import re

from sklearn.cluster import KMeans
import tensorflow as tf
if tf.__version__.startswith("1.2"):
  from tensorflow.python.util import nest
elif tf.__version__.startswith("1.4"):
  from tensorflow.contrib.framework import nest
else:
  raise ValueError("Cannot locate tensorflow 'nest' package!")

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D, proj3d
# from IPython import display

np.seterr(all='raise')

NOISE_LEVEL = [
    100., # x axis
    5.,  # y axis
    100., # z axis
]

def differentiate(x, delta_t):
  """ test input:
        array([[0., 1.],
              [1., 2.],
              [3., 3.]]), 
        0.5
      output:
        array([[0., 0.],
              [2., 2.],
              [4., 2.]), 
        array([0., 1.])
  """
  diff_x = (x[1:, :] - x[:-1, :]) / delta_t
  diff_x = np.concatenate((np.zeros((1, diff_x.shape[1])), diff_x), axis=0)
  start_x = x[0, :]
  return diff_x, start_x 

def integral(x, start_x, delta_t):
  """ test input:
        array([[0., 0.],
              [2., 2.],
              [4., 2.]]), 
        array([0., 1.]),
        0.5
      output:
        array([[0., 1.],
              [1., 2.],
              [3., 3.]])
  """
  integral_x = np.cumsum(x, axis=0) * delta_t + start_x
  # integral_x = np.vstack((start_x, integral_x))
  return integral_x

def interp_nan(col):
  """
      test input:
        mat = np.asarray([[1.0, 2.0], [nan, nan], [3.0, 4.0]])
        interp_nan(mat[:, 0])
      expected return:
        array([ 1.,  2.,  3.])
  """
  nan_loc = np.isnan(col)
  get_index = lambda bool_idx: np.argwhere(bool_idx).T[0]
  col[nan_loc] = np.interp(get_index(nan_loc), get_index(~nan_loc), col[~nan_loc])
  return col

def interp_nan_2d(data):
  """ linearly interpolate NaNs in data
      data is of 2D, [time, dim]
      interpolate on the 1st dim
  """
  for i in range(data.shape[1]):
    data[:, i] = interp_nan(data[:, i])
  return data

def numpy_fillzeros(data, sublength=None, fill_val=0):
    """ Reshape a list of 2D arrays to a 3D array.
        Insufficient length arrays with be filled with 0s.
        test input:
            [np.array([[1., 1., 1.]]), 
             np.array([[2., 2., 0.], [3., 3., 1.]])]
        expected output:
            np.array([[[1., 1., 1.],
                      [0., 0., 0.]],
                     [[2., 2., 0.],
                      [3., 3., 1.]]])
            np.array([[[1.],
                      [0.]],
                     [[1.],
                      [1.]]])
    """
    mask_dim = data[0].shape[1]
    # get the lengths of all arrays
    lens = np.array([ d.shape[0] for d in data]) # [batch]
    # mask out the insufficient part
    if sublength is None:
      max_len = lens.max()
    else:
      max_len = int(np.ceil(float(lens.max()) / sublength) * sublength)
    mask2d = np.arange(max_len) < lens[:, None] # [batch, max_len]
    mask = np.repeat(mask2d[:, :, None], mask_dim, axis=2) # [batch, max_len, mask_dim]
    # setup a matrix of the right size and fill in the values
    out = np.ones(mask.shape) * fill_val # [batch, max_len, mask_dim]
    out[mask] = np.concatenate(data).flatten()
    return out, mask2d[:,:,None].astype(np.float32), lens

# noramlize & rotate
def print_data_properties(trajs_p1):
  min_val = np.min([ np.min(t, axis=0) for t in trajs_p1 ], axis=0)
  max_val = np.max([ np.max(t, axis=0) for t in trajs_p1 ], axis=0)
  average = np.mean(np.concatenate(trajs_p1, axis=0), axis=0)
  std = np.std(np.concatenate(trajs_p1, axis=0), axis=0)
  print("Min.X %.4f, Max.X %.4f, Average.X %.4f, Std.X %.4f" 
        % (min_val[0], max_val[0], average[0], std[0]))
  print("Min.Y %.4f, Max.Y %.4f, Average.Y %.4f, Std.Y %.4f" 
        % (min_val[1], max_val[1], average[1], std[1]))
  print("Min.Z %.4f, Max.Z %.4f, Average.Z %.4f, Std.Z %.4f" 
        % (min_val[2], max_val[2], average[2], std[2]))
  return average

def plot_trajectory(trajs_p1, trajs_p2, case_label, case):

  traj_1_set = [trajs_p1[i] for i in range(len(trajs_p1)) if case_label[i] == case ]
  traj_2_set = [trajs_p2[i] for i in range(len(trajs_p2)) if case_label[i] == case ]
  traj_1 = np.concatenate(traj_1_set, axis=0)
  traj_2 = np.concatenate(traj_2_set, axis=0)
  print(traj_1.shape)
  plt.plot(traj_1[:, 0], traj_1[:, 1], c='b', label='Person 1')
  plt.plot(traj_2[:, 0], traj_2[:, 1], c='g', label='Person 2')
  for t in traj_1_set:
    plt.scatter(t[-1, 0], t[-1, 1], c='k')
  plt.legend()
  plt.show()

def make_every_traj_zero_mean(trajs):
  trajs = [ t - np.mean(t, axis=0) for t in trajs ]
  return trajs


def filter_cases(trajs_p1, trajs_p2, case_label, keep_cases):
  trajs_p1 = [trajs_p1[i] for i in range(len(trajs_p1)) if case_label[i] in keep_cases]
  trajs_p2 = [trajs_p2[i] for i in range(len(trajs_p2)) if case_label[i] in keep_cases]
  case_label = [case_label[i] for i in range(len(case_label)) if case_label[i] in keep_cases]
  return trajs_p1, trajs_p2, case_label


def make_every_traj_start_the_same(trajs_p1, trajs_p2):
  new_p1 = []
  new_p2 = []
  for i in range(len(trajs_p1)):
    offset = trajs_p1[i][0, :]
    new_p1.append(trajs_p1[i] - offset)
    new_p2.append(trajs_p2[i] - offset)
  return new_p1, new_p2


def divide_forward_backward(end_points, kmeans):
  end_points = np.array(end_points)
  labels = kmeans.predict(end_points)
  # plt.scatter(end_points[labels == 0, 0], end_points[labels == 0, 1], c='b')
  # plt.scatter(end_points[labels == 1, 0], end_points[labels == 1, 1], c='r')
  # plt.show()
  return labels


def rotate_dataset(trajs):
  trajs_rotate = trajs[:]
  for i in range(3):
    # rotate 90 degree clockwise: x, y = -y, x
    trajs = [ np.array([-t[:, 1], t[:, 0], t[:, 2]]).T for t in trajs ]
    trajs_rotate.extend(trajs)

  trajs = trajs_rotate
  return trajs


def repulsive_force(obs_rel):
  if len(obs_rel.shape) == 1:
    single_dim = True
    obs_rel = obs_rel[None, :]
  else:
    single_dim = False
  dist_hand_obs = np.sqrt(np.sum(np.square(obs_rel), axis=1))
  dist_hand_obs = dist_hand_obs[:, None]
  dist_star = 0.4
  # flip sign
  sign = np.ones((len(dist_hand_obs), 1))
  sign[dist_hand_obs > dist_star] = 0

  obs_repulsive = (1. / dist_star - 1. / dist_hand_obs) \
                  * (1. / dist_hand_obs) * obs_rel * sign
  if not single_dim:
    return obs_repulsive
  else:
    return obs_repulsive[0, :]


class DataLoader():
  def __init__(self, batch_size=50, scale_factor=1000., 
      subseq_length=None, filter_size=11, joints=[["Wrist", "Hand"]], T_stop=0,
      do_shuffle=True):
    self.do_shuffle = do_shuffle
    self.T_stop = T_stop
    self.subseq_length = subseq_length
    self.data_dir = "./data"
    self.batch_size = batch_size
    self.scale_factor = scale_factor # divide data by this factor
    self.filter_size = filter_size # the size of the filter before downsampling
    self.down_sample_ratio = 8
    self.joints = joints
    self.joints.sort(reverse=True) # so that we have a fixed order in joints
    self.time_interval = self.down_sample_ratio / 250.

    data_file = os.path.join(self.data_dir, "preprocessed_training_data.cpkl")
    raw_data_dir = os.path.join(self.data_dir, 
        "data_wolfgang/Data_Segmented_All/OntologySorted")

    if not (os.path.exists(data_file)) :
        print("creating training data pkl file from raw source")
        self.preprocess(raw_data_dir, data_file)

    self.load_preprocessed(data_file)
    self.reset_batch_pointer()

  def preprocess(self, data_dir, data_file):
    # create data file from raw .mat file sources
    # TODO: load a specific case
    # TODO: preprocessing everything at the beginning, instead of just some joints

    # build the list of xml files
    filelist = []
    # Set the directory you want to start from
    rootDir = data_dir
    for dirName, subdirList, fileList in os.walk(rootDir):
      #print('Found directory: %s' % dirName)
      for fname in fileList:
        #print('\t%s' % fname)
        filelist.append(dirName+"/"+fname)

    # function to read each individual mat file
    def get_trajectories(filename):

      mat_content = sio.loadmat(filename)
      mdata = mat_content['data']
      mdtype = mdata.dtype
      ndata = {n: mdata[n][0, 0] for n in mdtype.names}

      data_all = ndata['Data']
      p1segs = np.asarray(ndata['SegmentationP1N'] - 1) * 5
      p2segs = np.asarray(ndata['SegmentationP2N'] - 1) * 5
      
      labels = np.squeeze(ndata['Labels'])
      labels = np.asarray([str(l[0]) for l in labels])

      data = {}
      
      for idx, l in enumerate(labels):
        if any([joint in l for joint in nest.flatten(self.joints)]):
          # if the current data is what we want to load
          data[l] = data_all[idx].T # [time, DoF(x_y_z_var)]

      data['P1_Segmentation'] = p1segs
      data['P2_Segmentation'] = p1segs # all the segmentation should based on the first person

      # print(data.keys())
      return data

    def split_trajectory(traj, segs):
      subtrajs = []
      traj = interp_nan_2d(traj)
      for start, end in segs:
        if start < end:
          subtraj = traj[start:end, :]
          subtraj = subtraj[::self.down_sample_ratio, :]
          subtrajs.append(subtraj)
        else:
          print("Error in splitting zero length segment")
      return subtrajs

    def split_trajectories_by_person(data, person):
      traj_all_joints = []
      # loop thru all joints
      for joint in self.joints:
        trajectories = []
        # average over left and right attacher of this joint
        for key, val in data.items():
          if person in key and (joint[0] in key or joint[1] in key):
            # treat all joints in a list as the same
            trajectories.append(val[:, :3])
        trajectory = np.mean(trajectories, axis=0)
        traj_all_joints.append(trajectory)
      trajectory = np.concatenate(traj_all_joints, axis=1)

      # split the data, interp the nan, filter and downsample
      segs = data[person + '_Segmentation']
      subtrajs = split_trajectory(trajectory, segs)

      return subtrajs # [num_split, [time, DoF]]

    trajs_p1 = []
    trajs_p2 = []
    case_label = []
    for i in range(len(filelist)):
      if filelist[i].endswith(".mat"):
      # if filelist[i].endswith("/Case01/P0101S.mat"):
        print('processing '+filelist[i])
        case = re.findall(r"Case[0-9]+", filelist[i])[0]
        case = [int(case[-2:])]
        data = get_trajectories(filelist[i])
        # Person 1
        splits_p1 = split_trajectories_by_person(data, "P1")
        trajs_p1.extend(splits_p1)
        # Person 2
        splits_p2 = split_trajectories_by_person(data, "P2")
        trajs_p2.extend(splits_p2)
        case_label.extend(case * len(splits_p1))
        # debug
        # print(list((t1.shape, t2.shape) for t1, t2 in zip(splits_p1, splits_p2)))
        # raise Exception

    # plot_trajectory(trajs_p1, trajs_p2, case_label, 1)
    # plot_trajectory(trajs_p1, trajs_p2, case_label, 1)
    
    # trajs_p1, trajs_p2 = make_every_traj_start_the_same(trajs_p1, trajs_p2)
    # plot_trajectory(trajs_p1, trajs_p2, case_label, 1)

    # trajs_p1, trajs_p2, case_label = filter_cases(trajs_p1, trajs_p2, case_label, [1, 2, 3])

    # trajs_p1, trajs_p2, case_label = rotate_dataset(trajs_p1, traj_p2, case_label)

    assert len(case_label) == len(trajs_p1)

    preprocessed = {}
    preprocessed["trajectories_p1"] = trajs_p1
    preprocessed["trajectories_p2"] = trajs_p2
    preprocessed["case_label"] = case_label

    with open(data_file,"wb") as f:
        pickle.dump(preprocessed, f, protocol=2)

  def load_preprocessed(self, data_file):
    with open(data_file,"rb") as f:
        self.raw_data = pickle.load(f, encoding='latin1')#,  encoding='latin1', 'iso-8859-1'
        #self.raw_data = pickle.Unpickler(f).load()

    trajs_p1 = self.raw_data["trajectories_p1"]
    trajs_p2 = self.raw_data["trajectories_p2"]
    case_label = self.raw_data["case_label"]

    self.data = {
      "inputs": [], 
      "targets": [], 
      "conditions": []
      } 
    self.valid_data = [{"acc": [], "obstacle": [], "targets": [], "conditions": []} 
                        for _ in range(20)] # for each of the 20 cases 

    self.kmeans = [] # save the kmeans model

    counter = 0

    case_counter = [0] * 20
    valid_case_counter = [0] * 20

    cur_data_counter = 0

    end_points = [[]] * 20 # to calculate kmeans

    max_len = 0
    for traj_p1, traj_p2, case_num in zip(trajs_p1, trajs_p2, case_label):
      

      traj_p1 = np.asarray(traj_p1) # [time, DoF*joints]
      traj_p2 = np.asarray(traj_p2)

      # plt.plot(traj_p1)
      # plt.show()

      # an experiment to attach a stopping phase
      # traj_stop_p1 = np.repeat(traj_p1[-1:, :], self.T_stop, axis=0)
      # traj_stop_p2 = np.repeat(traj_p2[-1:, :], self.T_stop, axis=0)
      # traj_p1 = np.concatenate((traj_p1, traj_stop_p1), axis=0)
      # traj_p2 = np.concatenate((traj_p2, traj_stop_p2), axis=0)

      max_len = max(max_len, traj_p1.shape[0]) 

      traj_p1 /= self.scale_factor 
      traj_p2 /= self.scale_factor
      tar_p1 = traj_p1[-1, :] # [DoF*joints]
      tar_p2 = traj_p2[-1, :]

      obstacle_relative = traj_p2[:-1, :] - traj_p1[:-1, :]
      target_relative = tar_p1 - traj_p1[:-1, :]

      # [time, DoF*joints], [DoF*joints]
      _, x_start_p1 = differentiate(traj_p1, self.time_interval) # only for getting the start point
      vel_p1 = np.gradient(traj_p1, self.time_interval)[0]
      # [time, DoF*joints], [DoF*joints]
      _, v_start_p1 = differentiate(vel_p1, self.time_interval)
      acc_p1 = np.gradient(vel_p1, self.time_interval)[0]

      targets = acc_p1[1:, :]
      # conditions = v_start_p1
      # print(x_start_p1)

      # to verify the reverse of differentiation
      # vel_p1 = integral(acc_p1, v_start_p1, self.time_interval) 
      # traj_reconstuct_p1 = integral(vel_p1, x_start_p1, self.time_interval) 
      # fig = plt.figure(figsize=(12, 8))
      # ax = fig.gca(projection='3d')
      # ax.plot(traj_p1[:, 0], traj_p1[:, 1], traj_p1[:, 2], c='b')
      # ax.plot(traj_reconstuct_p1[:, 0], traj_reconstuct_p1[:, 1], traj_reconstuct_p1[:, 2], c='r')
      # ax.scatter(tar_p1[0], tar_p1[1], tar_p1[2], c='k')
      # plt.show()

      cur_data_counter = cur_data_counter + 1

      if cur_data_counter % 20 == 0: # collect validation set
        valid_case_counter[case_num-1] += 1
        conditions = np.concatenate((x_start_p1, v_start_p1, tar_p1, tar_p2), axis=0)
        self.valid_data[case_num-1]["acc"].append(acc_p1[:-1, :]) # [time, DoF*joints]
        self.valid_data[case_num-1]["obstacle"].append(traj_p2[:-1, :]) # [time, DoF*joints]
        self.valid_data[case_num-1]["targets"].append(targets) # [time, DoF*joints]
        self.valid_data[case_num-1]["conditions"].append(conditions) # [DoF*joints*4]
      else:
        end_points[case_num - 1].append(tar_p1[:2])

        # rotate train set
        acc_p1_set = rotate_dataset([acc_p1])
        obstacle_relative_set = rotate_dataset([obstacle_relative])
        target_relative_set = rotate_dataset([target_relative])
        targets_set = rotate_dataset([targets])

        x_start_p1_set = rotate_dataset([x_start_p1[None, :]])
        v_start_p1_set = rotate_dataset([v_start_p1[None, :]])
        tar_p1_set = rotate_dataset([tar_p1[None, :]])
        tar_p2_set = rotate_dataset([tar_p2[None, :]])

        for acc_p1, obs_rel, tar_rel, target, x_st, v_st, tar_p1, tar_p2 in \
          zip(acc_p1_set, obstacle_relative_set, target_relative_set, targets_set, \
              x_start_p1_set, v_start_p1_set, tar_p1_set, tar_p2_set):
          case_counter[case_num-1] += 1
          x_st = x_st[0]
          v_st = v_st[0]
          tar_p1 = tar_p1[0]
          tar_p2 = tar_p2[0]

          obs_repulsive = repulsive_force(obs_rel)

          inputs = np.concatenate(
            (tar_rel, obs_repulsive), axis=1)
          conditions = np.concatenate((x_st, v_st, tar_p1, tar_p2), axis=0)

          self.data["inputs"].append(inputs) # [time, 3*DoF*joints]
          self.data["targets"].append(target) # [time, DoF*joints]
          self.data["conditions"].append(conditions) # [4*DoF*joints]
          counter += int(len(traj_p1)/((self.subseq_length+1)))

    for end_points_per_case in end_points:
      if end_points_per_case:
        self.kmeans.append(KMeans(n_clusters=2).fit(end_points_per_case))

    # self.average = average = print_data_properties(self.data["inputs"])
    # self.data["inputs"] = [ t - average for t in self.data["inputs"] ]
    # self.data["targets"] = [ t - average[:3] for t in self.data["targets"] ]

    # print("train data: %d, valid data: %d"
    #       % (len(self.data["inputs"]), len(self.valid_data["inputs"])))
    print("Maximal length of the training data is %d" % max_len)
    print("Training data case distribution: " + str(case_counter))
    print("Validation data case distribution: " + str(valid_case_counter))
    self.num_batches = int(np.ceil(counter / self.batch_size))
    self.num_sequences = len(self.data["inputs"])

  def validation_data(self, case_labels):
    # returns validation data wrt. the case labels
    # for example, case_labels can be [1, 1, 10] then the return samples will be 
    # three, the first two from the first case and the third from the 10th case.
    # within each case, the sample is randomly selected.
    x_batch = []
    y_batch = []
    c_batch = []
    for case in case_labels:
      sample_idx = random.randint(0, len(self.valid_data[case - 1]["acc"]) - 1)
      inp_1 = self.valid_data[case - 1]["acc"][sample_idx]
      inp_2 = self.valid_data[case - 1]["obstacle"][sample_idx]
      inp = np.concatenate((inp_1, inp_2), axis=1)
      tar = self.valid_data[case - 1]["targets"][sample_idx]
      cond = self.valid_data[case - 1]["conditions"][sample_idx]
      x_batch.append(np.copy(inp))
      y_batch.append(np.copy(tar))
      c_batch.append(np.copy(cond))
    x_batch, x_weights, x_lens = numpy_fillzeros(x_batch, self.subseq_length)
    y_batch, _, _ = numpy_fillzeros(y_batch, self.subseq_length)
    return x_batch, y_batch, x_weights, np.asarray(c_batch)

  def validation_data_all_wo_kmeans(self, case_idx):
    """Returns all the validation samples from one case label"""
    x_batch = []
    y_batch = []
    c_batch = []
    for sample_idx in range(len(self.valid_data[case_idx-1]["acc"])):
      inp_1 = self.valid_data[case_idx - 1]["acc"][sample_idx]
      inp_2 = self.valid_data[case_idx - 1]["obstacle"][sample_idx]
      inp = np.concatenate((inp_1, inp_2), axis=1)
      tar = self.valid_data[case_idx - 1]["targets"][sample_idx]
      cond = self.valid_data[case_idx - 1]["conditions"][sample_idx]
      x_batch.append(np.copy(inp))
      y_batch.append(np.copy(tar))
      c_batch.append(np.copy(cond))
    x_batch, x_weights, x_lens = numpy_fillzeros(x_batch, self.subseq_length)
    y_batch, _, _ = numpy_fillzeros(y_batch, self.subseq_length)
    return x_batch, y_batch, x_weights, np.asarray(c_batch)

  def validation_data_all(self, case):
    """ Returns all the validation samples from one case label
    Followed by a k-means step to divide the forward / backward trajectories.
    """
    end_points = []
    x_set = []
    y_set = []
    w_set = [] # this part is replace as forward / backward labels
    c_set = []
    # plot the start point of validation data for visualization
    for i in range(len(self.valid_data[case - 1]["acc"])):
      inp_1 = self.valid_data[case - 1]["acc"][i]
      inp_2 = self.valid_data[case - 1]["obstacle"][i]
      inp = np.concatenate((inp_1, inp_2), axis=1)
      tar = self.valid_data[case - 1]["targets"][i]
      cond = self.valid_data[case - 1]["conditions"][i]
      x_set.append(np.copy(inp))
      y_set.append(np.copy(tar))
      c_set.append(np.copy(cond))
      # plt.scatter(cond[0], cond[1], c='b')
      # plt.scatter(cond[6], cond[7], c='r')
      end_points.append(cond[6:8])
    # plt.show()
    labels = divide_forward_backward(end_points, self.kmeans[case-1])

    return x_set, y_set, labels, c_set


  def next_batch(self):
    # returns a randomised, seq_length sized portion of the training data
    x_batch = []
    y_batch = []
    c_batch = []
    for i in range(self.batch_size):
      inp = self.data["inputs"][self.pointer]
      tar = self.data["targets"][self.pointer]
      cond = self.data["conditions"][self.pointer]
      x_batch.append(np.copy(inp))
      y_batch.append(np.copy(tar))
      c_batch.append(np.copy(cond))
      self.tick_batch_pointer()
    x_batch, x_weights, x_lens = numpy_fillzeros(x_batch, self.subseq_length)
    y_batch, _, _ = numpy_fillzeros(y_batch, self.subseq_length)
    return x_batch, y_batch, x_weights, np.asarray(c_batch), x_lens

  def tick_batch_pointer(self):
    self.pointer += 1
    if (self.pointer >= len(self.data["inputs"])):
      if self.do_shuffle: self.shuffle()
      self.pointer = 0

  def reset_batch_pointer(self):
    if self.do_shuffle: self.shuffle()
    self.pointer = 0

  def shuffle(self):
    print("Shuffling training data...")
    data_len = len(self.data["inputs"])
    idx = np.random.permutation(data_len)
    new_data = {name: [] for name in self.data.keys()}
    for name in self.data.keys():
      for i in idx:
        new_data[name].append(self.data[name][i])
    self.data = new_data

if __name__ == "__main__":
    data_loader = DataLoader(subseq_length=300)
    data_loader.validation_data_all(1)
    data_loader.validation_data_all(2)
    data_loader.validation_data_all(3)
    
    # x, y, w, t = data_loader.next_batch()
    # print(x.shape, y.shape, w.shape, t.shape)
    # x, y, w, t = data_loader.validation_data()
    # print(x.shape, y.shape, w.shape, t.shape)
    # for i in range(100):
    #   plt.plot(y[i])
    #   plt.show()
    
    # x = np.array([[0., 1.],
    #             [1., 2.],
    #             [3., 3.]])

    # # print(x, differentiate(x, 0.5))
    # res1, res2 = differentiate(x, 0.5)
    # print(res1, res2)
    # print(integral(res1, res2, 0.5))

