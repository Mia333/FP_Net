
import os
import time
import math
import random

import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

# the nest package has changed its location
if tf.__version__.startswith("1.2"):
  from tensorflow.python.util import nest
elif tf.__version__.startswith("1.4"):
  from tensorflow.contrib.framework import nest
else:
  raise ValueError("Cannot locate tensorflow 'nest' package!")

# useful ops for wrting cell wrapper
from tensorflow.python.ops import rnn_cell_impl

from data_loader import integral, repulsive_force


# weights / biases initializer
def weights(name, shape):
  return tf.get_variable(
    name, 
    shape=shape, 
    initializer=tf.contrib.layers.xavier_initializer()
    )


def biases(name, shape):
  return tf.get_variable(
    name, 
    initializer=tf.zeros(shape)
    )


def tf_repeat(tensor, repeats):
    """
    Adopted from https://github.com/tensorflow/tensorflow/issues/8246
      by qianyizhang at github

    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be
      the same as the number of dimensions in input

    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor


class Model():
  def __init__(self, args, sampling=False, bias=0.):

    self.board_writer = None

    # MSE loss output mapping and loss calculation
    def output_map(cell_outputs, num_mixture=20):
      batch_size, _, dim_rec = cell_outputs.shape
      cell_outputs_flat = tf.reshape(cell_outputs, [-1, dim_rec.value]) # [batch*time, dim_rec]

      dim_out = 3

      # output mapping
      W = weights("out_W", [dim_rec, dim_out])
      b = biases("out_b", [dim_out])
      outputs = tf.matmul(cell_outputs_flat, W) + b

      return outputs

    def loss_mse(evidence, predictions, weights):
      # hard code the output dim as 3
      evidence_flat = tf.reshape(evidence, [-1, 3]) # shape of evidence [batch, time ,3]

      def full_loss(evidence, predictions):
        loss = tf.square(evidence - predictions)
        loss = tf.reduce_sum(loss, [1], keep_dims=True) #? [batch * time ,1]
        return loss

      loss = full_loss(evidence_flat, predictions)

      loss = tf.reshape(loss, [self.args.batch_size, -1, 1]) #? [batch_size, time , 1]

      def convergence_loss_3d(acc_flat):
        accs = tf.reshape(acc_flat, [self.args.batch_size, -1, 3])
        return tf.reduce_sum(tf.square(tf.reduce_sum(accs, 1)), 1)
      
      convergence_loss = tf.reduce_mean(convergence_loss_3d(predictions)) # [self.args.batch_size] --> 1, mean over time and three hard output

      loss = loss * weights # shape????????

      loss_per_step = tf.reduce_sum(loss) / tf.reduce_sum(weights) # ?????? []

      try:
        const_factor = args.constraint_factor
      except:
        const_factor = 0.

      loss_final = loss_per_step + const_factor * convergence_loss

      return loss_final

    # GMM loss output mapping and loss calculation
    def coeff_map(cell_outputs, num_mixture=20, bias=0.):

      batch_size, _, dim_rec = cell_outputs.shape # dim_rec: size of RNN hidden state
      cell_outputs_flat = tf.reshape(cell_outputs, [-1, dim_rec.value]) # [batch*time, dim_rec = 128 default]

      # 6 = 3 of means, 3 of variance, 1 of cluster prior, 3 of correlation
      dim_coeff = 10 * num_mixture

      # output mapping
      W = weights("out_W", [dim_rec, dim_coeff])
      b = biases("out_b", [dim_coeff])
      outputs = tf.matmul(cell_outputs_flat, W) + b # [batch*time, dim_coeff=10 * num_mixture]

      # split the coeff. along different clusters
      # pi: cluster prior
      # m1, m2, m3: mean
      # s1, s2, s3: std
      # c12, c13, c23: correlation
      pi, m1, m2, m3, s1, s2, s3, c12, c13, c23 = tf.split(
        axis=1, num_or_size_splits=10, value=outputs) # [batch*time, num_mixture]

      # output functions
      pi = tf.nn.softmax(pi * (1+bias))

      s1 = tf.exp(s1 - bias)
      s2 = tf.exp(s2 - bias)
      s3 = tf.exp(s3 - bias)

      c12 = tf.tanh(c12)
      c13 = tf.tanh(c13)
      c23 = tf.tanh(c23)

      # pack them for convenience
      mu = (m1, m2, m3)
      sigma = (s1, s2, s3)
      corr = (c12, c13, c23)

      return pi, mu, sigma, corr

    def loss_gmm(evidence, pi, mu, sigma, corr, weights):
      # hard code the output dim as 3
      evidence_flat = tf.reshape(evidence, [-1, 3]) # [batch*time, 3], targets

      def single_mixture_loss(x1, x2, x3, m1, m2, m3, s1, s2, s3, c12, c13, c23):
        diff1 = x1 - m1
        diff2 = x2 - m2
        diff3 = x3 - m3

        c12_sq = tf.square(c12) # correlation between 1 and 2 squared
        c13_sq = tf.square(c13)
        c23_sq = tf.square(c23)

        corr_mix = 1. + 2. * c12 * c13 * c23 - c12_sq - c13_sq - c23_sq
        
        exponent = (- (1. - c23_sq) / tf.square(s1) * tf.square(diff1) / 2 \
                    - (1. - c13_sq) / tf.square(s2) * tf.square(diff2) / 2 \
                    - (1. - c12_sq) / tf.square(s3) * tf.square(diff3) / 2 \
                    + (c12 - c13 * c23) / (s1 * s2) * diff1 * diff2 \
                    + (c13 - c12 * c23) / (s1 * s3) * diff1 * diff3 \
                    + (c23 - c12 * c13) / (s2 * s3) * diff2 * diff3) / corr_mix
        normalizer = 1. / ((2*math.pi) ** 1.5) / (s1 * s2 * s3) / tf.sqrt(corr_mix)

        result = normalizer * tf.exp(exponent)
        return result

      def full_loss(x1, x2, x3, pi, m1, m2, m3, s1, s2, s3, c12, c13, c23):
        single_loss = single_mixture_loss(x1, x2, x3, m1, m2, m3, 
                                          s1, s2, s3, c12, c13, c23)
        gmm_loss = tf.reduce_sum(pi * single_loss, 1, keep_dims=True) # [batch*time, 1]
        gmm_loss = tf.maximum(gmm_loss, 1e-4) # avoid nan : return the max of gmm_loss or 1e-4, element-wise
        log_loss = - tf.log(gmm_loss)
        return log_loss # [batch*time, 1]

    ##???????????????
      x1 = evidence_flat[:, 0:1]
      x2 = evidence_flat[:, 1:2]
      x3 = evidence_flat[:, 2:3]

      m1, m2, m3 = mu
      s1, s2, s3 = sigma
      c12, c13, c23 = corr

      def convergence_loss_1d(acc_flat):
        # a tryout to constrain the final position
        accs = tf.reshape(acc_flat, [self.args.batch_size, -1, 1]) #[batch, time, 1]
        return tf.square(tf.reduce_sum(accs, [1,2]))

      convergence_loss = tf.reduce_mean(convergence_loss_1d(m1)
                         + convergence_loss_1d(m2)
                         + convergence_loss_1d(m3)) #[batch] --> 1 : m1^2 + m2^2 + m3^2

      loss = full_loss(x1, x2, x3, pi, m1, m2, m3, s1, s2, s3, c12, c13, c23)

      loss = tf.reshape(loss, [self.args.batch_size, -1, 1])  # [batch, time, 1]

      loss = loss * weights

      loss_per_step = tf.reduce_sum(loss) / tf.reduce_sum(weights) #???????????was machen

      loss_final = loss_per_step + args.constraint_factor * convergence_loss # 'the weight for constraint term in the cost function.' 

      return loss_final

    self.args = args

    # hard code the batch_size as 1 for sampling and bias as 0 for training
    if sampling:
      args.batch_size = 1
    else:
      bias = 0.
    # placeholders
    # the dimension looks like this: [batch, time, data_dimension]
    self.inputs = inputs = tf.placeholder(
      tf.float32, 
      [args.batch_size, None, 6]
      ) #[batch, time, 6]
    self.targets = targets = tf.placeholder(
      tf.float32,
      [args.batch_size, None, 3]
      )
    self.weights = tf.placeholder(
      tf.float32, 
      [args.batch_size, None, 1]
      )
    # self.conditions = tf.placeholder(
    #   tf.float32, 
    #   [args.batch_size, 3]
    #   )
    # conditions = tf.expand_dims(self.conditions, 1)
    # print(conditions)
    # if not sampling:
    #   conditions = tf_repeat(conditions, [1, args.bptt_length, 1])
    #   conditions.set_shape([args.batch_size, None, 3])
    #   print(conditions)
    # cell_inputs = tf.concat((self.inputs, conditions), axis=2)
    # print(cell_inputs) 
    cell_inputs = self.inputs 

    # cells
    def cell_init(dim_rec):
      return tf.contrib.rnn.BasicLSTMCell(dim_rec)

    cell = tf.contrib.rnn.MultiRNNCell(
      [cell_init(args.dim_rec) for _ in range(args.num_layers)]
      )

    self.state_in = cell.zero_state(args.batch_size, dtype=tf.float32) #cell_state?????[batch,state_size:size(s) of state(s) used by this cell.] #[10, 128]?????????????????

    # build structures
    cell_outputs, self.state_out = tf.nn.dynamic_rnn(
      cell=cell,
      inputs=cell_inputs,
      initial_state=self.state_in
      )
    if args.loss_form == 'mse':
      self.outputs = output_map(cell_outputs)
      self.loss = loss = loss_mse(targets, self.outputs, self.weights)
    elif args.loss_form == 'gmm':
      self.pi, self.mu, self.sigma, self.corr = coeff_map(
        cell_outputs, num_mixture=args.num_mixture, bias=bias)
      self.loss = loss = loss_gmm(targets, self.pi, self.mu, 
                                  self.sigma, self.corr, self.weights)

    # optimizer
    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables() #?????????????????????A list of Variable objects, parameter theta????
    print("Number of trainable variables %d" % len(tvars))
    print(tvars)
    grads, _ = tf.clip_by_global_norm(
      tf.gradients(loss, tvars),
      args.max_grad_norm
      )
    # optimizer = tf.train.GradientDescentOptimizer(self.lr)
    optimizer = tf.train.AdamOptimizer(self.lr)
    # optimizer = tf.train.RMSPropOptimizer(self.lr)
    # optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='BFGS', options={'maxiter': args.num_epochs, 'disp': True, 'gtol': 1e-16, 'norm': 2})
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        

  def sample(self, sess, obstacle_traj, conditions, gmm=False, noise_level=0.001):
    """ Sample a sequence
    The sampled trajectory is as long as the given obstacle_trajectory.
    The conditions argument contains [x_start, v_start, x_target, obstacle_target]
    """
    def step_noise():
      # simulate a observation noise
      return np.random.normal(0, noise_level, 3)

    def sample_3d_gaussian(mu, Sigma):
      x1, x2, x3 = np.random.multivariate_normal(mu, Sigma+1e-4*np.eye(3), 1)[0]
      return x1, x2, x3

    def sample_cluster(pi):
      rand = random.random()
      accumulate = 0.
      for i in range(len(pi)):
        accumulate += pi[i]
        if accumulate >= rand:
          return i
      raise ValueError("Cannot sample a cluster!")

    def sample_mix_gaussian(pi, mu, sigma, corr):
      mu1, mu2, mu3 = mu
      s1, s2, s3 = sigma
      c12, c13, c23 = corr
      mu1, mu2, mu3 = mu1[-1], mu2[-1], mu3[-1]
      s1, s2, s3 = s1[-1], s2[-1], s3[-1]
      c12, c13, c23 = c12[-1], c13[-1], c23[-1]
      # print(s1, s2, s3) 
      # buid the mu and covariance
      mu_vec = np.stack((mu1, mu2, mu3), axis=1) # [cluster, 3]
      col1 = np.stack((s1*s1, s1*s2*c12, s1*s3*c13), axis=1) # [cluster, 3]
      col2 = np.stack((s1*s2*c12, s2*s2, s2*s3*c23), axis=1) # [cluster, 3]
      col3 = np.stack((s1*s3*c13, s2*s3*c23, s3*s3), axis=1) # [cluster, 3]
      cov_mat = np.stack((col1, col2, col3), axis=2) # [cluster, 3, 3]
      # print(cov_mat)
      cluster_idx = sample_cluster(pi)
      x1, x2, x3 = sample_3d_gaussian(mu_vec[cluster_idx], cov_mat[cluster_idx])
      return np.array([x1, x2, x3])

    def step_integral(acc, time_interval, v_start=np.array([0., 0., 0.])):
      v_new = v_start + acc * time_interval
      # x_diff = v_start * time_interval + 0.5 * acc * (time_interval ** 2)
      x_diff = v_new * time_interval
      return x_diff, v_new

    if len(obstacle_traj.shape) == 2:
      # make input as 3 dimension for consistancy
      obstacle_traj = obstacle_traj[None, :, :]
    
    x_start = conditions[:3]
    target = conditions[6:9]

    # by default, the starting accelerations are all 0
    initial_inputs = np.zeros((1, 1, 6), dtype=np.float32)
    initial_inputs[0, 0, :3] = target - x_start # + step_noise()
    initial_inputs[0, 0, 3:] = repulsive_force(obstacle_traj[0, 0, :] - x_start)#+ step_noise()
    
    # init_feed = {self.inputs: initial_inputs, self.conditions: conditions}
    init_feed = {self.inputs: initial_inputs}

    # initial run
    if not gmm:
      outputs = sess.run(
        [self.outputs] + nest.flatten(self.state_out),
        feed_dict=init_feed
        )

      acc = outputs[0][-1, :]
      prev_state = outputs[1:]
    else:
      outputs = sess.run(
        [self.pi, self.mu, self.sigma, self.corr] + nest.flatten(self.state_out),
        feed_dict=init_feed
        )
      pi, mu, sigma, corr = outputs[:4]
      prev_state = outputs[4:]
      acc = sample_mix_gaussian(pi[-1], mu, sigma, corr)

    accs = []
    traj = []

    traj.append(x_start)

    x_diff, v_new = step_integral(acc, time_interval=8./250) # change the time_interval if necessary
    x_new = x_start + x_diff
    traj.append(x_new)

    prev_inp = np.zeros((1, 1, 6), dtype=np.float32)
    prev_inp[0, 0, :3] = target - x_new
    prev_inp[0, 0, 3:] = repulsive_force(obstacle_traj[0, 1, :] - x_new) #+ step_noise()
    v_last = v_new
    x_last = x_new
    
    for i in range(obstacle_traj.shape[1] - 1):
      feed = {tensor: s 
              for tensor, s in zip(nest.flatten(self.state_in), prev_state)}
      feed[self.inputs] = prev_inp
      # feed[self.conditions] = conditions
      if not gmm:
        outputs = sess.run(
          [self.outputs]+nest.flatten(self.state_out),
          feed_dict=feed
          )
        acc = outputs[0][0, :]    
        state = outputs[1:]
      else:
        outputs = sess.run(
          [self.pi, self.mu, self.sigma, self.corr] + nest.flatten(self.state_out),
          feed_dict=feed
          )
        pi, mu, sigma, corr = outputs[:4]
        state = outputs[4:]
        acc = sample_mix_gaussian(pi[-1], mu, sigma, corr)

      accs.append(np.squeeze(acc))

      if obstacle_traj.shape[1] - 2 == i:
        break

      x_diff, v_new = step_integral(acc, time_interval=8./250, v_start=v_last)
      x_new = x_last + x_diff
      traj.append(x_new)

      prev_inp = np.zeros((1, 1, 6), dtype=np.float32)
      prev_inp[0, 0, :3] = target - x_new
      prev_inp[0, 0, 3:] = repulsive_force(obstacle_traj[0, i+2, :] - x_new) #+ step_noise()

      v_last = v_new
      x_last = x_new
 
      prev_state = state

    accs = np.asarray(accs)
    traj = np.asarray(traj)
    return accs, traj


  def train(self, sess, sequence, targets, weights, 
            conditions, subseq_length, step_count):
    """ Cut the training sequences into multiple sub-sequences for training
    """

    if self.board_writer is None:
      # tensorboard
      tf.summary.scalar('loss', self.loss)
      self.merged = tf.summary.merge_all()
      self.board_writer = tf.summary.FileWriter(
        self.args.summary_dir + "/" + str(int(time.time())), 
        sess.graph
        )

    loss_list = []
    init_feed = {}
    init_feed[self.inputs] = sequence[:, :subseq_length, :]
    init_feed[self.targets] = targets[:, :subseq_length, :]
    init_feed[self.weights] = weights[:, :subseq_length, :]
    # init_feed[self.conditions] = conditions 
    outputs = sess.run(
      [self.loss, self.merged, self.train_op] + nest.flatten(self.state_out),
      init_feed
      )  # 
    train_loss, summary, _ = outputs[:3]
    prev_state = outputs[3:]
    self.board_writer.add_summary(summary, step_count)
    step_count += 1

    loss_list.append(train_loss)
    
    while sequence.shape[1] > subseq_length:
      sequence = sequence[:, subseq_length:, :]
      targets = targets[:, subseq_length:, :]
      weights = weights[:, subseq_length:, :]

      feed = {tensor: s 
              for tensor, s in zip(nest.flatten(self.state_in), prev_state)}
      feed[self.inputs] = sequence[:, :subseq_length, :]
      feed[self.targets] = targets[:, :subseq_length, :]
      feed[self.weights] = weights[:, :subseq_length, :]
      # feed[self.conditions] = conditions

      outputs = sess.run(
        [self.loss, self.merged, self.train_op]+nest.flatten(self.state_out),
        feed_dict=feed
        )
      train_loss, summary, _ = outputs[:3]
      prev_state = outputs[3:]

      self.board_writer.add_summary(summary, step_count)
      step_count += 1

      loss_list.append(train_loss)

    return loss_list, step_count
