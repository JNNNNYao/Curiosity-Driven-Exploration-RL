import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from constants import constants

class A2C(object):
    def __init__(self, sess):
        self.s = tf.placeholder(tf.float32, [None, constants['FRAME_SHAPE'][0], constants['FRAME_SHAPE'][1], 4], 'S')
        self.next_s = tf.placeholder(tf.float32, [None, constants['FRAME_SHAPE'][0], constants['FRAME_SHAPE'][1], 4], 'Next_S')
        self.a_his = tf.placeholder(tf.int32, [None, 1], 'A')
        self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

        self.sess = sess

        self._build_net()
        self._build_ICM()

        self.inv_loss_buffer = []
        self.forward_loss_buffer = []
        self.a_loss_buffer = []
        self.c_loss_buffer = []

        self.inv_loss_record = []
        self.forward_loss_record = []
        self.a_loss_record = []
        self.c_loss_record = []

        with tf.name_scope('loss'):
            # critic loss
            td = tf.subtract(self.v_target, self.v, name='TD_error')
            self.c_loss = 0.5 * tf.reduce_mean(tf.square(td))
            # actor loss
            log_prob = tf.reduce_sum(
                tf.log(self.a_prob) * tf.one_hot(self.a_his[:, 0], constants['N_A'], dtype=tf.float32), axis=1, keep_dims=True)
            exp_v = log_prob * tf.stop_gradient(td)
            entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob), axis=1, keep_dims=True)  # encourage exploration
            self.exp_v = constants['ENTROPY_BETA'] * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)
            # policy loss
            self.loss = 0.5 * self.c_loss + self.a_loss
            # ICM loss
            self.ICM_loss = 10.0 * ((self.invloss * (1-constants['FORWARD_LOSS_WT']) + self.forwardloss * constants['FORWARD_LOSS_WT']))

        with tf.name_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(constants['LR'], decay=0.99, epsilon=1e-5).minimize(20 * self.loss)
            self.train_ICM_op = tf.train.AdamOptimizer(constants['ICM_LR']).minimize(self.ICM_loss)

    def _build_net(self):
        with tf.variable_scope('Policy'):
            x = tf.layers.conv2d(self.s, 32, [3, 3], strides=(2, 2), padding='same', name='l1')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, 32, [3, 3], strides=(2, 2), padding='same', name='l2')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, 32, [3, 3], strides=(2, 2), padding='same', name='l3')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, 32, [3, 3], strides=(2, 2), padding='same', name='l4')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.flatten(x)
            self.x = tf.layers.dense(x, 256, tf.nn.relu, name='l5')
            self.logits = tf.layers.dense(self.x, constants['N_A'], name='ap')
            self.a_prob = tf.nn.softmax(self.logits)
            self.v = tf.layers.dense(self.x, 1, name='v')  # state value

    def _build_ICM(self):
        with tf.variable_scope('ICM'):
            self.phi1 = self._Encoder(self.s)
            self.phi2 = self._Encoder(self.next_s, reuse=True)
            # inverse model
            g = tf.concat([self.phi1, self.phi2], 1)
            g = tf.layers.dense(g, 256, tf.nn.relu, name='g1')
            ICM_logits = tf.layers.dense(g, constants['N_A'], name='g2')
            self.invloss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=ICM_logits, labels=self.a_his[:, 0], name="invloss"))
            # forward model
            f = tf.concat([self.phi1, tf.one_hot(self.a_his[:, 0], constants['N_A'], dtype=tf.float32)], 1)
            f = tf.layers.dense(f, 256, tf.nn.relu, name='f1')
            f = tf.layers.dense(f, self.phi1.get_shape()[1].value, name='f2')
            forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, self.phi2)), name='forwardloss')
            self.forwardloss = forwardloss * 288.0
            self.bonus = constants['PREDICTION_BETA'] * self.forwardloss

    def _Encoder(self, x, reuse=False):
        with tf.variable_scope('Encoder', reuse=reuse):
            x = tf.layers.conv2d(x, 32, [3, 3], strides=(2, 2), padding='same', name='l1')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, 32, [3, 3], strides=(2, 2), padding='same', name='l2')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, 32, [3, 3], strides=(2, 2), padding='same', name='l3')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, 32, [3, 3], strides=(2, 2), padding='same', name='l4')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.flatten(x)
            return x
    
    def load_pretrain(self, path='../ckpt/pretrain/pretrain.ckpt-5'):
        print('load pretrain weight from {}'.format(path))
        backbone_var = {}
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ICM/Encoder'):
            backbone_var[var.name.split('/', 1)[1].split(':')[0]] = var
            
        saver = tf.train.Saver(backbone_var)
        saver.restore(self.sess, path)

    def choose_action(self, s):
        feed_dict = {
            self.s: s[np.newaxis, :],
        }
        prob_weights, value = self.sess.run([self.a_prob, self.v], feed_dict=feed_dict)
        # stochastic policy
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action, value

    def get_v(self, s):
        feed_dict = {
            self.s: s,
        }
        return self.sess.run(self.v, feed_dict=feed_dict)
    
    def get_bonus(self, s, next_s, a_his):
        feed_dict = {
            self.s: s,
            self.next_s: next_s,
            self.a_his: a_his
        }
        return self.sess.run(self.bonus, feed_dict=feed_dict)

    def update(self, feed_dict, done):
        f, i, a, c, _, _ = self.sess.run([self.forwardloss, self.invloss, self.a_loss, self.c_loss, self.train_op, self.train_ICM_op], feed_dict)
        self.inv_loss_buffer.append(i)
        self.forward_loss_buffer.append(f)
        self.a_loss_buffer.append(a)
        self.c_loss_buffer.append(c)
        if done:
            print('inv_loss: {}'.format(np.mean(self.inv_loss_buffer)))
            print('forward_loss: {}'.format(np.mean(self.forward_loss_buffer)))
            print('a_loss: {}'.format(np.mean(self.a_loss_buffer)))
            print('c_loss: {}'.format(np.mean(self.c_loss_buffer)))
            self.inv_loss_record.append(np.mean(self.inv_loss_buffer))
            self.forward_loss_record.append(np.mean(self.forward_loss_buffer))
            self.a_loss_record.append(np.mean(self.a_loss_buffer))
            self.c_loss_record.append(np.mean(self.c_loss_buffer))
            self.inv_loss_buffer = []
            self.forward_loss_buffer = []
            self.a_loss_buffer = []
            self.c_loss_buffer = []



class Agent(object):
    def __init__(self, sess):
        self.s = tf.placeholder(tf.float32, [None, constants['FRAME_SHAPE'][0], constants['FRAME_SHAPE'][1], 4], 'S')

        self.sess = sess

        self._build_net()

    def _build_net(self):
        with tf.variable_scope('Policy'):
            x = tf.layers.conv2d(self.s, 32, [3, 3], strides=(2, 2), padding='same', name='l1')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, 32, [3, 3], strides=(2, 2), padding='same', name='l2')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, 32, [3, 3], strides=(2, 2), padding='same', name='l3')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(x, 32, [3, 3], strides=(2, 2), padding='same', name='l4')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.flatten(x)
            self.x = tf.layers.dense(x, 256, tf.nn.relu, name='l5')
            self.logits = tf.layers.dense(self.x, constants['N_A'], name='ap')
            self.a_prob = tf.nn.softmax(self.logits)
            self.v = tf.layers.dense(self.x, 1, name='v')  # state value
        self.policy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Policy')

    def load_weight(self, path):
        print('load weight from {}'.format(path))
        backbone_var = {}
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Policy'):
            backbone_var[var.name.split(':')[0]] = var
        saver = tf.train.Saver(backbone_var)
        
        saver.restore(self.sess, path)

    def choose_action(self, s, greedy):
        feed_dict = {
            self.s: s[np.newaxis, :],
        }
        prob_weights = self.sess.run(self.a_prob, feed_dict=feed_dict)
        if greedy:
            action = np.argmax(prob_weights)
        else:
            action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action