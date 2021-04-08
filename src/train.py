import numpy as np
from env import env
from EnvObs import EnvObs
from constants import constants
import argparse
from model import A2C
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-s', '--scenario', default='dense', type=str,
                    help="dense or sparse or verySparse")
parser.add_argument('-p', '--ckpt_path', default=None, type=str,
                    help="")

def run():
    args = parser.parse_args()
    tf.reset_default_graph()
    with tf.Session() as sess:
        agent = A2C(sess)
        saver = tf.train.Saver(max_to_keep=100)
        sess.run(tf.global_variables_initializer())
        agent.load_pretrain()
        training(agent, sess, saver, args.scenario, args.ckpt_path)

def training(agent, sess, saver, scenario='dense', ckpt_path=None):
    game = env(scenario)
    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    if ckpt_path != None:
        saver.restore(sess, ckpt_path)

    obs = EnvObs()

    # set training steps
    step = 1
    if ckpt_path != None:
        step = 2e5 * (int(ckpt_path.split('-')[-1]) + 1)
    print('step: {}'.format(step))

    ckpt_save = 0
    if ckpt_path != None:
        ckpt_save = int(ckpt_path.split('-')[-1])+1
    print('ckpt_save: {}'.format(ckpt_save))

    num_episode = 1

    # initialize buffer
    buffer_s, buffer_ns, buffer_a, buffer_r, buffer_v = [], [], [], [], []
    R = []

    while(step < constants['TRAINING_STEPS']):
        # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
        game.new_episode()
        state = game.get_state()
        s = obs.reset(state)
        
        # episode start, initialized
        forward, left, right = (0, 0, 0)
        ep_r = 0

        while True:
            a, value = agent.choose_action(s)
            if a == 0:
                forward += 1
            elif a == 1:
                left += 1
            elif a == 2:
                right += 1

            r = 0
            for i in range(constants['FRAME_SKIP']):
                r += game.make_action(actions[a])
                if r > 0:
                    r = 1.0
                if r > 0 or game.is_episode_finished():
                    break
            done = r > 0 or game.is_episode_finished()
            state_ = game.get_state()
            s_ = obs.observation(state_)

            bonus = agent.get_bonus(s[np.newaxis, :], s_[np.newaxis, :], a[np.newaxis, np.newaxis])
            bonus = np.clip(bonus, 0, constants['REWARD_CLIP'])

            ep_r += (r + bonus)

            buffer_s.append(s)
            buffer_ns.append(s_)
            buffer_a.append(a)
            buffer_r.append(r + bonus)
            buffer_v.append(value[0, 0])

            if step % constants['UPDATE_ITER'] == 0 or done:
                if done and r > 0:
                    if state_ != None:
                        print('done!!!! step = {} r = {}'.format(state_.number, r))
                    else:
                        print('done!!!! step = {} r = {}'.format(2100, r))
                    print(buffer_v)
                    v_s_ = 0  # terminal
                else:
                    v_s_ = agent.get_v(s_[np.newaxis, :])[0, 0]
                # v_target
                buffer_v_target = []
                v_curr = v_s_
                for r in buffer_r[::-1]:  # reverse buffer r
                    v_curr = r + constants['GAMMA'] * v_curr
                    buffer_v_target.append(v_curr)
                buffer_v_target.reverse()
                # convert to numpy array
                buffer_s, buffer_ns, buffer_a, buffer_v_target = np.array(buffer_s), np.array(buffer_ns), np.array(buffer_a), np.array(buffer_v_target)
                feed_dict = {
                    agent.s: buffer_s,
                    agent.next_s: buffer_ns,
                    agent.a_his: buffer_a[:, np.newaxis],
                    agent.v_target: buffer_v_target[:, np.newaxis],
                }
                agent.update(feed_dict, done)
                buffer_s, buffer_ns, buffer_a, buffer_r, buffer_v = [], [], [], [], []
            s = s_
            step += 1
            if step % 2e5 == 0:
                saver.save(sess, '../ckpt/{}/ICM.ckpt'.format(scenario), global_step=ckpt_save)
                ckpt_save += 1
                eval_Reward = eval(agent, game)
                print('step: {}, Reward: {}'.format(step, eval_Reward))
                R.append(eval_Reward)
            if done:
                num_episode += 1
                print('episode reward: {:.8f}, forward: {:3d}, left: {:3d}, right: {:3d}'.format(ep_r, forward, left, right))
                break

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()

def eval(agent, game):
    actions = [[True, False, False], [False, True, False], [False, False, True]]
    R = []
    obs = EnvObs()
    for episode in range(10):
        game.new_episode()
        state = game.get_state()
        s = obs.reset(state)
        Reward = 0
        while True:
            a, _ = agent.choose_action(s)

            r = game.make_action(actions[a])
            if r > 0:
                r = 1.0
            done = r > 0 or game.is_episode_finished()
            state_ = game.get_state()
            s_ = obs.observation(state_)
            Reward += r
            s = s_
            if done:
                R.append(Reward)
                break

    return np.mean(R)

if __name__ == "__main__":
    run()