import numpy as np
from env import env
from EnvObs import EnvObs
from constants import constants
import argparse
from model import Agent
import matplotlib.pyplot as plt
import imageio
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-s', '--scenario', default='dense', type=str,
                    help="dense or sparse or verySparse")
parser.add_argument('-p', '--ckpt_path', default=None, type=str,
                    help="")
parser.add_argument('-g', '--greedy', action='store_true',
                    help="")
parser.add_argument('-f', '--save_gif', action='store_true',
                    help="")

def run():
    args = parser.parse_args()
    tf.reset_default_graph()
    with tf.Session() as sess:
        agent = Agent(sess)
        sess.run(tf.global_variables_initializer())
        agent.load_weight(args.ckpt_path)
        r = eval(agent, sess, args.scenario, args.greedy, args.save_gif)

def eval(agent, sess, scenario, greedy, save_gif):
    if greedy:
        print('greedy')
    game = env(scenario)
    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]

    obs = EnvObs()

    # initialize buffer
    R = []

    for i in range(20):
        game.new_episode()
        state = game.get_state()
        s = obs.reset(state)
        Reward = 0
        frame = []
        frame.append(state.screen_buffer)
        while True:
            a = agent.choose_action(s, greedy)

            r = game.make_action(actions[a])
            if r > 0:
                r = 1.0
            done = r > 0 or game.is_episode_finished()
            state_ = game.get_state()
            if state_ != None:
                frame.append(state_.screen_buffer)
            s_ = obs.observation(state_)
            Reward += r
            s = s_
            if r > 0:
                print('episode {} success !'.format(i))
                if save_gif:
                    frame = np.array(frame)
                    imageio.mimsave("../gif/episode{}.gif".format(i), (frame).astype(np.uint8), fps=40)
            if done:
                frame = []
                R.append(Reward)
                break

    return R

if __name__ == "__main__":
    run()
