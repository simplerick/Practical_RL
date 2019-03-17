import random
import numpy as np
import gym
from gym.core import ObservationWrapper
from gym.spaces import Box
import cv2
import atari_wrappers
from framebuffer import FrameBuffer
import torch
import torch.nn as nn
import os
import time
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter



class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)
        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size)
    def _to_gray_scale(self, rgb, channel_weights=[0.8, 0.1, 0.1]):
        return (rgb*channel_weights).sum(axis=-1)
    def _observation(self, img):
        """what happens to each observation"""
        # Here's what you need to do:
        #  * crop image, remove irrelevant parts
        #  * resize image to self.img_size
        #     (use imresize from any library you want,
        #      e.g. opencv, skimage, PIL, keras)
        #  * cast image to grayscale
        #  * convert image pixels to (0,1) range, float32 type
        image = img[30:195,6:156]
        image = self._to_gray_scale(image)
        image = cv2.resize(image,self.img_size[1:],interpolation=cv2.INTER_NEAREST)
        image = image.reshape(self.img_size)
        image = np.float32(image)/255
        return image

def PrimaryAtariWrap(env, clip_rewards=True):
    assert 'NoFrameskip' in env.spec.id
    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
    env = atari_wrappers.EpisodicLifeEnv(env)
    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    env = atari_wrappers.FireResetEnv(env)
    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)
    # This wrapper is yours :)
    env = PreprocessAtariObs(env)
    return env

def make_env(clip_rewards=True, seed=None):
    env = gym.make("BreakoutNoFrameskip-v4")  # create raw env
    if seed is not None:
        env.seed(seed)
    env = PrimaryAtariWrap(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        # Define your network body here. Please make sure agent is fully contained here
        self.net = nn.Sequential(
                nn.Conv2d(state_shape[0], 16, (3,3), stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, (3,3), stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, (3,3), stride=2),
                nn.ReLU(),
                Flatten(),
                nn.Linear(3136,256),
                nn.ReLU(),
                nn.Linear(256,n_actions)
        )

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        qvalues = self.net(state_t)
        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break
        rewards.append(reward)
    return np.mean(rewards)

class Net(nn.Module):
    def __init__(self, state_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(state_shape[0], 16, (3,3), stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, (3,3), stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, (3,3), stride=2),
                nn.ReLU(),
                Flatten()
        )
        self.a = nn.Sequential(
            nn.Linear(3136,256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.v = nn.Sequential(
            nn.Linear(3136,256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self,x):
        h = self.conv(x)
        A = self.a(h)
        V = self.v(h)
        Q = V + A - torch.mean(A, dim=-1, keepdim=True)
        return Q

class DDQNAgent(DQNAgent):
    def __init__(self, state_shape, n_actions, epsilon):
        super().__init__(state_shape, n_actions, epsilon)
        self.net = Net(state_shape,n_actions)




def share_grad(net, shared_net):
    for param, shared_param in zip(net.parameters(),shared_net.parameters()):
        if param.grad == None:
            param.grad = torch.zeros_like(param) # initialization
        shared_param._grad = param.grad # reference


def eval(target_agent,T, I_eval, T_max):
    writer = SummaryWriter()
    T_prev = 0
    time_prev = 0
    while T.value<T_max:
        if (T.value - T_prev) < I_eval:
            time.sleep(1)
            continue
        time_per_iter = (time.time()-time_prev)/(T.value-T_prev)
        T_prev = T.value
        time_prev = time.time()
        clipped_env = make_env(seed = T_prev)
        env = make_env(clip_rewards=False,seed = T_prev)
        clipped_reward = evaluate(clipped_env,target_agent,n_games=5,greedy=True)
        reward = evaluate(env,target_agent,n_games=5,greedy=True)
        v0 = np.max(target_agent.get_qvalues([clipped_env.reset()]))
        writer.add_scalar('data/clipped_reward', clipped_reward, T_prev)
        writer.add_scalar('data/reward', reward, T_prev)
        writer.add_scalar('data/v0', v0, T_prev)
        writer.add_scalar('data/time_per_iter', time_per_iter, T_prev)
        env.close()
        clipped_env.close()
    writer.close()



def process_train(id, agent, target_agent, T, n_update, I_target,num_steps, T_max, lr, epsilon_decay,gamma):
    torch.set_default_tensor_type(next(agent.parameters()).type())
    device = next(agent.parameters()).device
    env = make_env(seed=id)
    s = env.reset()
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    np.random.seed(id)
    process_agent = DDQNAgent(state_shape, n_actions, epsilon=np.random.uniform(0.7,1)).to(device)
    epsilon_min = 10**np.random.uniform(-0.5,-2.2)
    share_grad(process_agent, agent)
    opt = torch.optim.Adam(agent.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[200000,500000,1000000], gamma=0.3) # lr scheduler

    while T.value<T_max:
        process_agent.load_state_dict(agent.state_dict())
        rewards = []
        states = []
        actions = []
        for _ in range(num_steps):
            states.append(s)
            qvalues = process_agent.get_qvalues([s])
            a = process_agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(a)
            actions.append(a)
            rewards.append(r)
            process_agent.epsilon = max(epsilon_min, process_agent.epsilon-epsilon_decay)
            if done:
                s = env.reset()
                break

        with T.get_lock():
            T.value += len(states)

        R = []
        if done:
            R.append(0)
        else:
            a_max = np.argmax(process_agent.get_qvalues([s]))
            R.append(target_agent.get_qvalues([s])[0,a_max])

        states = torch.tensor(states)
        Q = process_agent(states)[range(len(actions)), actions]
        for _ in range(len(rewards)):
            r = rewards.pop()
            R.append(r + gamma*R[-1])
        R = torch.tensor(R[-1:0:-1])
        loss = torch.mean((R-Q)**2)
        loss.backward()
        nn.utils.clip_grad_norm_(process_agent.parameters(), 20)
        opt.step()
        opt.zero_grad()
        scheduler.step()
        if T.value//I_target > n_update.value:
            with n_update.get_lock():
                n_update.value += 1
            target_agent.load_state_dict(agent.state_dict())



if __name__ == '__main__':

    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    env = make_env()
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    env.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gamma = 0.99
    num_processes = 8
    num_steps = 5
    T_max = int(1e7)
    I_eval = 20000
    I_target = 15000
    lr = 1e-4
    epsilon_decay = 5e-5

    agent = DDQNAgent(state_shape, n_actions, epsilon=0).to(device)
    target_agent = DDQNAgent(state_shape, n_actions, epsilon=0).to(device)
    target_agent.load_state_dict(agent.state_dict())
    agent.share_memory()
    target_agent.share_memory()

    processes = []

    T = mp.Value('I', 0)
    n_update = mp.Value('I', 0)

    p = mp.Process(target=eval, args=(target_agent,T, I_eval, T_max))
    p.start()
    processes.append(p)

    for id in range(1, num_processes):
        p = mp.Process(target=process_train, args=(id, agent, target_agent, T, n_update, I_target,num_steps, T_max, lr, epsilon_decay,gamma))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    torch.save(agent.state_dict(), str(T_max)+".state_dict")
