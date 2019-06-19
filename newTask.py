import numpy as np
from physics_sim import PhysicsSim
from scipy.stats import norm

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        #self.state_size = self.action_repeat * 9
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 1.]) 
        self.init_pose = init_pose if init_pose is not None else np.array([0., 0., 20.]) 
        self.last_pose = self.init_pose

    def get_reward(self):
        """Uses current pose of sim to return reward."""
                
        xy_penalty = 0.01 * abs(self.sim.v[0]) if abs(self.sim.v[0]) > 0. else 0
        xy_penalty += 0.01 * abs(self.sim.v[1]) if abs(self.sim.v[1]) > 0. else 0                
        downward_penalty = 0.3 * abs(self.sim.pose[2] - self.init_pose[2]) if self.init_pose[2] > self.sim.pose[2] else 0
        
        z_displacement = self.sim.pose[2] - self.last_pose[2] #To appoarch value 0.3 per 0.02s, 5m per second
        z_displacement *= 10 #scale up
        upward_reward = 0.5 * norm(loc = 3, scale = 1.).pdf(z_displacement) if z_displacement > 0. else 0
        #max 0.195
        
        upward_reward += min(1., 0.3 * abs(self.sim.pose[2] - self.init_pose[2])) if self.init_pose[2] < self.sim.pose[2] else 0
        #upward_reward += 0.3 * abs(self.sim.pose[2] - self.init_pose[2]) if self.init_pose[2] < self.sim.pose[2] else 0
        #max 1
        
        reward = -1*(xy_penalty + downward_penalty) + upward_reward
        reward = np.tanh(reward)
        
#         if self.sim.pose[2] > self.target_pos[2]:
#             reward += 10
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        
        self.last_pose = np.copy(self.sim.pose)
        
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            #pose_all.append(self.sim.v)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        #state = np.concatenate((self.sim.pose,self.sim.v))
        #state = np.concatenate([state] * self.action_repeat)
        self.last_pose = np.copy(self.init_pose)
        return state