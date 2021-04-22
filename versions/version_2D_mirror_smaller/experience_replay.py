# Module imports
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory

class ExperienceReplay(object):
    def __init__(self, agent, enviroment, batch_size):
        self._replay_buffer = TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=enviroment.batch_size,
            max_length=50000)

        self._random_policy = RandomTFPolicy(enviroment.time_step_spec(),
                                                enviroment.action_spec())

        self._fill_buffer(enviroment, self._random_policy, steps=100)

        self.dataset = self._replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2,
            single_deterministic_pass=False
        ).prefetch(3)

        self.iterator = iter(self.dataset)

    def _fill_buffer(self, enviroment, policy, steps):
        for _ in range(steps):
            self.timestamp_data(enviroment, policy)

    def _timestamp_data(self, environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)


        if next_time_step.reward > 0:
            timestamp_trajectory = trajectory.from_transition(time_step, action_step, next_time_step)

            self._replay_buffer.add_batch(timestamp_trajectory)

    def timestamp_data(self, environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        timestamp_trajectory = trajectory.from_transition(time_step, action_step, next_time_step)

        self._replay_buffer.add_batch(timestamp_trajectory)
