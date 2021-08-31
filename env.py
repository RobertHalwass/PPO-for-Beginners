import habitat
from habitat import Config

class PointNavEnv(habitat.RLEnv):

    def __init__(self, config: Config):
        super().__init__(config=config)
        self.previous_pointgoal = None

    def get_reward_range(self):
        return [-50, 50]

    def get_reward(self, observations):
        pointgoal = observations["pointgoal_with_gps_compass"]

        if self.habitat_env.episode_over:
            spl = self.habitat_env.get_metrics()["spl"]
            reward = 2.5 * spl
        else:
            delta_geo_dist = pointgoal[0] - self.previous_pointgoal[0]
            reward = - delta_geo_dist - 0.01

        self.previous_pointgoal = pointgoal
        return reward

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return {"Info": "No Info"}

    def reset(self):
        observations = self.habitat_env.reset()
        self.previous_pointgoal = observations["pointgoal_with_gps_compass"]
        return observations