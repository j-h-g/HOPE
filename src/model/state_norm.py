
import numpy as np

DEFAULT_UPDATE_MODAL = {'img':False, 'lidar':True, 'target':True, 'action_mask':False, 'astar_guide':True}

class StateNorm():
    def __init__(self, observation_shape:dict, update_modal:dict=DEFAULT_UPDATE_MODAL) -> None:
        self.observation_shape = observation_shape
        self.update_modal = update_modal
        self.n_state = 0
        self.state_mean, self.S, self.state_std = {}, {}, {}
        for obs_type in self.observation_shape.keys():
            self.state_mean[obs_type] = np.zeros(self.observation_shape[obs_type], dtype=np.float32)
            self.S[obs_type] = np.zeros(self.observation_shape[obs_type], dtype=np.float32)
            self.state_std[obs_type] = np.sqrt(self.S[obs_type])
        self.fixed = False

    def fix_parameters(self,):
        self.fixed = True
    
    def init_state_norm(self, mean, std, S, n_state):
        self.n_state = n_state
        self.mean, self.std, self.S = mean, std, S

    def state_norm(self, observation: dict, update=False):
        if self.n_state == 0:
            self.n_state += 1
            for obs_type in self.observation_shape.keys():
                if self.update_modal[obs_type]:
                    self.state_mean[obs_type] = observation[obs_type]
                    self.state_std[obs_type] = observation[obs_type]
                    observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (self.state_std[obs_type] + 1e-8)
        elif update==False or self.fixed:
            for obs_type in self.observation_shape.keys():
                if self.update_modal[obs_type]:
                    observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (self.state_std[obs_type] + 1e-8)
        elif update==True:
            self.n_state += 1
            for obs_type in self.observation_shape.keys():
                if self.update_modal[obs_type]:
                    old_mean = self.state_mean[obs_type].copy()
                    self.state_mean[obs_type] = old_mean + (observation[obs_type] - old_mean) / self.n_state
                    self.S[obs_type] = self.S[obs_type] + (observation[obs_type] - old_mean) *\
                        (observation[obs_type] - self.state_mean[obs_type])
                    self.state_std[obs_type] = np.sqrt(self.S[obs_type] / self.n_state)
                    observation[obs_type] = (observation[obs_type] - self.state_mean[obs_type]) / (self.state_std[obs_type] + 1e-8)
        return observation


class ZoneStateNorm:
    """
    V1 基础版本：按距离分区，分别归一化
    - 分区边界固定：[2.0, 5.0, 10.0] (单位：米)
    - 每个 Zone 维护独立的 running_mean / running_std
    - 样本硬分配到某一 Zone（Zone 边界处存在跳变，待 V2 修复）
    - 初期 EMA 方差保护较简单（待 V2 增强）
    """

    def __init__(self, observation_shape: dict,
                 zone_boundaries: list = [2.0, 5.0, 10.0],
                 update_modal: dict = DEFAULT_UPDATE_MODAL):
        self.observation_shape = observation_shape
        self.zone_boundaries = np.array(zone_boundaries)
        self.K = len(zone_boundaries) + 1
        self.update_modal = update_modal

        self.running_mean = {
            z: {k: None for k in observation_shape.keys()}
            for z in range(self.K)
        }
        self.running_std = {
            z: {k: None for k in observation_shape.keys()}
            for z in range(self.K)
        }
        self.count = {z: 0 for z in range(self.K)}

    def _get_zone_id(self, distance: np.ndarray) -> np.ndarray:
        """
        将距离数组硬分配到 Zone id

        输入: distance [N, ]  每样本到终点距离（米）
        返回: zone_ids [N, ]  每样本所属 Zone id (0 ~ K-1)
        """
        distance = np.asarray(distance).flatten()
        zone_ids = np.full_like(distance, fill_value=self.K - 1, dtype=int)
        for z, boundary in enumerate(self.zone_boundaries):
            zone_ids[distance < boundary] = z
        return zone_ids

    def _concat_obs(self, obs_list: list) -> dict:
        """将 list of obs_dict concat 为单个 obs_dict (沿 axis=0)"""
        if len(obs_list) == 0:
            return {}
        obs_types = obs_list[0].keys()
        merged = {}
        for obs_type in obs_types:
            stacked = np.stack([obs[obs_type] for obs in obs_list], axis=0)
            merged[obs_type] = stacked
        return merged

    def update(self, observation: dict, zone_ids: np.ndarray):
        """
        对外接口：外部手动触发统计量更新（兼容旧调用方式）
        """
        self._update_stats(observation, zone_ids)

    def _update_stats(self, observation: dict, zone_ids: np.ndarray):
        """
        内部逻辑：基于 Welford 在线算法 + EMA 混合更新各 Zone 的 running 统计量。
        注意：只更新统计量，不做归一化。
        """
        if isinstance(observation, list):
            obs_types = observation[0].keys()
            merged = {}
            for obs_type in obs_types:
                merged[obs_type] = np.concatenate(
                    [o[obs_type] for o in observation], axis=0
                )
            observation = merged

        zone_ids = np.asarray(zone_ids).flatten()
        for obs_type in self.observation_shape.keys():
            x = observation[obs_type]
            if len(x.shape) == 1:
                x = x[np.newaxis, :]
            for z in range(self.K):
                mask = (zone_ids == z)
                if mask.sum() == 0:
                    continue
                x_zone = x[mask]
                x_mean = x_zone.mean(axis=0, keepdims=True)
                x_std = x_zone.std(axis=0, keepdims=True) + 1e-8

                self.count[z] += mask.sum()
                beta = 0.01

                if self.running_mean[z][obs_type] is None:
                    self.running_mean[z][obs_type] = x_mean
                    self.running_std[z][obs_type] = x_std
                else:
                    self.running_mean[z][obs_type] = (
                        (1 - beta) * self.running_mean[z][obs_type]
                        + beta * x_mean
                    )
                    self.running_std[z][obs_type] = np.sqrt(
                        (1 - beta) * self.running_std[z][obs_type]**2
                        + beta * x_std**2
                    )

    def normalize(self, observation: dict, zone_ids: np.ndarray, update: bool = True) -> dict:
        """
        主归一化方法：按 Zone 归一化，同时（可选）更新统计量。

        修复说明：
        - 修复问题 A：移除全局 all(count<2) early return，改为 per-zone None 检查
        - 修复问题 B：在归一化前先调用 _update_stats()，确保统计量更新和归一化
          使用同一批 concat 数据（s + s'），避免数据不一致

        Args:
            observation: obs_dict 或 list of obs_dict（通常是 concat 后的 [s; s']）
            zone_ids:    [N, ] 每样本的 Zone id
            update:      是否在归一化前同步更新统计量
        Returns:
            normed_observation: 归一化后的 obs_dict（copy，不原地修改）
        """
        if isinstance(observation, list):
            obs_types = observation[0].keys()
            N = sum(o[list(obs_types)[0]].shape[0] for o in observation)
            merged = {}
            for obs_type in obs_types:
                merged[obs_type] = np.concatenate(
                    [o[obs_type] for o in observation], axis=0
                )
            observation = merged
            if len(zone_ids) != N:
                raise ValueError(f"zone_ids length {len(zone_ids)} != observation size {N}")

        # 修复问题 B：先更新统计量，再归一化（统一使用 concat 后的数据）
        if update:
            self._update_stats(observation, zone_ids)

        normed = {}
        for obs_type in self.observation_shape.keys():
            x = observation[obs_type]
            if len(x.shape) == 1:
                x = x[np.newaxis, :]
            zone_ids_arr = np.asarray(zone_ids).flatten()

            normed[obs_type] = np.zeros_like(x)
            for z in range(self.K):
                mask = (zone_ids_arr == z)
                if mask.sum() == 0:
                    continue
                # 修复问题 A：per-zone None 检查替代全局 all(count<2)
                # 统计量未就绪时降级返回原值，不影响其他 Zone
                if self.running_mean[z][obs_type] is None:
                    normed[obs_type][mask] = x[mask]
                else:
                    normed[obs_type][mask] = (
                        x[mask] - self.running_mean[z][obs_type]
                    ) / (self.running_std[z][obs_type] + 1e-8)
        return normed