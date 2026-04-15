"""
A* 粗轨迹引导特征计算模块

将 A* 生成的粗轨迹 (coarse_traj) 转换为 RL 智能体可用的引导特征向量，
融入观察空间使其直接"看到"路径方向，而不是仅通过微弱的奖励信号间接学习。
"""

import math
import numpy as np


class AStarGuide:
    """
    管理 A* 粗轨迹，计算 8 维引导特征向量
    
    特征设计：
    - 到目标的方向 (cos, sin) 和距离 (归一化)
    - 到亚目标的方向 (cos, sin) 和距离 (归一化)
    - 路径对齐度 (车辆朝向与路径切线方向的一致性)
    - 剩余路径长度比 (反映导航进度)
    """
    
    FEATURE_DIM = 8
    SUB_GOAL_LOOKAHEAD = 5      # 提前看5个waypoint
    SUB_GOAL_MIN_DIST = 2.0     # 亚目标最小距离（米）
    MAX_DIST_NORM = 20.0        # 距离归一化最大值（米）
    
    def __init__(self, coarse_traj, vehicle_state, dest_state, start_state=None):
        """
        初始化引导器
        
        Args:
            coarse_traj: AStarPlanner.plan() 返回的点列表 [(x,y), ...]
            vehicle_state: 当前车辆状态 (State 对象)
            dest_state: 目标状态 (State 对象，含 loc 和 heading)
            start_state: 起始状态 (可选，用于计算路径长度比)
        """
        self.traj = coarse_traj
        self.dest_loc = (dest_state.loc.x, dest_state.loc.y)
        self.dest_heading = dest_state.heading
        self.start_state = start_state
        
        # 预计算一些静态属性
        self.traj_len = len(coarse_traj) if coarse_traj else 0
        
        # 计算起点到目标的直线距离（用于归一化）
        if start_state is not None:
            vx, vy = vehicle_state.loc.x, vehicle_state.loc.y
            self.start_dest_dist = math.hypot(vx - self.dest_loc[0], vy - self.dest_loc[1])
        else:
            self.start_dest_dist = self.MAX_DIST_NORM
        
        # 初始化特征
        self.features = np.zeros(self.FEATURE_DIM, dtype=np.float64)
        
        if coarse_traj and len(coarse_traj) >= 2:
            self.update(vehicle_state)
    
    def update(self, vehicle_state):
        """
        每步更新引导特征

        倒车模式处理：泊车场景中车辆经常需要倒车沿A*轨迹行驶。
        此时 v_heading 指向路径反方向，如果不做处理会导致所有方向特征
        （亚目标方向、路径对齐度）全部算反，使车辆越沿路径倒车、特征越告诉它"掉头"。
        通过检测 v_heading 与路径方向的夹角判断是否倒车，从而正确翻转特征。
        """
        if not self.traj or self.traj_len < 2:
            self.features = np.zeros(self.FEATURE_DIM, dtype=np.float64)
            return

        vx, vy = vehicle_state.loc.x, vehicle_state.loc.y
        v_heading = vehicle_state.heading
        vehicle_pos = (vx, vy)

        # ========== 1. 找到最近waypoint ==========
        nearest_idx, nearest_dist = self._find_nearest_waypoint(vehicle_pos)

        # ========== 2. 判断是否倒车模式 ==========
        # 计算路径在最近点处的前进方向（从 traj[nearest_idx] 指向下一个点）
        path_dir = self._get_path_dir(nearest_idx)
        # 车辆朝向与路径方向的夹角
        heading_diff = abs(self._normalize_angle(v_heading - path_dir))

        # 夹角接近 pi → 车辆朝向与路径前进方向相反 → 在倒车沿轨迹行驶
        REVERSE_THRESHOLD = math.pi * 0.6  # > 108 度视为倒车
        is_reversing = heading_diff > REVERSE_THRESHOLD

        # ========== 3. 目标特征（始终指向终点，与行驶方向无关）==========
        dx_goal = self.dest_loc[0] - vx
        dy_goal = self.dest_loc[1] - vy
        dist_to_goal = math.hypot(dx_goal, dy_goal)

        angle_to_goal = math.atan2(dy_goal, dx_goal) - v_heading
        angle_to_goal = self._normalize_angle(angle_to_goal)

        self.features[0] = math.cos(angle_to_goal)
        self.features[1] = math.sin(angle_to_goal)
        self.features[2] = min(dist_to_goal / self.MAX_DIST_NORM, 1.0)

        # ========== 4. 亚目标特征（根据倒车模式选择方向）==========
        sub_goal = self._find_sub_goal(nearest_idx, vehicle_pos, is_reversing)

        if sub_goal is not None:
            dx_sg = sub_goal[0] - vx
            dy_sg = sub_goal[1] - vy
            dist_to_sg = math.hypot(dx_sg, dy_sg)

            angle_to_sg = math.atan2(dy_sg, dx_sg) - v_heading
            angle_to_sg = self._normalize_angle(angle_to_sg)

            self.features[3] = math.cos(angle_to_sg)
            self.features[4] = math.sin(angle_to_sg)
            self.features[5] = min(dist_to_sg / self.MAX_DIST_NORM, 1.0)
        else:
            self.features[3] = self.features[0]
            self.features[4] = self.features[1]
            self.features[5] = self.features[2]

        # ========== 5. 路径对齐度（倒车模式下方向取反）==========
        path_alignment = self._calc_path_alignment(vehicle_pos, nearest_idx, v_heading, is_reversing)
        self.features[6] = path_alignment

        # ========== 6. 剩余路径长度比 ==========
        remaining_ratio = self._calc_remaining_path_ratio(nearest_idx, is_reversing)
        self.features[7] = remaining_ratio
    
    def _get_path_dir(self, nearest_idx):
        """获取路径在最近点处的前进方向角"""
        if nearest_idx >= self.traj_len - 1:
            # 在终点，用前一个点
            p_curr = self.traj[self.traj_len - 1]
            p_prev = self.traj[max(0, self.traj_len - 2)]
            return math.atan2(p_curr[1] - p_prev[1], p_curr[0] - p_prev[0])
        else:
            # 用当前点和下一个点
            p_curr = self.traj[nearest_idx]
            p_next = self.traj[nearest_idx + 1]
            return math.atan2(p_next[1] - p_curr[1], p_next[0] - p_curr[0])

    def _find_nearest_waypoint(self, vehicle_pos):
        """找到最近的waypoint及其索引"""
        min_dist = float('inf')
        nearest_idx = 0

        for i, wp in enumerate(self.traj):
            d = math.hypot(vehicle_pos[0] - wp[0], vehicle_pos[1] - wp[1])
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        return nearest_idx, min_dist

    def _find_sub_goal(self, nearest_idx, vehicle_pos, is_reversing):
        """
        找到亚目标。

        前行模式：从最近点往后看，引导车辆沿A*轨迹前进。
        倒车模式：从最近点往前看（走过的路径），引导车辆沿A*轨迹倒车。

        Args:
            nearest_idx: 最近waypoint索引
            vehicle_pos: 车辆位置 (x, y)
            is_reversing: 是否在倒车模式
        """
        if is_reversing:
            # 倒车模式：在 nearest_idx 之前找亚目标
            look_start = max(0, nearest_idx - self.SUB_GOAL_LOOKAHEAD)
            look_end = nearest_idx
            step = 1
            default = self.traj[0]  # 路径起点作为倒车时的亚目标
        else:
            # 前行模式：在 nearest_idx 之后找亚目标
            look_start = nearest_idx + 1
            look_end = min(nearest_idx + self.SUB_GOAL_LOOKAHEAD + 1, self.traj_len)
            step = 1
            default = self.traj[-1]  # 终点作为前行时的亚目标

        # 如果路径太短，直接返回默认
        if look_start >= look_end or look_start >= self.traj_len:
            return default

        # 找距离 >= SUB_GOAL_MIN_DIST 的候选点
        for i in range(look_start, look_end, step):
            sg = self.traj[i]
            d = math.hypot(vehicle_pos[0] - sg[0], vehicle_pos[1] - sg[1])
            if d >= self.SUB_GOAL_MIN_DIST:
                return sg

        # 没找到足够远的候选点
        return self.traj[look_end - 1]
    
    def _calc_path_alignment(self, vehicle_pos, nearest_idx, vehicle_heading, is_reversing):
        """
        计算路径对齐度。

        前行模式：车辆朝向与路径切线方向越一致，对齐度越高。
        倒车模式：车辆朝向（车头方向的反方向）与路径方向越一致，对齐度越高。

        返回 cos(夹角)，范围 [-1, 1]
        """
        if nearest_idx == 0:
            p_curr = self.traj[0]
            p_next = self.traj[min(1, self.traj_len - 1)]
        elif nearest_idx >= self.traj_len - 1:
            p_curr = self.traj[self.traj_len - 1]
            p_next = self.traj[max(0, self.traj_len - 2)]
        else:
            p_curr = self.traj[nearest_idx]
            p_next = self.traj[min(nearest_idx + 1, self.traj_len - 1)]

        path_dir = math.atan2(
            p_next[1] - p_curr[1],
            p_next[0] - p_curr[0]
        )

        if is_reversing:
            # 倒车：车辆实际运动方向 = 车头方向 + π
            # 对齐度 = cos(车头朝向与路径反向的夹角) = cos((v_heading+π) - path_dir)
            effective_dir = vehicle_heading + math.pi
        else:
            effective_dir = vehicle_heading

        angle_diff = self._normalize_angle(effective_dir - path_dir)
        return math.cos(angle_diff)
    
    def _calc_remaining_path_ratio(self, nearest_idx, is_reversing=False):
        """
        计算剩余路径长度占总路径长度的比例。

        前行模式：计算 nearest_idx 到终点的路径长度比。
        倒车模式：计算起点到 nearest_idx 的路径长度比（因为车辆在往回走）。
        """
        if self.traj_len < 2:
            return 0.0

        if is_reversing:
            # 倒车模式：计算从起点(0)到 nearest_idx 的长度占总长度的比例
            remaining_len = 0.0
            for i in range(0, nearest_idx):
                remaining_len += math.hypot(
                    self.traj[i+1][0] - self.traj[i][0],
                    self.traj[i+1][1] - self.traj[i][1]
                )
            total_len = 0.0
            for i in range(self.traj_len - 1):
                total_len += math.hypot(
                    self.traj[i+1][0] - self.traj[i][0],
                    self.traj[i+1][1] - self.traj[i][1]
                )
            if total_len < 1e-6:
                return 0.0
            # 返回已完成的比例（倒车走得越多，比例越接近1）
            completed_ratio = remaining_len / total_len
            return completed_ratio
        else:
            # 前行模式：从当前点到终点的路径长度
            remaining_len = 0.0
            for i in range(nearest_idx, self.traj_len - 1):
                remaining_len += math.hypot(
                    self.traj[i+1][0] - self.traj[i][0],
                    self.traj[i+1][1] - self.traj[i][1]
                )
            total_len = 0.0
            for i in range(self.traj_len - 1):
                total_len += math.hypot(
                    self.traj[i+1][0] - self.traj[i][0],
                    self.traj[i+1][1] - self.traj[i][1]
                )
            if total_len < 1e-6:
                return 1.0
            return min(remaining_len / total_len, 1.0)
    
    def _normalize_angle(self, angle):
        """将角度归一化到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def get_features(self) -> np.ndarray:
        """
        返回当前的 8 维引导特征向量
        
        Returns:
            np.ndarray: shape (8,), dtype float64
        """
        return self.features.copy()
    
    def is_valid(self) -> bool:
        """检查引导器是否有效（有可用轨迹）"""
        return self.traj is not None and self.traj_len >= 2
    
    def get_remaining_waypoints(self, vehicle_pos, max_count=5):
        """
        获取车辆前方的剩余waypoint列表（用于可视化或调试）
        
        Args:
            vehicle_pos: (x, y)
            max_count: 最大返回数量
            
        Returns:
            list of (x, y) tuples
        """
        if not self.traj:
            return []
        
        nearest_idx, _ = self._find_nearest_waypoint(vehicle_pos)
        start = min(nearest_idx + 1, self.traj_len - 1)
        end = min(start + max_count, self.traj_len)
        
        return self.traj[start:end]
