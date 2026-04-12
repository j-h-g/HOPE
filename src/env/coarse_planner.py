import math
import heapq
import numpy as np
from shapely.geometry import Point

class AStarPlanner:
    def __init__(self, map_data, grid_resolution=0.5, safe_margin=0.5):
        """
        初始化 A* 规划器
        修改：默认 safe_margin 降为 0.5 米，避免堵死狭窄通道
        """
        self.map = map_data
        self.reso = grid_resolution
        self.safe_margin = safe_margin
        
        self.min_x = self.map.xmin
        self.max_x = self.map.xmax
        self.min_y = self.map.ymin
        self.max_y = self.map.ymax
        
        self.motions = [
            (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))
        ]

    def _calc_grid_index(self, node):
        return (int(round(node[0] / self.reso)), int(round(node[1] / self.reso)))

    def _is_collision(self, x, y, is_endpoint=False):
        """利用 Shapely 检查某个坐标点是否碰撞"""
        if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
            return True
            
        point = Point(x, y)
        
        # 核心修改：如果是起点或终点，使用 0.1 米的极小容忍度，防止目标点被膨胀的墙壁吞噬
        margin = 0.1 if is_endpoint else self.safe_margin
        
        for obstacle in self.map.obstacles:
            if obstacle.shape.buffer(margin).contains(point):
                return True
        return False

    def plan(self, start_state, dest_state):
        start_node = (start_state.loc.x, start_state.loc.y)
        goal_node = (dest_state.loc.x, dest_state.loc.y)

        # 核心修改：判断起点和终点时，传入 is_endpoint=True
        if self._is_collision(start_node[0], start_node[1], is_endpoint=True) or \
           self._is_collision(goal_node[0], goal_node[1], is_endpoint=True):
            print("[A* Planner] Start or Goal is in collision!")
            return None

        open_set = []
        heapq.heappush(open_set, (0.0, start_node))
        
        came_from = dict()
        g_score = {self._calc_grid_index(start_node): 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            curr_idx = self._calc_grid_index(current)

            if math.hypot(current[0] - goal_node[0], current[1] - goal_node[1]) <= self.reso:
                return self._reconstruct_path(came_from, current, start_node)

            for dx, dy, cost in self.motions:
                neighbor = (current[0] + dx * self.reso, current[1] + dy * self.reso)
                neighbor_idx = self._calc_grid_index(neighbor)

                # 核心修改：如果邻居节点靠近终点，给予同样的端点特权豁免
                is_end = math.hypot(neighbor[0] - goal_node[0], neighbor[1] - goal_node[1]) <= self.reso
                if self._is_collision(neighbor[0], neighbor[1], is_endpoint=is_end):
                    continue

                tentative_g_score = g_score[curr_idx] + cost * self.reso

                if neighbor_idx not in g_score or tentative_g_score < g_score[neighbor_idx]:
                    came_from[neighbor_idx] = current
                    g_score[neighbor_idx] = tentative_g_score
                    h_score = math.hypot(neighbor[0] - goal_node[0], neighbor[1] - goal_node[1])
                    f_score = tentative_g_score + h_score
                    heapq.heappush(open_set, (f_score, neighbor))

        print("[A* Planner] Path not found!")
        return None

    def _reconstruct_path(self, came_from, current, start_node):
        path = [current]
        curr_idx = self._calc_grid_index(current)
        while curr_idx in came_from:
            current = came_from[curr_idx]
            path.append(current)
            curr_idx = self._calc_grid_index(current)
            if current == start_node:
                break
        path.reverse()
        return path