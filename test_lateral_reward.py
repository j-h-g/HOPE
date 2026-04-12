"""
测试 A* 横向偏差引导奖励的计算
验证 _calc_lateral_distance() 和奖励函数是否正常工作
"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

import numpy as np
from env.car_parking_base import CarParking


def test_lateral_distance():
    """测试横向距离计算"""
    print("=" * 60)
    print("测试 A* 横向偏差引导奖励")
    print("=" * 60)

    env = CarParking(
        render_mode=None,
        use_img_observation=False,
        use_lidar_observation=True,
        use_action_mask=False,
        verbose=True
    )

    for episode in range(3):
        print(f"\n--- Episode {episode + 1}/3 ---")
        obs = env.reset(level='Normal')

        # 检查 A* 轨迹是否生成
        if env.coarse_traj is None:
            print("[WARNING] A* 轨迹生成失败!")
            continue

        print(f"A* 轨迹点数: {len(env.coarse_traj)}")
        print(f"起点: ({env.coarse_traj[0][0]:.2f}, {env.coarse_traj[0][1]:.2f})")
        print(f"终点: ({env.coarse_traj[-1][0]:.2f}, {env.coarse_traj[-1][1]:.2f})")

        # 测试横向距离计算
        lateral_dist = env._calc_lateral_distance()
        if lateral_dist is not None:
            print(f"初始横向距离: {lateral_dist:.3f} 米")

            # 计算对应的奖励值
            from configs import ASTAR_GUIDE_REWARD_WEIGHT, ASTAR_LATERAL_DECAY, ASTAR_MAX_LATERAL_DIST
            clipped_dist = min(lateral_dist, ASTAR_MAX_LATERAL_DIST)
            expected_reward = ASTAR_GUIDE_REWARD_WEIGHT * np.exp(-clipped_dist / ASTAR_LATERAL_DECAY)
            print(f"预期引导奖励: {expected_reward:.4f}")
        else:
            print("[WARNING] 无法计算横向距离!")

        # 模拟几步动作，获取奖励信息
        print("\n模拟 5 步动作:")
        for step in range(5):
            action = np.array([0.0, 0.5])  # 直行
            obs, reward_info, status, info = env.step(action)
            print(f"  Step {step+1}: lateral_guide_reward = {reward_info['lateral_guide_reward']:.4f}")

            if status.value != 1:  # CONTINUE = 1
                print(f"  Episode 结束，状态: {status}")
                break

    env.close()
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


def test_reward_decay():
    """测试奖励衰减特性"""
    print("\n" + "=" * 60)
    print("测试奖励衰减曲线")
    print("=" * 60)

    from configs import ASTAR_GUIDE_REWARD_WEIGHT, ASTAR_LATERAL_DECAY, ASTAR_MAX_LATERAL_DIST

    print(f"参数: weight={ASTAR_GUIDE_REWARD_WEIGHT}, decay={ASTAR_LATERAL_DECAY}, max_dist={ASTAR_MAX_LATERAL_DIST}")
    print("\n横向距离 -> 奖励值:")
    print("-" * 40)

    for dist in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        clipped = min(dist, ASTAR_MAX_LATERAL_DIST)
        reward = ASTAR_GUIDE_REWARD_WEIGHT * np.exp(-clipped / ASTAR_LATERAL_DECAY)
        print(f"  {dist:5.1f} 米 -> {reward:.4f}")


if __name__ == "__main__":
    test_lateral_distance()
    test_reward_decay()
