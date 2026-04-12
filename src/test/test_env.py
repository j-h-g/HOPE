import sys
import os

# 自动获取当前脚本所在的绝对路径，并把包含 env 的目录强行塞进 Python 的环境变量里
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# 如果你把脚本放在了 HOPE 项目最外层，加上这句就能找到 src 里的 env 了
sys.path.append(os.path.join(current_dir, 'src')) 

import time
import pygame
from env.car_parking_base import CarParking

def test_coarse_trajectory():
    print("正在初始化环境，强制开启可视化 (render_mode='human')...")
    
    # 初始化环境：开启人类可视化，务必关闭图像观测 (use_img_observation=False)
    env = CarParking(
        render_mode="human", 
        use_img_observation=False, 
        use_lidar_observation=True,
        use_action_mask=True
    )
    
    # 循环测试 5 个不同的随机场景
    for episode in range(5):
        print(f"\n--- 正在生成第 {episode + 1}/5 个场景 ---")
        
        # 使用 'Extrem' 或 'dlp' 级别来专门测试复杂的死胡同场景
        obs = env.reset(level='Extrem') 
        print("环境重置完成，A* 轨迹已生成！请查看弹出窗口。")
        
        # 在当前场景停留几秒钟，方便你用肉眼观察轨迹
        for step in range(60): 
            env.render()
            # ================= 新增：直接把渲染出来的画面存成图片 =================
            if step == 5: # 只截取第 5 步的画面（这时候场景已经完全加载好了）
                # 获取当前的渲染表面
                screen_surface = pygame.display.get_surface()
                # 拼接图片保存路径
                img_path = f"debug_episode_{episode+1}.png"
                # 将画面保存到当前目录下
                pygame.image.save(screen_surface, img_path)
                print(f"成功将场景 {episode+1} 的画面保存为 {img_path} ！快去文件夹里看看！")
            # ======================================================================
            
            # 这一段非常重要：防止 pygame 窗口卡死
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("手动关闭窗口，退出测试。")
                    env.close()
                    return
                    
            time.sleep(0.05)
            
    env.close()
    print("测试完毕！")

if __name__ == "__main__":
    test_coarse_trajectory()