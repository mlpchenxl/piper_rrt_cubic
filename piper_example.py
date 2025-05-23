import mujoco_viewer
import mujoco,time,threading
import numpy as np
import pinocchio
from mpl_toolkits.mplot3d import Axes3D
import itertools
import ikpy.chain
import transformations as tf

from pyroboplan.core.utils import (
    get_random_collision_free_state,
    extract_cartesian_poses,
)
from pyroboplan.models.piper import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.trajectory.trajectory_optimization import (
    CubicTrajectoryOptimization,
    CubicTrajectoryOptimizationOptions,
)

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=180, elevation=-30)
        self.path = path

        self.my_chain = ikpy.chain.Chain.from_urdf_file("./assets/piper_n.urdf")
    
    def runBefore(self, q_start, target_position, target_orientation_euler):
        # Create models and data
        self.model_roboplan, self.collision_model, visual_model = load_models(use_sphere_collisions=True)
        
        if self.collision_model is None:
            print(f"error !!! ")

        # return
        add_self_collisions(self.model_roboplan, self.collision_model)
        add_object_collisions(self.model_roboplan, self.collision_model, visual_model, inflation_radius=0.1)

        print(self.model_roboplan.names[1:])
        print([f.name for f in self.model_roboplan.frames])

        self.target_frame = "link6"
        np.set_printoptions(precision=3)
        self.distance_padding = 0.001

        self.init_state = self.data.qpos.copy()

        # 通过 ik 计算目标关节角度
        # 初始关节角默认为0
        target_orientation = tf.euler_matrix(*target_orientation_euler)[:3, :3]
        # 添加显示
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler('xyz', target_orientation_euler)
        rot_mat = rot.as_matrix()
        arrow_rot = np.eye(3)
        arrow_rot[:, 0] = rot_mat[:, 0]  # X 轴方向作为箭头方向

        # 使用 self.handle.user_scn.ngeom 获取当前已添加的几何体数量
        geom_id = self.handle.user_scn.ngeom

        # 添加一个小球可视化位置
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[geom_id],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0],  # 球半径为 0.01
            pos=target_position,
            mat=np.eye(3).flatten(),  # 不旋转
            rgba=np.array([0, 1, 0, 1])  # 绿色
        )
        geom_id += 1

        # 添加一个箭头表示姿态的X方向（红色）
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[geom_id],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[0.05, 0.003, 0.003],
            pos=target_position,
            mat=arrow_rot.flatten(),
            rgba=np.array([1, 0, 1, 1])  # 红色
        )
        geom_id += 1
        

        # 更新总几何体数量
        self.handle.user_scn.ngeom = geom_id
        # 计算逆运动学解
        target_joint_angles = self.my_chain.inverse_kinematics(target_position, target_orientation, "all")
        q_goal = np.array(target_joint_angles[1:])


        while True:            
            if q_goal is None:
                raise RuntimeError("ik 求解失败")
            
            

            # q_goal = self.random_valid_state()
            print(f"q_start type : {type(q_start)}")
            print(f"q_start : {q_start}")
            print(f"q_goal : {q_goal}")

            # Search for a path
            options = RRTPlannerOptions(
                max_step_size=0.05,
                max_connection_dist=5.0,
                rrt_connect=False,
                bidirectional_rrt=True,
                rrt_star=True,
                max_rewire_dist=5.0,
                max_planning_time=20.0,
                fast_return=True,
                goal_biasing_probability=0.15,
                collision_distance_padding=0.01,
            )
            print("")
            print(f"Planning a path...")
            planner = RRTPlanner(self.model_roboplan, self.collision_model, options=options)
            q_path = planner.plan(q_start, q_goal)
            if len(q_path) > 0:
                print(f"Got a path with {len(q_path)} waypoints")
            else:
                print("Failed to plan.")

            # Perform trajectory optimization.
            dt = 0.025
            options = CubicTrajectoryOptimizationOptions(
                num_waypoints=len(q_path),
                samples_per_segment=7,
                min_segment_time=0.5,
                max_segment_time=10.0,
                min_vel=-1.5,
                max_vel=1.5,
                min_accel=-0.75,
                max_accel=0.75,
                min_jerk=-1.0,
                max_jerk=1.0,
                max_planning_time=30.0,
                check_collisions=True,
                min_collision_dist=self.distance_padding,
                collision_influence_dist=0.05,
                collision_avoidance_cost_weight=0.0,
                collision_link_list=[
                    # "obstacle_box_1",
                    # "obstacle_box_2",
                    # "obstacle_sphere_1",
                    # "obstacle_sphere_2",
                    "ground_plane",
                    "link6",
                ],
            )
            print("Optimizing the path...")
            optimizer = CubicTrajectoryOptimization(self.model_roboplan, self.collision_model, options)
            traj = optimizer.plan([q_path[0], q_path[-1]], init_path=q_path)

            if traj is None:
                print("Retrying with all the RRT waypoints...")
                traj = optimizer.plan(q_path, init_path=q_path)

            if traj is not None:
                print("Trajectory optimization successful")
                traj_gen = traj.generate(dt)
                self.q_vec = traj_gen[1]
                print(f"path has {self.q_vec.shape[1]} points")
                self.tforms = extract_cartesian_poses(self.model_roboplan, "link6", self.q_vec.T)
                # 提取位置信息
                positions = []
                print(self.tforms[0].translation)
                print(self.tforms[0].rotation)
                self.handle.user_scn.ngeom = 0
                i = 0
                print(f"")
                for i, tform in enumerate(self.tforms):
                    if i % 2 == 0:
                        continue
                    position = tform.translation
                    rotation_matrix = tform.rotation
                    mujoco.mjv_initGeom(
                        self.handle.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.005, 0, 0],
                        pos=np.array([tform.translation[0], tform.translation[1], tform.translation[2]]),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 1])
                    )
                    i += 1
                self.handle.user_scn.ngeom = i
                print(f"Added {i} spheres to the scene.")
                for tform in self.tforms:
                    position = tform.translation
                    positions.append(position)

                positions = np.array(positions)
                break
        
        self.index = 0
        

    def random_valid_state(self):

        res =  get_random_collision_free_state(
            self.model_roboplan, self.collision_model, distance_padding=0.01
        )
        print("collisionPairs after SRDF :", len(self.collision_model.collisionPairs))
        return res
        

    def runFunc(self):
        self.data.qpos[:6] = self.q_vec[:6, self.index]
        self.index += 1
        if self.index >= self.q_vec.shape[1]:
            self.index = 0
        time.sleep(0.01)
        

if __name__ == "__main__":
    test = Test("./piper_model/agilex_piper/scene.xml")
    q_start = np.zeros(6)
    target_position = np.array([0.40069003, 0.09497203, 0.1113601])
    target_orientation_euler = np.array([0.0, 2.6, 0.0])
    test.run_loop(q_start, target_position, target_orientation_euler)

    