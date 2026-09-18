import os
import ctypes
import contextlib
from dataclasses import dataclass, asdict, field

import numpy as np


def _preload_acados_libs():
    acados_dir = os.environ.get("ACADOS_SOURCE_DIR",
                                os.path.expanduser("~/acados"))
    libdir = os.path.join(acados_dir, "lib")
    for name in ("libblasfeo.so", "libhpipm.so", "libqpOASES_e.so",
                 "libdaqp.so", "libosqp.so", "libacados.so"):
        path = os.path.join(libdir, name)
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

_preload_acados_libs()

import pybullet as p
import pybullet_data
import pinocchio as pin
import casadi as ca
from scipy.spatial.transform import Rotation
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

USE_WORLD_FRAME_YAW = False


class Config:
    N = 10
    Tf = 0.25
    dt_control = Tf / N
    dt_physics = dt_control / 10
    num_physics_steps = 10

    ABAD_L, HIP_L, KNEE_L, WHEEL_L = 1, 2, 3, 4
    ABAD_R, HIP_R, KNEE_R, WHEEL_R = 5, 6, 7, 8
    LEG_JOINTS = [ABAD_L, HIP_L, KNEE_L, ABAD_R, HIP_R, KNEE_R]
    ALL_JOINTS = [1, 2, 3, 4, 5, 6, 7, 8]


    Kp_abad_fb, Kd_abad_fb = 60, 15
    Kp_hip_fb, Kd_hip_fb = 30, 15
    Kp_knee_fb, Kd_knee_fb = 20, 10

    Ki_leg = np.array([10.0, 1.0, 1.0])
    I_LEG_CLAMP = np.array([15.0, 5.0, 5.0])

    ALPHA_ACC_FILTER = np.array([0., 0., 0.])
    ISM_PHASE = {1: 0, 2: 0, 5:0, 10:0}
    state_weights = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    lambda_min = np.array([40, 20, 50, .1, 0.5,
                           40, 20, 50, .1, 0.5])
    W_gains = np.array([10, 10, 50, .5, 1,
                        10, 10, 50, .5, 1])

    K_AW = 0.3
    ALPHA_VEL_FILTER = np.array([0., 0., 0.])
    VEL_CLIP = np.array([0.05, 0.05, 0.05])
    ACC_CLIP = np.array([.1, .1, .1])
    target_vx = 1.5
    acceleration_x = .5
    target_pz = 0.55
    K_wheel = np.array([[25, 3],
                        [25, 3]])
    wheel_radius = 0.1429

    URDF_PATH = "TRON1A/WF_TRON1A/urdf/robot.urdf"
    m = 22.27
    I_xx = 0.527
    I_yy = 0.432
    I_zz = 0.400
    r_L = np.array([-0.0480, 0.150, 0.0019])
    r_R = np.array([-0.0480, -0.150, 0.0019])


def _setup_casadi_states():
    names = ['px', 'py', 'pz', 'roll', 'pitch', 'yaw',
             'vx', 'vy', 'vz', 'roll_rate', 'pitch_rate', 'yaw_rate']
    states = ca.vertcat(*[ca.SX.sym(n) for n in names])
    return states, int(states.size1())


def _setup_casadi_controls():
    names = ['fx_L', 'fy_L', 'fz_L', 'ny_L', 'nx_L',
             'fx_R', 'fy_R', 'fz_R', 'ny_R', 'nx_R']
    controls = ca.vertcat(*[ca.SX.sym(n) for n in names])
    return controls, int(controls.size1())


def _setup_casadi_params():
    m, g = ca.SX.sym('m'), ca.SX.sym('g')
    I_xx, I_yy, I_zz = ca.SX.sym('I_xx'), ca.SX.sym('I_yy'), ca.SX.sym('I_zz')
    r_hip_L = ca.SX.sym('r_hip_L', 3)
    r_hip_R = ca.SX.sym('r_hip_R', 3)
    r_wheel_L_y = ca.SX.sym('r_wheel_L_y')
    r_wheel_R_y = ca.SX.sym('r_wheel_R_y')
    params = ca.vertcat(m, g, I_xx, I_yy, I_zz, r_hip_L, r_hip_R,
                        r_wheel_L_y, r_wheel_R_y)
    return params, int(params.size1())


def get_robot_params():
    return {
        'm': Config.m, 'g': 9.81,
        'I_xx': Config.I_xx, 'I_yy': Config.I_yy, 'I_zz': Config.I_zz,
        'r_hip_L': Config.r_L,
        'r_hip_R': Config.r_R,
        'r_wheel_L_y': 0.15,
        'r_wheel_R_y': -0.15,
    }


def _param_vector(r_wheel_L_y=None, r_wheel_R_y=None):
    rp = get_robot_params()
    if r_wheel_L_y is None:
        r_wheel_L_y = rp['r_wheel_L_y']
    if r_wheel_R_y is None:
        r_wheel_R_y = rp['r_wheel_R_y']
    return np.array([
        rp['m'], rp['g'], rp['I_xx'], rp['I_yy'], rp['I_zz'],
        rp['r_hip_L'][0], rp['r_hip_L'][1], rp['r_hip_L'][2],
        rp['r_hip_R'][0], rp['r_hip_R'][1], rp['r_hip_R'][2],
        r_wheel_L_y, r_wheel_R_y
    ], dtype=float)


def _rotation_matrix(roll, pitch, yaw, cos=ca.cos, sin=ca.sin,
                     stack_row=ca.horzcat, stack_col=ca.vertcat):
    c_r, s_r = cos(roll), sin(roll)
    c_p, s_p = cos(pitch), sin(pitch)
    c_y, s_y = cos(yaw), sin(yaw)
    return stack_col(
        stack_row(c_y * c_p, c_y * s_p * s_r - s_y * c_r, c_y * s_p * c_r + s_y * s_r),
        stack_row(s_y * c_p, s_y * s_p * s_r + c_y * c_r, s_y * s_p * c_r - c_y * s_r),
        stack_row(-s_p, c_p * s_r, c_p * c_r),
    )


def _dynamics_model(state, inputs, pd):
    m, g = pd['m'], pd['g']
    I_xx, I_yy, I_zz = pd['I_xx'], pd['I_yy'], pd['I_zz']
    r_hip_L, r_hip_R = pd['r_hip_L'], pd['r_hip_R']
    r_wheel_L_y, r_wheel_R_y = pd['r_wheel_L_y'], pd['r_wheel_R_y']

    (_, _, _, roll, pitch, yaw,
     vx, vy, vz, omega_x, omega_y, omega_z) = (state[i] for i in range(12))
    (fx_L, fy_L, fz_L, ny_L, nx_L,
     fx_R, fy_R, fz_R, ny_R, nx_R) = (inputs[i] for i in range(10))

    c_r, s_r = ca.cos(roll), ca.sin(roll)
    c_p = ca.cos(pitch)
    c_y, s_y = ca.cos(yaw), ca.sin(yaw)

    R = _rotation_matrix(roll, pitch, yaw)

    omega_skew = ca.vertcat(
        ca.horzcat(0, -omega_z, omega_y),
        ca.horzcat(omega_z, 0, -omega_x),
        ca.horzcat(-omega_y, omega_x, 0)
    )
    R_dot = R @ omega_skew

    pitch_dot = -R_dot[2, 0] / c_p
    roll_dot = (R_dot[2, 1] * c_r - R_dot[2, 2] * s_r) / c_p
    yaw_dot = (R_dot[1, 0] * c_y - R_dot[0, 0] * s_y) / c_p

    f_world_L = R @ ca.vertcat(fx_L, fy_L, fz_L)
    f_world_R = R @ ca.vertcat(fx_R, fy_R, fz_R)
    f_world = f_world_L + f_world_R

    v_dot_x = f_world[0] / m
    v_dot_y = f_world[1] / m
    v_dot_z = f_world[2] / m - g

    tau_roll_L = (r_hip_L[1] * fz_L) + (r_hip_L[2] * fy_L) + nx_L
    tau_pitch_L = -(r_hip_L[2] * fx_L) - (r_hip_L[0] * fz_L) + ny_L
    tau_roll_R = (r_hip_R[1] * fz_R) + (r_hip_R[2] * fy_R) + nx_R
    tau_pitch_R = -(r_hip_R[2] * fx_R) - (r_hip_R[0] * fz_R) + ny_R

    if USE_WORLD_FRAME_YAW:
        s_p = ca.sin(pitch)
        f_head_L = c_p * fx_L - s_p * c_r * fz_L + s_p * s_r * fy_L
        f_head_R = c_p * fx_R - s_p * c_r * fz_R + s_p * s_r * fy_R
        tau_yaw_L = (r_wheel_L_y * f_head_L) - (r_hip_L[0] * fy_L)
        tau_yaw_R = (r_wheel_R_y * f_head_R) - (r_hip_R[0] * fy_R)
    else:
        tau_yaw_L = (r_wheel_L_y * fx_L) - (r_hip_L[0] * fy_L)
        tau_yaw_R = (r_wheel_R_y * fx_R) - (r_hip_R[0] * fy_R)

    tau_x = tau_roll_L + tau_roll_R
    tau_y = tau_pitch_L + tau_pitch_R
    tau_z = tau_yaw_L + tau_yaw_R

    omega_x_dot = (tau_x + (I_yy - I_zz) * omega_y * omega_z) / I_xx
    omega_y_dot = (tau_y + (I_zz - I_xx) * omega_z * omega_x) / I_yy
    omega_z_dot = (tau_z + (I_xx - I_yy) * omega_x * omega_y) / I_zz

    return ca.vertcat(vx, vy, vz, roll_dot, pitch_dot, yaw_dot,
                      v_dot_x, v_dot_y, v_dot_z,
                      omega_x_dot, omega_y_dot, omega_z_dot)


def create_acados_model():
    model = AcadosModel()
    model.name = ("full_3d_biped_mpc_mc_head" if USE_WORLD_FRAME_YAW
                  else "full_3d_biped_mpc_mc_body")
    states, nx = _setup_casadi_states()
    controls, _ = _setup_casadi_controls()
    params_sym, _ = _setup_casadi_params()
    model.x, model.u, model.p = states, controls, params_sym

    roll, pitch, yaw = states[3], states[4], states[5]
    (fx_L, fy_L, fz_L, ny_L, nx_L,
     fx_R, fy_R, fz_R, ny_R, nx_R) = (controls[i] for i in range(10))

    pd = {
        'm': params_sym[0], 'g': params_sym[1],
        'I_xx': params_sym[2], 'I_yy': params_sym[3], 'I_zz': params_sym[4],
        'r_hip_L': params_sym[5:8], 'r_hip_R': params_sym[8:11],
        'r_wheel_L_y': params_sym[11], 'r_wheel_R_y': params_sym[12],
    }

    model.xdot = ca.SX.sym('xdot', nx, 1)
    f_expl = _dynamics_model(states, controls, pd)
    model.f_expl_expr = f_expl
    model.f_impl_expr = model.xdot - f_expl


    R = _rotation_matrix(roll, pitch, yaw)
    f_world_L = R @ ca.vertcat(fx_L, fy_L, fz_L)
    f_world_R = R @ ca.vertcat(fx_R, fy_R, fz_R)

    mu = 0.6
    model.con_h_expr = ca.vertcat(
        f_world_L[0] ** 2 + f_world_L[1] ** 2 - (mu * f_world_L[2]) ** 2,
        f_world_R[0] ** 2 + f_world_R[1] ** 2 - (mu * f_world_R[2]) ** 2
    )
    return model, 2


def build_ocp(model, nh, x0, Tf, N):
    ocp = AcadosOcp()
    ocp.model = model
    nx, nu = int(model.x.size1()), int(model.u.size1())
    np_ = int(model.p.size1())
    ocp.dims.N, ocp.dims.nx, ocp.dims.nu = int(N), nx, nu
    ocp.dims.np, ocp.dims.nh = np_, int(nh)

    ocp.solver_options.tf = float(Tf)
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.qp_solver_tol_stat = 1e-5
    ocp.solver_options.qp_solver_tol_eq = 1e-5
    ocp.solver_options.qp_solver_tol_ineq = 1e-5
    ocp.solver_options.qp_solver_tol_comp = 1e-5
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.sim_method_num_stages = 4
    ocp.constraints.x0 = x0.astype(float)
    ocp.parameter_values = _param_vector()

    F_fwd, F_lat, F_vert = 50.0, 50.0, 200.0
    Tau_pitch, Tau_roll = 10, 10
    ocp.constraints.lbu = np.array([
        -F_fwd, -F_lat, 1e-5, -Tau_pitch, -Tau_roll,
        -F_fwd, -F_lat, 1e-5, -Tau_pitch, -Tau_roll], dtype=float)
    ocp.constraints.ubu = np.array([
        F_fwd, F_lat, F_vert, Tau_pitch, Tau_roll,
        F_fwd, F_lat, F_vert, Tau_pitch, Tau_roll], dtype=float)
    ocp.constraints.idxbu = np.arange(nu, dtype=int)

    if nh > 0:
        ocp.constraints.lh = -1e10 * np.ones(nh)
        ocp.constraints.uh = np.zeros(nh)
        ocp.constraints.idxsh = np.arange(nh)
        ocp.cost.zl = 1e4 * np.ones(nh)
        ocp.cost.zu = 1e4 * np.ones(nh)
        ocp.cost.Zl = 1e2 * np.ones(nh)
        ocp.cost.Zu = 1e2 * np.ones(nh)

    Q = np.diag([500, 2000, 2000, 2000, 1000, 100,
                 50, 500, 50, 5, 5, 1]).astype(float)
    Q_terminal = np.diag([100, 5000, 5000, 5000, 2000, 100,
                          10, 1000, 10, 10, 10, 1]).astype(float)
    R = np.diag([0.1, 1, 0.01, 1, 1, 0.1, 1, 0.01, 1, 1])

    ny = nx + nu
    W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.W = W.astype(float)
    ocp.cost.W_e = Q_terminal
    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx:, :nu] = np.eye(nu)
    ocp.cost.Vx, ocp.cost.Vu, ocp.cost.Vx_e = Vx, Vu, np.eye(nx)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((nx,))

    json_file = f'acados_ocp_{model.name}.json'
    acados_install_path = os.environ.get('ACADOS_INSTALL_DIR')
    if acados_install_path:
        ocp.acados_include_path = os.path.join(acados_install_path, 'include')
        ocp.acados_lib_path = os.path.join(acados_install_path, 'lib')
    return ocp, json_file


def solve_ocp(ocp_solver, x_current, x_ref_list,
              r_wheel_L_y=None, r_wheel_R_y=None):
    N = int(ocp_solver.N)
    nu = int(ocp_solver.acados_ocp.dims.nu)
    p_values = _param_vector(r_wheel_L_y, r_wheel_R_y)

    for i in range(N + 1):
        ocp_solver.set(i, 'p', p_values)
    ocp_solver.set(0, 'lbx', x_current.astype(float))
    ocp_solver.set(0, 'ubx', x_current.astype(float))
    for k in range(N):
        yref_k = np.concatenate([x_ref_list[k].astype(float),
                                 np.zeros(nu, dtype=float)])
        ocp_solver.set(k, 'yref', yref_k)
    ocp_solver.set(N, 'yref', x_ref_list[N].astype(float))

    if ocp_solver.solve() != 0:
        return None, None
    u0 = ocp_solver.get(0, 'u')
    x_pred = np.array([ocp_solver.get(i, 'x') for i in range(N + 1)],
                      dtype=float)
    return u0, x_pred


def fix_urdf_mesh_paths(urdf_path):
    with open(urdf_path, 'r') as f:
        content = f.read()
    fixed = content.replace(
        'robot_description/pointfoot/WF_TRON1A/meshes/', 'meshes/')
    fixed_path = urdf_path.replace('.urdf', '_fixed.urdf')
    if not (os.path.exists(fixed_path) and open(fixed_path).read() == fixed):
        with open(fixed_path, 'w') as f:
            f.write(fixed)
    return fixed_path

def get_robot_state(robot_id):
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    base_vel, base_ang_vel = p.getBaseVelocity(robot_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)
    R = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
    omega_body = R.T @ np.array(base_ang_vel)
    return np.array([
        base_pos[0], base_pos[1], base_pos[2], roll, pitch, yaw,
        base_vel[0], base_vel[1], base_vel[2],
        omega_body[0], omega_body[1], omega_body[2]])

def get_floor_height_below_com(robot_id, ray_length=10.0, offset_above=0.05):
    com_pos, _ = p.getBasePositionAndOrientation(robot_id)
    ray = p.rayTest([com_pos[0], com_pos[1], com_pos[2] + offset_above],
                    [com_pos[0], com_pos[1], com_pos[2] - ray_length])[0]
    if ray[0] >= 0:
        return ray[3][2]
    return 0.0


def get_joint_states(robot_id, joint_indices):
    js = p.getJointStates(robot_id, joint_indices)
    return [s[0] for s in js], [s[1] for s in js]


def get_end_effector_positions(robot_id):
    pos_L = p.getLinkState(robot_id, Config.WHEEL_L)[0]
    pos_R = p.getLinkState(robot_id, Config.WHEEL_R)[0]
    return np.array(pos_L), np.array(pos_R)


def measure_ground_wrench(robot_id, ground_id,
                          links=(Config.WHEEL_L, Config.WHEEL_R)):
    f = np.zeros(3)
    for link in links:
        for c in p.getContactPoints(bodyA=robot_id, bodyB=ground_id,
                                    linkIndexA=link):
            f += c[9] * np.array(c[7], dtype=float)
            f += c[10] * np.array(c[11], dtype=float)
            f += c[12] * np.array(c[13], dtype=float)
    return f


def initiate_torque_control(robot_id, joint_indices, joint_friction=0.0):
    tau_f = float(joint_friction)
    for j in joint_indices:
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL,
                                targetVelocity=0.0, force=tau_f)


def apply_joint_torque(robot_id, joint_index, torque, max_torque=50.0):
    safe = float(np.clip(torque, -max_torque, max_torque))
    p.setJointMotorControl2(robot_id, joint_index, p.TORQUE_CONTROL, force=safe)


def follow_robot_camera(robot_id, distance=2.0, yaw=65, pitch=-20):
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=yaw,
                                 cameraPitch=pitch, cameraTargetPosition=pos)


SS_START = 6.0

_PIN_MODEL = None


def _pin_model():
    global _PIN_MODEL
    if _PIN_MODEL is None:
        _PIN_MODEL = pin.buildModelFromUrdf(Config.URDF_PATH,
                                            pin.JointModelFreeFlyer())
    return _PIN_MODEL


def generate_reference_trajectory(current_time, dt, N, robot_id,
                                  lean_angle=0.0):
    DESIRED_YAW = 0.0
    c_d, s_d = np.cos(DESIRED_YAW), np.sin(DESIRED_YAW)

    t_acc = Config.target_vx / (Config.acceleration_x + 1e-9)
    px_acc = 0.5 * Config.acceleration_x * t_acc ** 2

    wheel_pos_L = np.array(p.getLinkState(robot_id, Config.WHEEL_L)[0])
    wheel_pos_R = np.array(p.getLinkState(robot_id, Config.WHEEL_R)[0])
    wheel_mid = 0.5 * (wheel_pos_L + wheel_pos_R)
    e_lat = -s_d * wheel_mid[0] + c_d * wheel_mid[1]

    x_ref_list = []
    for k in range(N + 1):
        t = current_time + k * dt
        if t < t_acc:
            s = 0.5 * Config.acceleration_x * t ** 2
            v = Config.acceleration_x * t
        else:
            s = px_acc + Config.target_vx * (t - t_acc)
            v = Config.target_vx

        x_ref_list.append(np.array([
            s * c_d - e_lat * s_d, s * s_d + e_lat * c_d, Config.target_pz,
            0.0, lean_angle, DESIRED_YAW,
            v * c_d, v * s_d, 0.0,
            0.0, 0.0, 0.0]))
    return x_ref_list


def calculate_delta_x(u_optimal, x_current, robot_id):
    fx_L, fz_L = u_optimal[0], u_optimal[2]
    fx_R, fz_R = u_optimal[5], u_optimal[7]
    wL, wR = get_end_effector_positions(robot_id)
    z_drop = x_current[2]- 0.5 * (wL[2] + wR[2])
    return np.clip((z_drop* (fx_L + fx_R)) / (fz_L + fz_R + 1e-6),
                   -0.075, 0.075)

def calculate_lean_angle(robot_id, d_x,x_current):
    wL, wR = get_end_effector_positions(robot_id)
    z_drop = x_current[2]- 0.5 * (wL[2] + wR[2])
    return float(np.arctan2(-d_x, max(z_drop, 1e-3)))


def compute_individual_wheel_positions(v_x, v_y, yaw_rate, roll, roll_rate):
    return 0.15, -0.15


def solve_ik_for_legs(robot_id, delta_x, x_current):
    com_pos, com_orn_quat = p.getBasePositionAndOrientation(robot_id)
    com_pos = np.array(com_pos)
    R = np.array(p.getMatrixFromQuaternion(com_orn_quat)).reshape(3, 3)
    body_x = R[:, 0]
    body_y = R[:, 1]

    wheel_midpoint = com_pos - (delta_x * body_x)

    r_wheel_L_y, r_wheel_R_y = compute_individual_wheel_positions(
        x_current[6], x_current[7], x_current[11], x_current[3], x_current[9])

    target_pos_L = wheel_midpoint + r_wheel_L_y * body_y
    target_pos_R = wheel_midpoint + r_wheel_R_y * body_y


    def _ground_under(pt, skip_id=robot_id):
        top, bottom = pt[2] + 0.5, pt[2] - 10.0
        for _ in range(5):
            ray = p.rayTest([pt[0], pt[1], top],
                            [pt[0], pt[1], bottom])[0]
            if ray[0] < 0:
                return 0.0
            if ray[0] != skip_id:
                return ray[3][2]
            top = ray[3][2] - 1e-3
        return 0.0
    z_wheel_L = _ground_under(target_pos_L) + Config.wheel_radius
    z_wheel_R = _ground_under(target_pos_R) + Config.wheel_radius

    target_pos_L[2] = z_wheel_L
    target_pos_R[2] = z_wheel_R

    ik_L = np.array(p.calculateInverseKinematics(robot_id, Config.WHEEL_L,
                                                 target_pos_L))
    ik_R = np.array(p.calculateInverseKinematics(robot_id, Config.WHEEL_R,
                                                 target_pos_R))
    return np.array([ik_L[0:3], ik_R[4:7]])


def get_current_joint_states(robot_id):
    q_L, qd_L = get_joint_states(
        robot_id, [Config.ABAD_L, Config.HIP_L, Config.KNEE_L])
    q_R, qd_R = get_joint_states(
        robot_id, [Config.ABAD_R, Config.HIP_R, Config.KNEE_R])
    return (np.array([np.array(q_L), np.array(q_R)]),
            np.array([np.array(qd_L), np.array(qd_R)]))


def calculate_contact_jacobian_for_legs(robot_id):
    num_joints = p.getNumJoints(robot_id)
    movable_joints, qs, qds, qdds = [], [], [], []
    for j in range(num_joints):
        if p.getJointInfo(robot_id, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            movable_joints.append(j)
            st = p.getJointState(robot_id, j)
            qs.append(st[0])
            qds.append(st[1])
            qdds.append(0.0)

    jacobians = {}
    for leg_name, wheel_link in zip(['L', 'R'], [Config.WHEEL_L, Config.WHEEL_R]):
        try:
            Jv, _ = p.calculateJacobian(robot_id, wheel_link,
                                        [0.0, 0.0, 0.0], qs, qds, qdds)
            Jv = np.array(Jv)
            if leg_name == 'L':
                leg_joints = [Config.ABAD_L, Config.HIP_L,
                              Config.KNEE_L, Config.WHEEL_L]
            else:
                leg_joints = [Config.ABAD_R, Config.HIP_R,
                              Config.KNEE_R, Config.WHEEL_R]
            cols = [movable_joints.index(j) for j in leg_joints]
            abad_col, hip_col, knee_col, _ = cols

            sgn = 1.0 if leg_name == 'L' else -1.0

            J = np.zeros((5, 3))
            J[0, 0] = Jv[0, 6 + abad_col]
            J[0, 1] = Jv[0, 6 + hip_col]
            J[0, 2] = Jv[0, 6 + knee_col]
            J[1, 0] = Jv[1, 6 + abad_col]
            J[1, 1] = Jv[1, 6 + hip_col]
            J[1, 2] = Jv[1, 6 + knee_col]
            J[2, 0] = Jv[2, 6 + abad_col]
            J[2, 1] = Jv[2, 6 + hip_col]
            J[2, 2] = Jv[2, 6 + knee_col]
            J[3, 1] = sgn
            J[4, 0] = sgn
            jacobians[f'J_{leg_name}'] = J
        except Exception:
            jacobians[f'J_{leg_name}'] = np.zeros((5, 3))
    return jacobians


_PB_TO_PIN = [(1, 7), (2, 8), (3, 9), (4, 10),
              (5, 11), (6, 12), (7, 13), (8, 14)]

DEG_JOINTS = {Config.WHEEL_L: 0.8, Config.KNEE_L: 0.8, Config.KNEE_R: 0.8}
JOINT_FRICTION = .5


def compute_gravity_compensation(robot_id, x_current):
    try:
        model = _pin_model()
        data = model.createData()
        q = np.zeros(model.nq)
        q[0:3] = x_current[0:3]
        q[3:7] = Rotation.from_euler('xyz', x_current[3:6]).as_quat()
        jp = {j: p.getJointState(robot_id, j)[0] for j in Config.ALL_JOINTS}
        for pb, pi in _PB_TO_PIN:
            q[pi] = jp[pb]
        pin.computeGeneralizedGravity(model, data, q)
        G = data.g[6:]
        return {1: G[0], 2: G[1], 3: G[2], 4: G[3],
                5: G[4], 6: G[5], 7: G[6], 8: G[7]}
    except Exception:
        return {j: 0.0 for j in Config.ALL_JOINTS}


def calculate_feedforward_torques(target_angles, desired_accs, robot_id):
    joint_indices = Config.LEG_JOINTS
    tgt = target_angles.flatten()
    acc = desired_accs.flatten()

    num_joints = p.getNumJoints(robot_id)
    movable, positions = [], []
    for j in range(num_joints):
        if p.getJointInfo(robot_id, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            movable.append(j)
            if j in joint_indices:
                positions.append(tgt[joint_indices.index(j)])
            else:
                positions.append(p.getJointState(robot_id, j)[0])

    M_full = np.array(p.calculateMassMatrix(robot_id, positions))
    num_base = M_full.shape[0] - len(movable)
    idx = [movable.index(j) + num_base for j in joint_indices]
    M = M_full[np.ix_(idx, idx)]
    return (M @ acc).reshape(2, 3)


def calculate_coriolis_torques(robot_id, target_positions,
                               target_velocities, x_current):
    try:
        model = _pin_model()
        data = model.createData()
        q = np.zeros(model.nq)
        v = np.zeros(model.nv)
        q[0:3] = x_current[0:3]
        q[3:7] = Rotation.from_euler('xyz', x_current[3:6]).as_quat()

        jp, jv = {}, {}
        for j in Config.ALL_JOINTS:
            st = p.getJointState(robot_id, j)
            jp[j], jv[j] = st[0], st[1]
        tgt_p = target_positions.flatten()
        tgt_v = target_velocities.flatten()
        for k, j in enumerate(Config.LEG_JOINTS):
            jp[j] = tgt_p[k]
            jv[j] = tgt_v[k]
        for pb, pi in _PB_TO_PIN:
            q[pi] = jp[pb]
            v[pi - 1] = jv[pb]

        pin.computeCoriolisMatrix(model, data, q, v)
        C_v = (data.C @ v)[6:]
        return np.array([C_v[0], C_v[1], C_v[2],
                         C_v[4], C_v[5], C_v[6]]).reshape(2, 3)
    except Exception:
        return np.zeros((2, 3))


def compute_leg_torques(robot_id, u_optimal, x_current, delta_x,
                        prev_target_angles, prev_desired_velocities,
                        prev_desired_accs, prev_int_err,
                        dt_physics, i, physics_step,
                        wbc_full=True, wbc_moments=True, wbc_gravity=True):
    target_angles = solve_ik_for_legs(robot_id, delta_x, x_current)
    current_positions, current_velocities = get_current_joint_states(robot_id)

    STANDUP_T = 0.5
    BLEND_T = 0.5
    t_now = i * Config.dt_control
    in_standup = t_now < STANDUP_T
    w_blend = float(np.clip((t_now - STANDUP_T) / BLEND_T, 0.0, 1.0))

    if in_standup or (i == 0 and physics_step == 0):
        desired_velocities_filtered = np.zeros((2, 3))
    elif physics_step == 0:
        desired_velocities_filtered = prev_desired_velocities
    else:
        desired_velocities_noisy = (target_angles - prev_target_angles) / dt_physics
        desired_velocities_filtered = (
            Config.ALPHA_VEL_FILTER * prev_desired_velocities
            + (1.0 - Config.ALPHA_VEL_FILTER) * desired_velocities_noisy)

    if in_standup or (i == 0 and physics_step == 0):
        desired_accelerations_filtered = np.zeros((2, 3))
    elif physics_step == 0:
        desired_accelerations_filtered = prev_desired_accs
    else:
        desired_accelerations_raw = (
            desired_velocities_filtered - prev_desired_velocities) / dt_physics
        desired_accelerations_filtered = (
            Config.ALPHA_ACC_FILTER * prev_desired_accs
            + (1.0 - Config.ALPHA_ACC_FILTER) * desired_accelerations_raw)

    if wbc_full:
        vel_ref_used = np.clip(w_blend * desired_velocities_filtered,
                               -Config.VEL_CLIP, Config.VEL_CLIP)
        acc_ref_used = np.clip(w_blend * desired_accelerations_filtered,
                               -Config.ACC_CLIP, Config.ACC_CLIP)
    else:
        vel_ref_used = np.zeros((2, 3))
        acc_ref_used = np.zeros((2, 3))

    angle_errors = target_angles - current_positions
    vel_errors = -current_velocities + vel_ref_used

    if in_standup or (i == 0 and physics_step == 0):
        int_err = np.zeros((2, 3))
    else:
        int_err = np.clip(
            prev_int_err + angle_errors * dt_physics,
            -Config.I_LEG_CLAMP / Config.Ki_leg,
            Config.I_LEG_CLAMP / Config.Ki_leg)
    tau_i = Config.Ki_leg * int_err

    if wbc_gravity:
        G = compute_gravity_compensation(robot_id, x_current)
    else:
        G = {j: 0.0 for j in Config.ALL_JOINTS}

    roll, pitch, yaw = x_current[3], x_current[4], x_current[5]
    fx_L, fy_L, fz_L, ny_L, nx_L = u_optimal[0:5]
    fx_R, fy_R, fz_R, ny_R, nx_R = u_optimal[5:10]

    J = calculate_contact_jacobian_for_legs(robot_id)

    R_body_to_world = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    F_world_L = R_body_to_world @ np.array([fx_L, fy_L, fz_L])
    F_world_R = R_body_to_world @ np.array([fx_R, fy_R, fz_R])

    fx_wheel_magnitude_L = fx_L * np.cos(pitch) - fz_L * np.sin(pitch)
    fx_wheel_magnitude_R = fx_R * np.cos(pitch) - fz_R * np.sin(pitch)
    wheel_direction_world = np.array([np.cos(yaw), np.sin(yaw), 0])
    wheel_force_world_L = fx_wheel_magnitude_L * wheel_direction_world
    wheel_force_world_R = fx_wheel_magnitude_R * wheel_direction_world

    F_leg_world_L = F_world_L - wheel_force_world_L
    F_leg_world_R = F_world_R - wheel_force_world_R

    if not wbc_moments:
        ny_L_map, nx_L_map, ny_R_map, nx_R_map = 0.0, 0.0, 0.0, 0.0
    else:
        ny_L_map, nx_L_map, ny_R_map, nx_R_map = ny_L, nx_L, ny_R, nx_R

    FL = np.array([F_leg_world_L[0], F_leg_world_L[1], F_leg_world_L[2],
                   ny_L_map, nx_L_map])
    FR = np.array([F_leg_world_R[0], F_leg_world_R[1], F_leg_world_R[2],
                   ny_R_map, nx_R_map])

    tauL = J['J_L'].T @ FL
    tauR = J['J_R'].T @ FR

    if (i == 0 and physics_step == 0) or not wbc_full:
        tau_M = np.zeros((2, 3))
        tau_C = np.zeros((2, 3))
    else:
        tau_M = calculate_feedforward_torques(target_angles, acc_ref_used, robot_id)
        tau_C = calculate_coriolis_torques(robot_id, target_angles,
                                           current_velocities, x_current)

    tau_pd = np.array([
        [Config.Kp_abad_fb * angle_errors[0, 0] + Config.Kd_abad_fb * vel_errors[0, 0],
         Config.Kp_hip_fb * angle_errors[0, 1] + Config.Kd_hip_fb * vel_errors[0, 1],
         Config.Kp_knee_fb * angle_errors[0, 2] + Config.Kd_knee_fb * vel_errors[0, 2]],
        [Config.Kp_abad_fb * angle_errors[1, 0] + Config.Kd_abad_fb * vel_errors[1, 0],
         Config.Kp_hip_fb * angle_errors[1, 1] + Config.Kd_hip_fb * vel_errors[1, 1],
         Config.Kp_knee_fb * angle_errors[1, 2] + Config.Kd_knee_fb * vel_errors[1, 2]]
    ]) + tau_i

    total_tau = np.array([
        [-tauL[0] + G[1] + tau_M[0, 0] + tau_C[0, 0] + tau_pd[0, 0],
         -tauL[1] + G[2] + tau_M[0, 1] + tau_C[0, 1] + tau_pd[0, 1],
         -tauL[2] + G[3] + tau_M[0, 2] + tau_C[0, 2] + tau_pd[0, 2]],
        [-tauR[0] + G[5] + tau_M[1, 0] + tau_C[1, 0] + tau_pd[1, 0],
         -tauR[1] + G[6] + tau_M[1, 1] + tau_C[1, 1] + tau_pd[1, 1],
         -tauR[2] + G[7] + tau_M[1, 2] + tau_C[1, 2] + tau_pd[1, 2]]
    ])

    return {
        'total': total_tau.flatten(),
        'forces_world': [F_world_L, F_world_R],
        'prev_target_angles': target_angles.copy(),
        'prev_desired_velocities': desired_velocities_filtered.copy(),
        'prev_desired_accs': acc_ref_used.copy(),
        'prev_int_err': int_err.copy(),
    }
def compute_wheel_torques(x_current, robot_id, FLX, FRX, FLZ, FRZ,
                          theta_prev, i, dt, dx):
    wheel_pos_L = np.array(p.getLinkState(robot_id, Config.WHEEL_L)[0])
    wheel_pos_R = np.array(p.getLinkState(robot_id, Config.WHEEL_R)[0])
    wheel_midpoint = (wheel_pos_L + wheel_pos_R) / 2

    com_pos = x_current[0:3]
    z_body = com_pos[2] - wheel_midpoint[2]
    pitch = x_current[4]
    yaw = x_current[5]

    tau_ff_L = -Config.wheel_radius * (FLX * np.cos(pitch) - FLZ * np.sin(pitch))
    tau_ff_R = -Config.wheel_radius * (FRX * np.cos(pitch) - FRZ * np.sin(pitch))

    offset_world = com_pos - wheel_midpoint
    R_w2b = Rotation.from_euler('xyz', [0, 0, yaw]).as_matrix().T
    dx_body = (R_w2b @ offset_world)[0]

    theta = np.arctan2(dx_body, z_body)
    theta_dot = (theta - theta_prev) / dt if i > 0 else 0.0
    theta_desired = np.arctan2(dx, z_body)

    fb = Config.K_wheel @ np.array([theta - theta_desired, theta_dot])
    return tau_ff_L + fb[0], tau_ff_R + fb[1], theta


def compute_ISM_matrices(roll_c, pitch_c, yaw_c, x_current, robot_id):
    m = Config.m
    I_xx, I_yy, I_zz = Config.I_xx, Config.I_yy, Config.I_zz
    r_x_L, r_y_L, r_z_L = Config.r_L
    r_x_R, r_y_R, r_z_R = Config.r_R

    wL, wR = get_end_effector_positions(robot_id)
    com_pos = x_current[0:3]
    R_w2b = Rotation.from_euler('xyz', x_current[3:6]).as_matrix().T
    r_wheel_L_y = float((R_w2b @ (wL - com_pos))[1])
    r_wheel_R_y = float((R_w2b @ (wR - com_pos))[1])

    c_r, s_r = np.cos(roll_c), np.sin(roll_c)
    c_p, s_p = np.cos(pitch_c), np.sin(pitch_c)
    c_y, s_y = np.cos(yaw_c), np.sin(yaw_c)

    B = np.zeros((12, 10))
    B[6, 0] = (c_y * c_p) / m
    B[6, 1] = (c_y * s_p * s_r - s_y * c_r) / m
    B[6, 2] = (c_y * s_p * c_r + s_y * s_r) / m
    B[6, 5] = (c_y * c_p) / m
    B[6, 6] = (c_y * s_p * s_r - s_y * c_r) / m
    B[6, 7] = (c_y * s_p * c_r + s_y * s_r) / m

    B[7, 0] = (s_y * c_p) / m
    B[7, 1] = (s_y * s_p * s_r + c_y * c_r) / m
    B[7, 2] = (s_y * s_p * c_r - c_y * s_r) / m
    B[7, 5] = (s_y * c_p) / m
    B[7, 6] = (s_y * s_p * s_r + c_y * c_r) / m
    B[7, 7] = (s_y * s_p * c_r - c_y * s_r) / m

    B[8, 0] = (-s_p) / m
    B[8, 1] = (c_p * s_r) / m
    B[8, 2] = (c_p * c_r) / m
    B[8, 5] = (-s_p) / m
    B[8, 6] = (c_p * s_r) / m
    B[8, 7] = (c_p * c_r) / m

    B[9, 1] = r_z_L / I_xx
    B[9, 2] = r_y_L / I_xx
    B[9, 4] = 1.0 / I_xx
    B[9, 6] = r_z_R / I_xx
    B[9, 7] = r_y_R / I_xx
    B[9, 9] = 1.0 / I_xx

    B[10, 0] = -r_z_L / I_yy
    B[10, 2] = -r_x_L / I_yy
    B[10, 3] = 1.0 / I_yy
    B[10, 5] = -r_z_R / I_yy
    B[10, 7] = -r_x_R / I_yy
    B[10, 8] = 1.0 / I_yy

    if USE_WORLD_FRAME_YAW:
        B[11, 0] = (r_wheel_L_y * c_p) / I_zz
        B[11, 1] = -r_x_L / I_zz + (r_wheel_L_y * s_p * s_r) / I_zz
        B[11, 2] = -(r_wheel_L_y * s_p * c_r) / I_zz
        B[11, 5] = (r_wheel_R_y * c_p) / I_zz
        B[11, 6] = -r_x_R / I_zz + (r_wheel_R_y * s_p * s_r) / I_zz
        B[11, 7] = -(r_wheel_R_y * s_p * c_r) / I_zz
    else:
        B[11, 0] = r_wheel_L_y / I_zz
        B[11, 1] = -r_x_L / I_zz
        B[11, 5] = r_wheel_R_y / I_zz
        B[11, 6] = -r_x_R / I_zz

    return B, B.T


@dataclass
class ControllerCfg:
    label: str = "nmpc_stsmc"
    use_ism: bool = True
    ism_every_n: int = 1
    wbc_full: bool = True
    wbc_moments: bool = True
    wbc_gravity: bool = True
    use_dob: bool = False
    dob_start: float = 1.5
    dob_ki_scale: float = 10
    dob_dhat_max: float = 100.0

    def __post_init__(self):
        assert not (self.use_ism and self.use_dob), \
            "use_ism and use_dob are mutually exclusive"


@dataclass
class TrialParams:
    seed: int = 0
    imp_time: float = 6.0
    imp_dir: float = np.pi
    imp_mag: float = 100.0
    imp_dur: float = 0.20
    mu_ground: float = 1.0
    gain_loss: dict = field(default_factory=dict)
    noise_att: float = 0.0
    noise_rate: float = 0.005
    noise_vel: float = 0.02
    noise_pos: float = 0.0
    att_bias: np.ndarray = None
    t_sim: float = 15.0
    tau_max: float = 50.0
    rand_start: float = 2.0
    rand_ramp: float = 1.0
    recovery_att_roll_band: float = 0.10
    recovery_att_pitch_band: float = 0.10
    recovery_att_hold: float = 0.5
    fall_check_start: float = 1.5
    fall_pitch: float = 1
    fall_roll: float = 1
    fall_pz: float = 0.35


    scenario: str = "impulse"
    joint_friction: float = 0.0
    ss_start: float = SS_START


    post_fall_run: float = 0.0

    terrain_seed: int = -1
    terrain_bump_height: float = 0.10
    terrain_mu_scale: float = 1.0
    metric_start_x: float = np.nan
    ref_target_vx: float = np.nan
    ref_acceleration_x: float = np.nan


PUSH_X = -0.08
PUSH_Z = 0.0


def compute_push_point(robot_id):
    shape = p.getCollisionShapeData(robot_id, -1)[0]
    dims, local_pos = shape[3], shape[5]
    half_width_y = dims[1] / 2.0
    return [PUSH_X, local_pos[1] + half_width_y, PUSH_Z]


NOMINAL_IMP_DIR = float(np.arctan2(-1.0, 0.0))
IMP_DIR_SPREAD = np.radians(30.0)


def draw_trial_params(seed: int) -> TrialParams:
    rng = np.random.default_rng(seed)
    return TrialParams(
        seed=seed,
        imp_time=float(rng.uniform(6, 8)),
        imp_dir=float(NOMINAL_IMP_DIR + rng.uniform(-IMP_DIR_SPREAD,
                                                    IMP_DIR_SPREAD)),
        imp_mag=float(rng.uniform(90.0, 110.0)),
        mu_ground=float(rng.uniform(0.7, 1.0)),
        gain_loss={j: float(rng.uniform(0.8, 1.0)) for j in Config.ALL_JOINTS},
        noise_rate=0.005,
        noise_vel=0.02,
        noise_pos=0.0,
    )


def nominal_trial_params() -> TrialParams:
    return TrialParams(
        seed=-1,
        imp_time=6.0,
        imp_dir=NOMINAL_IMP_DIR,
        imp_mag=100.0,
        imp_dur=0.2,
        mu_ground=1.0,
        gain_loss={},
        noise_rate=0.0,
        noise_vel=0.0,
        noise_pos=0.0,
    )


MU_NOMINAL = 1.0
GROUND_X_MIN, GROUND_X_MAX, GROUND_HALF_W = -15.0, 30.0, 25.0

TERRAIN_START_X = -1.3
TERRAIN_FLAT_LEN = 3.7
ROUGH_START_X = TERRAIN_START_X + TERRAIN_FLAT_LEN

TERRAIN_MU_SCALE_LO, TERRAIN_MU_SCALE_HI = 0.7, 1.
NOMINAL_TERRAIN_SEED = 0


def draw_degradation_params(seed: int) -> TrialParams:
    rng = np.random.default_rng(seed)
    gains = {}
    for j in Config.ALL_JOINTS:
        if j in DEG_JOINTS:
            gains[j] = float(rng.uniform(0.75, 0.85))
        else:
            gains[j] = float(rng.uniform(0.90, 1.00))
    return TrialParams(
        seed=seed,
        scenario="degradation",
        imp_time=1e9,
        imp_mag=0.0,
        fall_pz=0.425,
        gain_loss=gains,
        joint_friction=float(rng.uniform(0.40, 0.60)),
        mu_ground=float(rng.uniform(0.7, 1.0)),
        noise_rate=0.005,
        noise_vel=0.02,
        noise_pos=0.0,
        rand_start=0.0,
        rand_ramp=0.0,
    )


def nominal_degradation_params() -> TrialParams:
    return TrialParams(
        seed=-3,
        scenario="degradation",
        imp_time=1e9,
        imp_mag=0.0,
        fall_pz=0.425,
        gain_loss=dict(DEG_JOINTS),
        joint_friction=JOINT_FRICTION,
        mu_ground=1.0,
        noise_rate=0.0,
        noise_vel=0.0,
        noise_pos=0.0,
        rand_start=0.0,
        rand_ramp=0.0,
    )


def describe(tp: TrialParams = None) -> str:
    tp = tp or nominal_trial_params()
    degraded = ", ".join(f"j{j}={g:.2f}"
                         for j, g in sorted(tp.gain_loss.items()) if g < 1.0)
    return (f"actuator degradation | gains: {degraded or 'none'} "
            f"| joint friction={tp.joint_friction:.2f} Nm "
            f"| mu={tp.mu_ground:.2f} | from t={tp.rand_start:.1f}s "
            f"| ss window from t={tp.ss_start:.1f}s | t_sim={tp.t_sim:.1f}s")


def _build_variable_ground(rng: np.random.Generator,
                           bump_height: float = 0.10,
                           mu_scale: float = 1.0):
    PATCHES = [
        ("flat_start",   TERRAIN_FLAT_LEN, 1.0, 0.0),
        ("rough",        15.0, 1.0, 0.0),
    ]
    WIDTH_Y = 8.0
    START_X = TERRAIN_START_X
    COLORS = {"flat_start": (0.85, 0.85, 0.85, 1),
              "rough": (0.3, 0.3, 0.3, 1)}

    def flat_box(x_c, length, color, rest):
        col = p.createCollisionShape(p.GEOM_BOX,
                                     halfExtents=[length/2, WIDTH_Y/2, 0.05])
        vis = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[length/2, WIDTH_Y/2, 0.05],
                                  rgbaColor=color)
        pid = p.createMultiBody(0, col, vis, [x_c, 0, -0.05])
        p.changeDynamics(pid, -1, lateralFriction=1,
                         restitution=min(rest, 0.9))

    def rough_patch(x_c, length, mu,
                    width=10.0,
                    cell_x=.75,
                    cell_y=.25,
                    edge_ramp=0.20):
        nx = max(2, int(round(length / cell_x)) + 1)
        ny = max(2, int(round(width / cell_y)) + 1)

        xi = np.arange(nx) * cell_x
        yj = np.arange(ny) * cell_y
        tx = np.clip(np.minimum(xi, length - xi) / edge_ramp, 0, 1)
        ty = np.clip(np.minimum(yj, width - yj) / edge_ramp, 0, 1)
        taper = (tx[:, None] * ty[None, :]) ** 2

        h = rng.uniform(0, 1, (nx, ny)) * bump_height * taper

        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD,
            meshScale=[length / (nx - 1), width / (ny - 1), 1.0],
            heightfieldData=h.flatten(order='C').tolist(),
            numHeightfieldRows=nx, numHeightfieldColumns=ny)
        base_z = float((h.min() + h.max()) / 2.0) - (bump_height)
        hf = p.createMultiBody(0, shape, shape, [x_c, 0, base_z])
        p.changeDynamics(hf, -1, lateralFriction=mu * mu_scale)

    x = START_X
    for name, length, mu, rest in PATCHES:
        x_c = x + length/2
        if name == "rough":
            rough_patch(x_c, length, mu)
        else:
            flat_box(x_c, length, COLORS[name], rest)
        x += length


def draw_terrain_params(seed: int) -> TrialParams:
    if seed == NOMINAL_TERRAIN_SEED:
        bump, mu_scale = 0.10, 1.0
    else:
        _rng = np.random.default_rng(seed)
        bump = float(_rng.uniform(0.05, 0.15))
        mu_scale = float(_rng.uniform(TERRAIN_MU_SCALE_LO,
                                      TERRAIN_MU_SCALE_HI))
    return TrialParams(
        seed=seed,
        scenario="terrain",
        terrain_seed=seed,
        terrain_bump_height=bump,
        terrain_mu_scale=mu_scale,
        imp_time=1e9,
        imp_mag=0.0,
        mu_ground=MU_NOMINAL,
        gain_loss={},
        noise_att=0.0, noise_rate=0.0, noise_vel=0.0, noise_pos=0.0,
        att_bias=np.zeros(3),
        t_sim=15.0,
        metric_start_x=ROUGH_START_X,
        ref_target_vx=1.0,
        ref_acceleration_x=0.2,


        fall_check_start=2.5,
        fall_pitch=1,
        fall_roll=1,
        fall_pz=0.35,
    )


def nominal_terrain_params() -> TrialParams:
    return draw_terrain_params(NOMINAL_TERRAIN_SEED)


def _setup(tp: TrialParams, gui: bool):
    p.connect(p.GUI if gui else p.DIRECT, options="--opengl2")
    p.setTimeStep(1.0 / 400.0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(os.path.dirname(os.path.abspath(__file__)))
    p.setGravity(0, 0, -9.81)

    if tp.terrain_seed >= 0:
        _build_variable_ground(np.random.default_rng(tp.terrain_seed),
                               bump_height=tp.terrain_bump_height,
                               mu_scale=tp.terrain_mu_scale)
        gid = None
    else:
        half_len = (GROUND_X_MAX - GROUND_X_MIN) / 2.0
        x_center = (GROUND_X_MAX + GROUND_X_MIN) / 2.0
        half_thk = 0.05
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half_len, GROUND_HALF_W, half_thk])
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[half_len, GROUND_HALF_W, half_thk],
            rgbaColor=(0.8, 0.8, 0.8, 1.0))
        gid = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis,
                                basePosition=[x_center, 0, -half_thk])
        p.changeDynamics(gid, -1, lateralFriction=MU_NOMINAL)

    fixed_urdf = fix_urdf_mesh_paths(Config.URDF_PATH)
    robot_id = p.loadURDF(fixed_urdf, [0, 0, 0.59], useFixedBase=False)
    start = {0: 0., 1: 0., 2: -0.2, 3: 1.4, 4: 0.,
             5: 0., 6: 0.2, 7: -1.4, 8: 0.}
    for j, q in start.items():
        p.resetJointState(robot_id, j, q, targetVelocity=0.0)
    crouch = {0: 0., 1: 0., 2: -1.4, 3: 1.8, 4: 0.,
              5: 0., 6: 1.4, 7: -1.8, 8: 0.}
    for j, q in crouch.items():
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                targetPosition=q,
                                positionGain=0.5, velocityGain=0.5)
    for _ in range(240):
        p.stepSimulation()
    return robot_id, gid


def _ramp(tp: TrialParams, t: float) -> float:
    if tp.rand_ramp <= 0:
        return 1.0 if t >= tp.rand_start else 0.0
    return float(np.clip((t - tp.rand_start) / tp.rand_ramp, 0.0, 1.0))


def make_measurement_fn(tp: TrialParams, rng: np.random.Generator):
    bias = (np.zeros(3) if tp.att_bias is None
            else np.asarray(tp.att_bias, dtype=float))

    def measure(robot_id, t):
        x = get_robot_state(robot_id)
        r = _ramp(tp, t)
        if r > 0.0:
            x[0:3] += rng.normal(0.0, r * tp.noise_pos, 3)
            x[3:6] += r * bias
            x[6:9] += rng.normal(0.0, r * tp.noise_vel, 3)
            x[9:12] += rng.normal(0.0, r * tp.noise_rate, 3)
        return x
    return measure


def impulse_at(tp: TrialParams, t: float):
    if tp.imp_time <= t < tp.imp_time + tp.imp_dur:
        return [tp.imp_mag * np.cos(tp.imp_dir),
                tp.imp_mag * np.sin(tp.imp_dir), 0.0]
    return None


def apply_torques_with_loss(robot_id, torque_dict, gain_loss, tau_max,
                            sat_counter):
    for j, tau in torque_dict.items():
        tau_s = tau * gain_loss.get(j, 1.0)
        if abs(tau_s) > tau_max:
            sat_counter[0] += 1
        apply_joint_torque(robot_id, j, tau_s, max_torque=tau_max)


_OCP = None
_OCP_JSON = None
_OCP_BUILT = False


def _get_solver(x0):
    global _OCP, _OCP_JSON, _OCP_BUILT
    if _OCP is None:
        model, nh = create_acados_model()
        _OCP, _OCP_JSON = build_ocp(model, nh, x0, Config.Tf, Config.N)
    if not _OCP_BUILT:
        solver = AcadosOcpSolver(_OCP, json_file=_OCP_JSON)
        _OCP_BUILT = True
        return solver
    try:
        return AcadosOcpSolver(_OCP, json_file=_OCP_JSON,
                               build=False, generate=False)
    except TypeError:
        return AcadosOcpSolver(_OCP, json_file=_OCP_JSON)


@contextlib.contextmanager
def _silence_fds():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


_CFG_TARGET_VX0 = Config.target_vx
_CFG_ACCEL_X0 = Config.acceleration_x


def run_trial(seed: int, ctrl_cfg: ControllerCfg, tp: TrialParams = None,
              gui: bool = False, quiet: bool = None,
              keep_trace: bool = False) -> dict:
    tp = tp or draw_trial_params(seed)
    rng = np.random.default_rng(seed + 10_000)
    if quiet is None:
        quiet = not gui
    cm = _silence_fds() if quiet else contextlib.nullcontext()
    with cm:
        try:
            out = _run_trial_inner(tp, ctrl_cfg, rng, gui,
                                   keep_trace=keep_trace)
        except Exception as e:
            import traceback
            out = _nan_row(tp, ctrl_cfg)
            out["error"] = repr(e)
            out["traceback"] = traceback.format_exc()
        finally:
            if p.isConnected():
                p.disconnect()
    return out


def _run_trial_inner(tp: TrialParams, cfg: ControllerCfg, rng, gui,
                     keep_trace: bool = False) -> dict:
    import time as _time


    Config.target_vx = (tp.ref_target_vx
                        if np.isfinite(tp.ref_target_vx)
                        else _CFG_TARGET_VX0)
    Config.acceleration_x = (tp.ref_acceleration_x
                             if np.isfinite(tp.ref_acceleration_x)
                             else _CFG_ACCEL_X0)

    robot_id, ground_id = _setup(tp, gui)
    measure = make_measurement_fn(tp, rng)
    num_steps = int(tp.t_sim / Config.dt_control)
    p.setTimeStep(Config.dt_physics)

    push_point = compute_push_point(robot_id)

    x_current = get_robot_state(robot_id)
    solver = _get_solver(x_current)
    initiate_torque_control(robot_id, Config.ALL_JOINTS,
                            joint_friction=tp.joint_friction)

    prev_target_angles = np.zeros((2, 3))
    prev_desired_velocities = np.zeros((2, 3))
    prev_desired_accs = np.zeros((2, 3))
    theta_prev = 0.0
    sta_integral_term = np.zeros(10)
    F_ISM_hold = np.zeros(10)
    d_hat = np.zeros(10)
    F_DOB_hold = np.zeros(10)
    sat_counter = [0]
    solve_times = []
    mpc_fail_count = 0
    lean_filt = 0.0
    delta_x_filt = 0.0
    tau_wL_cmd = tau_wR_cmd = 0.0
    T, X_true, X_ref, F_des = [], [], [], []
    F_gr, Tau_wheel = [], []
    T_ism, U_ism, U_mpc_log = [], [], []
    fall = False
    fall_time = np.nan
    fall_idx = None
    fall_sat = fall_mpc_fails = 0
    u1_last = np.zeros(10)
    x_mpc_predicted_next = None
    x_mpc_prev_pred = None
    prev_int_err = np.zeros((2, 3))
    for i in range(num_steps):
        current_time = i * Config.dt_control
        x_meas = measure(robot_id, current_time)

        r = _ramp(tp, current_time)
        gl_active = ({j: 1.0 - r * (1.0 - g)
                      for j, g in tp.gain_loss.items()}
                     if r > 0.0 else {})
        if ground_id is not None and 0.0 < r <= 1.0:
            mu_now = MU_NOMINAL + r * (tp.mu_ground - MU_NOMINAL)
            p.changeDynamics(ground_id, -1, lateralFriction=mu_now)

        x_ref_list = generate_reference_trajectory(
            current_time, Config.dt_control, Config.N, robot_id,
            lean_angle=lean_filt)


        if tp.terrain_seed >= 0:
            floor_height = get_floor_height_below_com(robot_id)
            if floor_height > 0.1:
                for j in range(len(x_ref_list)):
                    x_ref_list[j][2] += floor_height

        wL, wR = get_end_effector_positions(robot_id)
        Rwb = Rotation.from_euler('xyz', x_meas[3:6]).as_matrix().T
        r_wheel_L_y = float((Rwb @ (wL - x_meas[0:3]))[1])
        r_wheel_R_y = float((Rwb @ (wR - x_meas[0:3]))[1])

        t0 = _time.time()
        u_optimal, x_pred = solve_ocp(solver, x_meas, x_ref_list,
                                      r_wheel_L_y=r_wheel_L_y,
                                      r_wheel_R_y=r_wheel_R_y)
        solve_times.append(_time.time() - t0)
        if u_optimal is None:
            mpc_fail_count += 1
        u_mpc = u_optimal if u_optimal is not None else np.zeros(10)

        x_mpc_current = x_meas.copy()
        x_mpc_prev_pred = x_mpc_predicted_next
        x_mpc_predicted_next = (x_pred[1].copy()
                                if x_pred is not None and len(x_pred) > 1
                                else None)

        if u_optimal is not None:
            delta_x_filt = (0.3 * calculate_delta_x(u_mpc, x_meas, robot_id)
                            + 0.7 * delta_x_filt)

            lean_filt = calculate_lean_angle(robot_id, delta_x_filt,x_meas)
        for physics_step in range(Config.num_physics_steps):
            t_phys = current_time + physics_step * Config.dt_physics
            x_phys = measure(robot_id, t_phys)

            ism_active = cfg.use_ism and i >= 59

            if cfg.use_ism:
                if x_mpc_predicted_next is None or not ism_active:
                    F_ISM_hold = np.zeros(10)
                    sta_integral_term = np.zeros(10)
                    u1_last = np.zeros(10)
                elif (physics_step % cfg.ism_every_n
                      == Config.ISM_PHASE[cfg.ism_every_n]):
                    alpha = physics_step / Config.num_physics_steps

                    x_start = (
                        x_mpc_prev_pred
                        if x_mpc_prev_pred is not None
                        else x_mpc_current
                    )

                    x_exp = x_start + alpha * (x_mpc_predicted_next - x_start)
                    dxw = Config.state_weights * (x_phys - x_exp)
                    _, G_phys = compute_ISM_matrices(
                        x_phys[3], x_phys[4], x_phys[5], x_phys, robot_id)

                    s = G_phys @ dxw

                    s_sign = np.sign(s)
                    u1 = -Config.lambda_min * np.sqrt(np.abs(s)) * s_sign
                    u1_last = u1
                    dt_ism = Config.dt_physics * cfg.ism_every_n
                    sta_integral_term += dt_ism * (-Config.W_gains * s_sign)
                    F_ISM_hold = u1 + sta_integral_term

            elif cfg.use_dob:
                dob_active = t_phys >= cfg.dob_start
                if x_mpc_predicted_next is None or not dob_active:
                    F_DOB_hold = np.zeros(10)
                    d_hat = np.zeros(10)
                elif (physics_step % cfg.ism_every_n
                      == Config.ISM_PHASE[cfg.ism_every_n]):
                    alpha = physics_step / Config.num_physics_steps

                    x_start = (
                        x_mpc_prev_pred
                        if x_mpc_prev_pred is not None
                        else x_mpc_current
                    )

                    x_exp = x_start + alpha * (x_mpc_predicted_next - x_start)
                    dxw = Config.state_weights * (x_phys - x_exp)
                    _, G_phys = compute_ISM_matrices(
                        x_phys[3], x_phys[4], x_phys[5], x_phys, robot_id)

                    s = G_phys @ dxw
                    dt_dob = Config.dt_physics * cfg.ism_every_n
                    d_hat = np.clip(
                        d_hat + dt_dob * cfg.dob_ki_scale * Config.W_gains * s,
                        -cfg.dob_dhat_max, cfg.dob_dhat_max)
                    F_DOB_hold = -d_hat

            if cfg.use_ism:
                F_aux = F_ISM_hold.copy()
                u_tot = u_mpc + F_aux
                F_aux[[0, 1, 5, 6]] = (np.clip(u_tot[[0, 1, 5, 6]], -50, 50)
                                       - u_mpc[[0, 1, 5, 6]])
                F_aux[[2, 7]] = (np.clip(u_tot[[2, 7]], 1e-5, 200)
                                 - u_mpc[[2, 7]])
                F_aux[[3, 8]] = (np.clip(u_tot[[3, 8]], -10, 10)
                                 - u_mpc[[3, 8]])
                F_aux[[4, 9]] = (np.clip(u_tot[[4, 9]], -10, 10)
                                 - u_mpc[[4, 9]])
                if Config.K_AW > 0:
                    sta_integral_term += Config.K_AW * (F_aux - F_ISM_hold)
                    F_ISM_hold = u1_last + sta_integral_term
            elif cfg.use_dob:
                F_aux = F_DOB_hold.copy()
                u_tot = u_mpc + F_aux
                F_aux[[0, 1, 5, 6]] = (np.clip(u_tot[[0, 1, 5, 6]], -50, 50)
                                       - u_mpc[[0, 1, 5, 6]])
                F_aux[[2, 7]] = (np.clip(u_tot[[2, 7]], 1e-5, 200)
                                 - u_mpc[[2, 7]])
                F_aux[[3, 8]] = (np.clip(u_tot[[3, 8]], -10, 10)
                                 - u_mpc[[3, 8]])
                F_aux[[4, 9]] = (np.clip(u_tot[[4, 9]], -10, 10)
                                 - u_mpc[[4, 9]])
                if Config.K_AW > 0:
                    d_hat = np.clip(
                        d_hat - Config.K_AW * (F_aux - F_DOB_hold),
                        -cfg.dob_dhat_max, cfg.dob_dhat_max)
                    F_DOB_hold = -d_hat
            else:
                F_aux = np.zeros(10)

            u_total = u_mpc + F_aux
            T_ism.append(t_phys)
            U_ism.append(F_aux.copy())
            U_mpc_log.append(u_mpc.copy())

            if np.any(u_total != 0):

                leg = compute_leg_torques(
                    robot_id, u_total, x_phys, delta_x_filt,
                    prev_target_angles, prev_desired_velocities,
                    prev_desired_accs, prev_int_err,
                    Config.dt_physics, i, physics_step,
                    wbc_full=cfg.wbc_full, wbc_moments=cfg.wbc_moments,
                    wbc_gravity=cfg.wbc_gravity)
                prev_target_angles = leg['prev_target_angles']
                prev_desired_velocities = leg['prev_desired_velocities']
                prev_desired_accs = leg['prev_desired_accs']
                prev_int_err = leg['prev_int_err']

                apply_torques_with_loss(robot_id, {
                    Config.ABAD_L: leg['total'][0],
                    Config.HIP_L: leg['total'][1],
                    Config.KNEE_L: leg['total'][2],
                    Config.ABAD_R: leg['total'][3],
                    Config.HIP_R: leg['total'][4],
                    Config.KNEE_R: leg['total'][5],
                }, gl_active, tp.tau_max, sat_counter)

                tau_wL, tau_wR, theta_prev = compute_wheel_torques(
                    x_phys, robot_id, u_total[0], u_total[5],
                    u_total[2], u_total[7],
                    theta_prev, i, Config.dt_physics, delta_x_filt)
                tau_wL_cmd, tau_wR_cmd = tau_wL, tau_wR
                apply_torques_with_loss(
                    robot_id,
                    {Config.WHEEL_L: tau_wL, Config.WHEEL_R: tau_wR},
                    gl_active, tp.tau_max, sat_counter)

                if physics_step == 0:
                    F_des.append([leg['forces_world'][0].copy(),
                                  leg['forces_world'][1].copy()])

            f_imp = impulse_at(tp, t_phys)
            if f_imp is not None:
                p.applyExternalForce(robot_id, -1, f_imp, push_point,
                                     p.LINK_FRAME)

            if gui:
                follow_robot_camera(robot_id)
            p.stepSimulation()

        x_true = get_robot_state(robot_id)
        if tp.scenario == "degradation":
            F_gr.append(measure_ground_wrench(robot_id, ground_id))
            Tau_wheel.append([tau_wL_cmd * gl_active.get(Config.WHEEL_L, 1.0),
                              tau_wR_cmd * gl_active.get(Config.WHEEL_R, 1.0)])
        T.append(current_time)
        X_true.append(x_true)
        X_ref.append(x_ref_list[0])
        if len(F_des) < len(T):
            F_des.append([np.zeros(3), np.zeros(3)])

        if not fall and current_time >= tp.fall_check_start:

            local_ground = (get_floor_height_below_com(robot_id)
                            if tp.terrain_seed >= 0 else 0.0)
            if (abs(x_true[3]) > tp.fall_roll
                    or abs(x_true[4]) > tp.fall_pitch
                    or (x_true[2] - local_ground) < tp.fall_pz):
                fall = True
                fall_time = current_time
                fall_idx = len(T)
                fall_sat = sat_counter[0]
                fall_mpc_fails = mpc_fail_count
                if tp.post_fall_run <= 0:
                    break
        if fall and current_time >= fall_time + tp.post_fall_run:
            break

    extended = fall and tp.post_fall_run > 0 and fall_idx is not None
    n_m = fall_idx if extended else len(T)
    row = _compute_metrics(tp, cfg, np.array(T[:n_m]), np.array(X_true[:n_m]),
                           np.array(X_ref[:n_m]), F_des[:n_m],
                           fall_sat if extended else sat_counter[0],
                           solve_times[:n_m], fall, fall_time)
    row["mpc_fail_count"] = fall_mpc_fails if extended else mpc_fail_count
    row["imp_point_x"] = push_point[0]
    row["imp_point_y"] = push_point[1]
    row["imp_point_z"] = push_point[2]

    if tp.scenario == "degradation":


        Fg, Tw = np.array(F_gr), np.array(Tau_wheel)
        Tarr = np.array(T)
        wss = Tarr >= tp.ss_start if len(Tarr) else np.array([], dtype=bool)
        if len(Fg) and wss.any():
            fx = float(Fg[wss, 0].mean())
            fz = float(Fg[wss, 2].mean())
            tau_sum = float(Tw[wss].sum(axis=1).mean())
            row["ss_f_resist"] = fx
            row["ss_fz_ground"] = fz
            row["ss_theta_eq"] = float(np.arctan2(fx, max(fz, 1e-6)))
            row["ss_tau_wheel"] = tau_sum

            row["ss_theta_eq_tau"] = float(np.arctan2(
                tau_sum / Config.wheel_radius, Config.m * 9.81))

    if keep_trace and len(T) > 0:
        Ta = np.array(T)
        Xt = np.array(X_true)
        Xr = np.array(X_ref)
        Ea = Xt - Xr
        yaw_e = np.arctan2(np.sin(Ea[:, 5]), np.cos(Ea[:, 5]))
        Ea[:, 5] = yaw_e
        row["trace"] = {"T": Ta,
                        "X": Xt,
                        "Xr": Xr,
                        "E": Ea,
                        "roll_err": Ea[:, 3].copy(),
                        "pitch_err": Ea[:, 4].copy(),
                        "yaw_err": yaw_e,
                        "vx": Xt[:, 6].copy(),
                        "vx_ref": Xr[:, 6].copy(),
                        "F_gr": np.array(F_gr),
                        "Tau_wheel": np.array(Tau_wheel),
                        "T_ism": np.array(T_ism),
                        "U_ism": np.array(U_ism),
                        "U_mpc": np.array(U_mpc_log)}
    return row


def _cone_violations(tp, T, F_des) -> int:
    viol = 0
    for t, (FL, FR) in zip(T, F_des):
        if t < tp.fall_check_start:
            continue
        for F in (FL, FR):
            if not np.any(F):
                continue
            fz = max(F[2], 1e-6)
            if np.hypot(F[0], F[1]) > 0.6 * fz or F[2] < 0:
                viol += 1
    return viol


def _compute_metrics(tp, cfg, T, X, Xr, F_des, sat_count, solve_times,
                     fall, fall_time) -> dict:
    _ab = (np.zeros(3) if tp.att_bias is None
           else np.asarray(tp.att_bias, dtype=float))
    row = {**{f"tp_{k}": v for k, v in asdict(tp).items()
              if k not in ("gain_loss", "att_bias")},
           "tp_att_bias_roll": float(_ab[0]),
           "tp_att_bias_pitch": float(_ab[1]),
           "tp_att_bias_yaw": float(_ab[2]),
           "tp_gain_min": (min(tp.gain_loss.values())
                           if tp.gain_loss else 1.0),
           "tp_n_degraded": sum(1 for g in tp.gain_loss.values() if g < 1.0),
           "controller": cfg.label, "use_ism": cfg.use_ism,
           "use_dob": cfg.use_dob,
           "ism_every_n": cfg.ism_every_n, "wbc_full": cfg.wbc_full,
           "yaw_frame": ("heading" if USE_WORLD_FRAME_YAW else "body"),
           "wbc_moments": cfg.wbc_moments,
           "wbc_gravity": cfg.wbc_gravity,
           "fall": fall,
           "fall_time": fall_time, "sat_count": sat_count,
           "mean_solve_ms": (1e3 * float(np.mean(solve_times))
                             if solve_times else np.nan)}

    if tp.scenario == "degradation":
        return _degradation_metrics(tp, row, T, X, Xr, F_des)

    row["peak_pitch_abs"] = np.nan

    if len(T) == 0:
        row.update(peak_roll_err=np.nan, peak_pitch_err=np.nan,
                   peak_yaw_err=np.nan, peak_vx_err=np.nan,
                   rms_vx_err=np.nan, rms_roll_err=np.nan,
                   rms_pitch_err=np.nan, rms_yaw_err=np.nan,
                   recovery_att=np.nan,
                   cone_violations=np.nan)
        return row

    E = X - Xr
    E[:, 5] = np.arctan2(np.sin(E[:, 5]), np.cos(E[:, 5]))

    post = T >= tp.imp_time
    if np.isfinite(tp.metric_start_x):
        steady = X[:, 0] >= tp.metric_start_x
    else:
        steady = T >= tp.fall_check_start
    if steady.any():
        E_st = E[steady]
    elif not np.isfinite(tp.metric_start_x):
        E_st = E
    else:
        E_st = None
    seg = E[post] if post.any() else E_st
    if seg is None or len(seg) == 0:
        row.update(peak_roll_err=np.nan, peak_pitch_err=np.nan,
                   peak_yaw_err=np.nan, peak_vx_err=np.nan)
    else:
        row["peak_roll_err"] = float(np.max(np.abs(seg[:, 3])))
        row["peak_pitch_err"] = float(np.max(np.abs(seg[:, 4])))
        row["peak_yaw_err"] = float(np.max(np.abs(seg[:, 5])))
        row["peak_vx_err"] = float(np.max(np.abs(seg[:, 6])))

    if post.any():
        _m = post
    elif steady.any():
        _m = steady
    elif not np.isfinite(tp.metric_start_x):
        _m = np.ones(len(T), dtype=bool)
    else:
        _m = None
    if _m is not None and _m.any():
        row["peak_pitch_abs"] = float(np.max(np.abs(X[_m, 4])))

    if E_st is None or len(E_st) == 0:
        row.update(rms_vx_err=np.nan, rms_roll_err=np.nan,
                   rms_pitch_err=np.nan, rms_yaw_err=np.nan)
    else:
        row["rms_vx_err"] = float(np.sqrt(np.mean(E_st[:, 6] ** 2)))
        row["rms_roll_err"] = float(np.sqrt(np.mean(E_st[:, 3] ** 2)))
        row["rms_pitch_err"] = float(np.sqrt(np.mean(E_st[:, 4] ** 2)))
        row["rms_yaw_err"] = float(np.sqrt(np.mean(E_st[:, 5] ** 2)))

    t_end = tp.imp_time + tp.imp_dur

    def _first_recovery(inside, hold_s):
        hold_n = max(1, int(hold_s / Config.dt_control))
        idx = np.where(T >= tp.imp_time)[0]
        if len(idx) == 0:
            return np.nan
        ins = inside[idx]
        out = np.where(~ins)[0]
        if len(out) == 0:
            return 0.0
        for k in range(int(out[0]), len(ins) - hold_n + 1):
            if ins[k:k + hold_n].all():
                return float(T[idx[k]] - t_end)
        return np.nan

    row["recovery_att"] = np.nan
    if not fall and post.any():
        row["recovery_att"] = _first_recovery(
            (np.abs(E[:, 3]) < tp.recovery_att_roll_band)
            & (np.abs(E[:, 4]) < tp.recovery_att_pitch_band),
            tp.recovery_att_hold)

    row["cone_violations"] = _cone_violations(tp, T, F_des)
    return row

_SS_KEYS = ("ss_roll_err", "ss_pitch_err", "ss_yaw_err", "ss_vx_err",
            "ss_roll_err_abs", "ss_pitch_err_abs", "ss_vx_err_abs",
            "peak_roll_err", "peak_pitch_err", "peak_yaw_err",
            "rms_vx_err", "ss_window_n",
            "ss_f_resist", "ss_fz_ground", "ss_theta_eq",
            "ss_tau_wheel", "ss_theta_eq_tau")


def _degradation_metrics(tp, row, T, X, Xr, F_des) -> dict:
    for k in _SS_KEYS:
        row[k] = np.nan
    row["cone_violations"] = np.nan

    if len(T) == 0:
        return row

    E = X - Xr
    E[:, 5] = np.arctan2(np.sin(E[:, 5]), np.cos(E[:, 5]))

    w = T >= tp.ss_start
    if w.any():
        seg = E[w]
        row["ss_roll_err"] = float(np.mean(seg[:, 3]))
        row["ss_pitch_err"] = float(np.mean(seg[:, 4]))
        row["ss_yaw_err"] = float(np.mean(seg[:, 5]))
        row["ss_vx_err"] = float(np.mean(seg[:, 6]))
        row["ss_roll_err_abs"] = float(np.mean(np.abs(seg[:, 3])))
        row["ss_pitch_err_abs"] = float(np.mean(np.abs(seg[:, 4])))
        row["ss_vx_err_abs"] = float(np.mean(np.abs(seg[:, 6])))
        row["peak_roll_err"] = float(np.max(np.abs(seg[:, 3])))
        row["peak_pitch_err"] = float(np.max(np.abs(seg[:, 4])))
        row["peak_yaw_err"] = float(np.max(np.abs(seg[:, 5])))
        row["rms_vx_err"] = float(np.sqrt(np.mean(seg[:, 6] ** 2)))
        row["ss_window_n"] = int(w.sum())
        row["peak_pitch_abs"] = float(np.max(np.abs(X[w, 4])))

    row["cone_violations"] = _cone_violations(tp, T, F_des)
    return row

def _nan_row(tp, cfg) -> dict:
    return _compute_metrics(tp, cfg, np.array([]), np.array([]),
                            np.array([]), [], 0, [], True, np.nan)


PLOT_T0 = 0.0
PLOT_YLIM = {"pitch": 15 , "vx": None, "roll": None, "yaw": None}
PLOT_FORCE_YLIM = {"fx": None, "fz": None, "aux": None, "ny": None}
COLORS = {
    "nmpc": "tab:red",
    "nmpc_stsmc": "tab:blue",
    "nmpc_stsmc_200": "tab:cyan",
    "nmpc_stsmc_80": "tab:green",
    "nmpc_stsmc_40": "tab:olive",
    "nmpc_dob": "tab:orange",
}


def plot_errors(rows, ss_start=SS_START, tag="", save=True):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    labels = ["pitch error [deg]", "forward velocity error [m/s]",
              "roll error [deg]", "yaw error [deg]"]
    cols = [(4, np.degrees(1.0)), (6, 1.0), (3, np.degrees(1.0)),
            (5, np.degrees(1.0))]

    t_max = 0.0
    for name, row in rows.items():
        tr = row.get("trace")
        if tr is None:
            continue
        T, E = tr["T"], tr["E"]
        t_max = max(t_max, float(T[-1]))
        c = COLORS.get(name)
        ls = "--" if row.get("fall") else "-"
        lbl = name + (" (FELL)" if row.get("fall") else "")
        for ax, (j, sc) in zip(axes, cols):
            ax.plot(T, E[:, j] * sc, ls, color=c, lw=1.3,
                    label=lbl if ax is axes[0] else None)

    for ax, lab in zip(axes, labels):
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        if t_max > ss_start:
            ax.axvspan(ss_start, t_max, color="grey", alpha=0.12)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(lab)
    axes[0].set_xlim(PLOT_T0, t_max)
    for ax, key in zip(axes, ("pitch", "vx", "roll", "yaw")):
        lim = PLOT_YLIM.get(key)
        if lim:
            ax.set_ylim(-lim, lim)

    axes[0].legend(fontsize=9, ncol=3)
    axes[0].set_title("actuator degradation: left wheel and both knees "
                      "derated, Coulomb friction on all joints "
                      f"(steady state shaded from {ss_start:.1f} s)")
    axes[-1].set_xlabel("t  [s]")
    fig.tight_layout()

    if save:
        out = f"deg_errors{tag}.png"
        fig.savefig(out, dpi=150)
        print(f"saved {os.path.abspath(out)}")
    return fig


def plot_forces(rows, ss_start=SS_START, tag="", save=True):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    t_max = 0.0
    for name, row in rows.items():
        tr = row.get("trace")
        if tr is None or len(tr.get("T_ism", [])) == 0:
            continue
        T, U, Um = tr["T_ism"], tr["U_ism"], tr["U_mpc"]
        t_max = max(t_max, float(T[-1]))
        c = COLORS.get(name)

        tot = Um + U
        axes[0].plot(T, tot[:, 0] + tot[:, 5], lw=1.0, color=c, label=name)
        axes[1].plot(T, tot[:, 2] + tot[:, 7], lw=1.0, color=c)

        aux = (np.linalg.norm(U[:, 0:3], axis=1)
               + np.linalg.norm(U[:, 5:8], axis=1))
        axes[2].plot(T, aux, lw=1.0, color=c)
        axes[3].plot(T, U[:, 3] + U[:, 8], lw=1.0, color=c)

        w = T >= ss_start
        if w.any():
            print(f"  {name}: steady-state mean |f_aux| = {aux[w].mean():.2f} N,"
                  f"  mean aux pitch moment = "
                  f"{(U[w, 3] + U[w, 8]).mean():+.3f} Nm")

    for ax, lab in zip(axes, ["total commanded fx  L+R [N]",
                              "total commanded fz  L+R [N]",
                              "|f_aux| L+R  [N]",
                              "aux pitch moment  ny L+R [Nm]"]):
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        if t_max > ss_start:
            ax.axvspan(ss_start, t_max, color="grey", alpha=0.12)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(lab)
    axes[0].set_xlim(PLOT_T0, t_max)
    for ax, key in zip(axes, ("fx", "fz", "aux", "ny")):
        lim = PLOT_FORCE_YLIM.get(key)
        if lim:
            ax.set_ylim(*lim)

    axes[0].legend(fontsize=9, ncol=3)
    axes[0].set_title("actuator degradation: commanded wrench and "
                      "fast-layer correction")
    axes[-1].set_xlabel("t  [s]")
    fig.tight_layout()

    if save:
        out = f"deg_forces{tag}.png"
        fig.savefig(out, dpi=150)
        print(f"saved {os.path.abspath(out)}")
    return fig


def plot_all(rows, ss_start=SS_START, tag="", save=True, show=True):
    import matplotlib
    import matplotlib.pyplot as plt
    plot_errors(rows, ss_start, tag, save)
    plot_forces(rows, ss_start, tag, save)
    backend = matplotlib.get_backend()
    if show and backend.lower() != "agg":
        plt.show()
    elif show:
        print(f"matplotlib backend is {backend} (non-interactive), so the "
              f"figures were written to disk but no window will open")
