#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"
#include "../parameters.h"

using namespace Eigen;
using namespace std;

/**
 * [类功能描述]：ImageFrame - 存储单帧图像的完整状态信息
 * 
 * 该类封装了VINS系统中一帧图像的所有相关数据，包括：
 * - 视觉信息：特征点观测数据
 * - 时间信息：图像时间戳
 * - 运动信息：IMU预积分、位姿、速度、偏差
 * - 激光信息：激光雷达里程计提供的初始化数据
 * - 标志信息：关键帧标记、重定位ID等
 */
class ImageFrame
{
public:
    /**
     * [默认构造函数]：创建空的图像帧对象
     */
    ImageFrame(){};
    
    /**
     * [带参数构造函数]：从传感器数据构建完整的图像帧对象
     * 
     * @param _points：视觉特征点数据
     *        格式：map<特征点ID, vector<相机ID和8维特征向量>>
     *        8维向量：[归一化x, 归一化y, z, 像素u, 像素v, 速度vx, 速度vy, 深度]
     * 
     * @param _lidar_initialization_info：激光雷达里程计初始化信息（18维向量）
     *        索引0: 重置ID（用于检测激光里程计重定位）
     *        索引1-3: 位置向量 T = [x, y, z]
     *        索引4-7: 四元数 Q = [qx, qy, qz, qw]（注意：qw在最后）
     *        索引8-10: 速度向量 V = [vx, vy, vz]
     *        索引11-13: 加速度计偏差 Ba = [bax, bay, baz]
     *        索引14-16: 陀螺仪偏差 Bg = [bgx, bgy, bgz]
     *        索引17: 重力加速度大小
     * 
     * @param _t：图像时间戳（秒）
     */
    ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &_points,
               const vector<float> &_lidar_initialization_info, double _t)
        : t{_t}, is_key_frame{false}, reset_id{-1}, gravity{9.805}
    {
        // ===== 第1步：复制视觉特征点数据 =====
        points = _points;

        // ===== 第2步：解析激光雷达重置信息 =====
        // 重置ID用于检测激光雷达里程计是否发生重定位
        // 如果reset_id发生变化，说明激光里程计系统重启或重定位
        reset_id = (int)round(_lidar_initialization_info[0]);
        
        // ===== 第3步：解析位置信息 =====
        // 从激光雷达里程计获取的位置向量（世界坐标系）
        T.x() = _lidar_initialization_info[1];  // x坐标
        T.y() = _lidar_initialization_info[2];  // y坐标  
        T.z() = _lidar_initialization_info[3];  // z坐标
        
        // ===== 第4步：解析旋转信息 =====
        // 从四元数构建旋转矩阵
        // 注意：激光里程计输出的四元数格式为[qx, qy, qz, qw]
        Eigen::Quaterniond Q =
            Eigen::Quaterniond(_lidar_initialization_info[7],  // qw（标量部分）
                               _lidar_initialization_info[4],  // qx（向量部分x）
                               _lidar_initialization_info[5],  // qy（向量部分y）
                               _lidar_initialization_info[6]); // qz（向量部分z）
        // 归一化四元数并转换为旋转矩阵
        R = Q.normalized().toRotationMatrix();
        
        // ===== 第5步：解析速度信息 =====
        // 从激光雷达里程计获取的速度向量（世界坐标系）
        V.x() = _lidar_initialization_info[8];   // x方向速度
        V.y() = _lidar_initialization_info[9];   // y方向速度
        V.z() = _lidar_initialization_info[10];  // z方向速度
        
        // ===== 第6步：解析IMU偏差信息 =====
        // 加速度计偏差向量（设备坐标系）
        Ba.x() = _lidar_initialization_info[11]; // x轴加速度偏差
        Ba.y() = _lidar_initialization_info[12]; // y轴加速度偏差  
        Ba.z() = _lidar_initialization_info[13]; // z轴加速度偏差
        
        // 陀螺仪偏差向量（设备坐标系）
        Bg.x() = _lidar_initialization_info[14]; // x轴角速度偏差
        Bg.y() = _lidar_initialization_info[15]; // y轴角速度偏差
        Bg.z() = _lidar_initialization_info[16]; // z轴角速度偏差
        
        // ===== 第7步：解析重力信息 =====
        // 当前环境的重力加速度大小（m/s²）
        gravity = _lidar_initialization_info[17];
    };

    // ########################################
    // ##### 成员变量定义 #####
    // ########################################

    // ===== 视觉数据 =====
    // 该帧图像中检测到的所有特征点观测数据
    // 外层map的key: 特征点的全局唯一ID
    // 内层vector: 该特征点在不同相机中的观测（支持多相机系统）
    // pair结构: <相机ID, 8维特征向量>
    map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> points;
    
    // ===== 时间信息 =====
    // 图像帧的时间戳（秒，通常是ROS时间）
    double t;

    // ===== IMU预积分 =====
    // 从上一关键帧到当前帧的IMU预积分对象指针
    // 包含位置、速度、旋转的积分结果和协方差信息
    IntegrationBase *pre_integration;
    
    // ===== 关键帧标志 =====
    // 标记该帧是否为关键帧
    // true: 关键帧，参与优化和建图
    // false: 普通帧，仅用于特征跟踪
    bool is_key_frame;

    // ########################################
    // ##### 激光雷达里程计信息 #####
    // ########################################

    // ===== 重定位检测 =====
    // 激光雷达里程计的重置ID，用于检测系统重启或重定位
    // 当reset_id发生变化时，需要重新初始化VINS系统
    int reset_id;
    
    // ===== 位姿信息 =====
    // 位置向量：激光雷达里程计估计的位置（世界坐标系）
    Vector3d T;
    // 旋转矩阵：激光雷达里程计估计的姿态（世界坐标系到本体坐标系）
    Matrix3d R;
    
    // ===== 运动信息 =====
    // 速度向量：激光雷达里程计估计的速度（世界坐标系）
    Vector3d V;
    
    // ===== IMU偏差信息 =====
    // 加速度计偏差：三轴加速度计的零偏（本体坐标系）
    Vector3d Ba;
    // 陀螺仪偏差：三轴陀螺仪的零偏（本体坐标系）
    Vector3d Bg;
    
    // ===== 环境参数 =====
    // 当前环境的重力加速度大小（m/s²）
    // 用于IMU数据处理和姿态估计
    double gravity;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g,
                        VectorXd &x);

class odometryRegister
{
public:
    ros::NodeHandle    n;
    tf::Quaternion     q_lidar_to_imu;
    Eigen::Quaterniond q_lidar_to_imu_eigen;

    ros::Publisher pub_latest_odometry;

    odometryRegister(ros::NodeHandle n_in) : n(n_in)
    {
        // TODO: 修改为读配置文件决定外参
        // 怎么两个参数还是不一样的？
        //  1  0  0
        //  0 -1  0
        //  0  0 -1
        // q_lidar_to_cam = tf::Quaternion(0, 1, 0, 0); // rotate orientation // mark: camera - lidar
        // -1  0  0
        //  0 -1  0
        //  0  0  1
        // q_lidar_to_cam_eigen = Eigen::Quaterniond(0, 0, 0, 1); // rotate position by pi, (w, x, y, z) // mark: camera - lidar
        // // pub_latest_odometry = n.advertise<nav_msgs::Odometry>("odometry/test", 1000);
        // modified:
        q_lidar_to_imu       = tf::createQuaternionFromRPY(L_I_RX, L_I_RY, L_I_RZ);
        q_lidar_to_imu_eigen = Eigen::Quaterniond(
            q_lidar_to_imu.w(), q_lidar_to_imu.x(), q_lidar_to_imu.y(),
            q_lidar_to_imu.z());  // rotate position by pi, (w, x, y, z) // mark: camera - lidar
    }

    // convert odometry from ROS Lidar frame to VINS camera frame
    // DONE: ??? odomQueue 对应/odometry/imu 看起来是imu frame: 确实是lidar frame，可能是想表达imu频率的odometry
    vector<float> getOdometry(deque<nav_msgs::Odometry> &odomQueue, double img_time)
    {
        vector<float> odometry_channel;
        odometry_channel.resize(18, -1);  // reset id(1), P(3), Q(4), V(3), Ba(3), Bg(3), gravity(1)

        nav_msgs::Odometry odomCur;

        // pop old odometry msg
        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < img_time - 0.05)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
        {
            return odometry_channel;
        }

        // find the odometry time that is the closest to image time
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            odomCur = odomQueue[i];

            if (odomCur.header.stamp.toSec() < img_time - 0.002)  // 500Hz imu
                continue;
            else
                break;
        }

        // time stamp difference still too large
        if (abs(odomCur.header.stamp.toSec() - img_time) > 0.05)
        {
            return odometry_channel;
        }

        // convert odometry position from lidar ROS frame to VINS frame
        Eigen::Vector3d    p_eigen(odomCur.pose.pose.position.x, odomCur.pose.pose.position.y,
                                   odomCur.pose.pose.position.z);
        Eigen::Vector3d    v_eigen(odomCur.twist.twist.linear.x, odomCur.twist.twist.linear.y,
                                   odomCur.twist.twist.linear.z);
        Eigen::Quaterniond q_eigen(odomCur.pose.pose.orientation.w, odomCur.pose.pose.orientation.x,
                                   odomCur.pose.pose.orientation.y,
                                   odomCur.pose.pose.orientation.z);
        Eigen::Vector3d    p_eigen_new = q_lidar_to_imu_eigen * p_eigen;
        Eigen::Vector3d    v_eigen_new = q_lidar_to_imu_eigen * v_eigen;
        Eigen::Quaterniond q_eigen_new = q_eigen * q_lidar_to_imu_eigen.inverse();

        odomCur.pose.pose.position.x = p_eigen_new.x();
        odomCur.pose.pose.position.y = p_eigen_new.y();
        odomCur.pose.pose.position.z = p_eigen_new.z();

        odomCur.twist.twist.linear.x = v_eigen_new.x();
        odomCur.twist.twist.linear.y = v_eigen_new.y();
        odomCur.twist.twist.linear.z = v_eigen_new.z();

        odomCur.pose.pose.orientation.w = q_eigen_new.w();
        odomCur.pose.pose.orientation.x = q_eigen_new.x();
        odomCur.pose.pose.orientation.y = q_eigen_new.y();
        odomCur.pose.pose.orientation.z = q_eigen_new.z();

        // modified:
        odometry_channel[0]  = odomCur.pose.covariance[0];
        odometry_channel[1]  = odomCur.pose.pose.position.x;
        odometry_channel[2]  = odomCur.pose.pose.position.y;
        odometry_channel[3]  = odomCur.pose.pose.position.z;
        odometry_channel[4]  = odomCur.pose.pose.orientation.x;
        odometry_channel[5]  = odomCur.pose.pose.orientation.y;
        odometry_channel[6]  = odomCur.pose.pose.orientation.z;
        odometry_channel[7]  = odomCur.pose.pose.orientation.w;
        odometry_channel[8]  = odomCur.twist.twist.linear.x;
        odometry_channel[9]  = odomCur.twist.twist.linear.y;
        odometry_channel[10] = odomCur.twist.twist.linear.z;
        odometry_channel[11] = odomCur.pose.covariance[1];
        odometry_channel[12] = odomCur.pose.covariance[2];
        odometry_channel[13] = odomCur.pose.covariance[3];
        odometry_channel[14] = odomCur.pose.covariance[4];
        odometry_channel[15] = odomCur.pose.covariance[5];
        odometry_channel[16] = odomCur.pose.covariance[6];
        odometry_channel[17] = odomCur.pose.covariance[7];

        return odometry_channel;
    }
};
