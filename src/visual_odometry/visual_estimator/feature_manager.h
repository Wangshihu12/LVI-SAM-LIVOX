#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

/**
 * [类功能描述]：FeaturePerFrame - 存储特征点在单帧中的观测信息
 * 
 * 该类封装了一个特征点在某一帧图像中的所有相关信息，包括：
 * - 几何信息：归一化坐标、像素坐标、深度
 * - 运动信息：光流速度
 * - 时间信息：时间偏移
 * - 优化相关：雅可比矩阵、残差等
 */
class FeaturePerFrame
{
  public:
    /**
     * [构造函数]：从8维特征向量构建单帧观测对象
     * 
     * @param _point：8维特征向量 [归一化x, 归一化y, z, 像素u, 像素v, 速度vx, 速度vy, 深度]
     * @param td：相机-IMU时间偏移（秒）
     * 
     * 8维向量解析：
     * _point(0,1,2): 归一化相机坐标 (x/z, y/z, 1)
     * _point(3,4): 像素坐标 (u, v)  
     * _point(5,6): 光流速度 (vx, vy)
     * _point(7): 激光雷达深度信息
     */
    FeaturePerFrame(const Eigen::Matrix<double, 8, 1> &_point, double td)
    {
        // ========== 提取归一化相机坐标 ==========
        // 归一化坐标：消除了相机内参影响的标准化坐标
        // 用于几何计算，如三角化、位姿估计等
        point.x() = _point(0);  // 归一化x坐标
        point.y() = _point(1);  // 归一化y坐标
        point.z() = _point(2);  // 归一化z坐标（通常为1）
        
        // ========== 提取像素坐标 ==========
        // 原始图像中的像素位置，用于重投影误差计算
        uv.x() = _point(3);     // 像素u坐标（列）
        uv.y() = _point(4);     // 像素v坐标（行）
        
        // ========== 提取光流速度 ==========
        // 特征点在图像平面上的运动速度，单位：像素/秒
        // 用于运动预测和时间偏移补偿
        velocity.x() = _point(5);  // x方向速度分量
        velocity.y() = _point(6);  // y方向速度分量
        
        // ========== 提取深度信息 ==========
        // 来自激光雷达的深度测量，-1表示无深度信息
        depth = _point(7);
        
        // ========== 设置时间偏移 ==========
        // 相机与IMU之间的时间同步偏差
        cur_td = td;
    }

    // ########################################
    // ##### 时间同步相关 #####
    // ########################################
    
    // 当前观测的时间偏移（相机-IMU时间差）
    // 用于多传感器时间同步，补偿硬件延迟
    double cur_td;

    // ########################################
    // ##### 几何信息 #####
    // ########################################
    
    // 归一化相机坐标 (x/z, y/z, 1)
    // 消除了相机内参的影响，用于几何算法
    // 坐标系：相机坐标系，z轴朝前，x轴朝右，y轴朝下
    Vector3d point;

    // 像素坐标 (u, v)
    // 特征点在图像中的原始像素位置
    // u: 列坐标（水平方向），v: 行坐标（垂直方向）
    Vector2d uv;

    // 光流速度 (vx, vy)
    // 特征点在图像平面上的运动速度，单位：像素/秒
    // 计算方法：(当前位置 - 前一位置) / 时间间隔
    Vector2d velocity;

    // ########################################
    // ##### 深度和状态信息 #####
    // ########################################
    
    // 深度坐标（可能用于某些特殊情况）
    // 注：通常使用depth字段存储深度信息
    double z;

    // 特征点使用标志
    // true: 该观测参与优化计算
    // false: 该观测被排除（可能因为质量差或异常）
    bool is_used;

    // 视差值
    // 该特征点相对于参考帧的视差，用于评估三角化质量
    // 视差越大，三角化精度越高
    double parallax;

    // ########################################
    // ##### 优化相关矩阵 #####
    // ########################################
    
    // 雅可比矩阵A
    // 存储该观测对应的线性化雅可比矩阵
    // 用于最小二乘优化中的法方程构建
    MatrixXd A;

    // 残差向量b  
    // 存储该观测对应的残差向量
    // 用于最小二乘优化中的法方程构建
    VectorXd b;

    // 深度梯度
    // 深度参数的梯度信息，用于深度优化
    // 帮助判断深度参数的可观测性和优化方向
    double dep_gradient;

    // ########################################
    // ##### 激光雷达深度信息 #####
    // ########################################
    
    // 激光雷达提供的深度测量值
    // 初始值：-1（表示无深度信息）
    // 正值：来自激光雷达的有效深度测量
    // 用途：
    // - 提供深度先验约束
    // - 加速特征点三角化
    // - 提高深度估计精度
    double depth;
};

/**
 * [类功能描述]：FeaturePerId - 管理单个特征点在整个生命周期中的完整信息
 * 
 * 该类负责跟踪一个特征点从首次观测到最后观测的全过程，包括：
 * - 基本属性：ID、时间范围、观测序列
 * - 几何信息：深度估计、三角化状态
 * - 优化状态：是否为外点、边缘化标志
 * - 激光融合：激光雷达深度信息
 */
class FeaturePerId
{
  public:
    // ########################################
    // ##### 基本标识信息 #####
    // ########################################
    
    // 特征点的全局唯一标识符（常量，创建后不可修改）
    // 用于跨帧关联和数据管理，确保同一特征点在不同帧中的一致性
    const int feature_id;

    // 特征点首次被观测到的帧索引
    // 标记该特征点生命周期的起始时刻
    // 用于计算特征点的存活时间和观测历史
    int start_frame;

    // 特征点在各帧中的观测序列
    // 每个FeaturePerFrame对象包含该特征点在一帧中的完整观测信息
    // 索引对应：feature_per_frame[i] 对应帧 (start_frame + i)
    vector<FeaturePerFrame> feature_per_frame;

    // ########################################
    // ##### 统计和状态信息 #####
    // ########################################

    // 特征点被观测的总帧数
    // 等于feature_per_frame.size()，用于快速获取观测次数
    // 观测次数越多，特征点越稳定，优化约束越可靠
    int used_num;

    // 外点标志
    // true: 该特征点被认为是外点（异常观测），不参与优化
    // false: 正常的内点，参与Bundle Adjustment优化
    // 外点通常由RANSAC或其他鲁棒估计算法检测
    bool is_outlier;

    // 边缘化标志  
    // true: 该特征点即将被边缘化（从优化窗口中移除）
    // false: 特征点仍在当前优化窗口中
    // 用于滑动窗口管理和内存释放
    bool is_margin;

    // ########################################
    // ##### 深度估计相关 #####
    // ########################################

    // 特征点的深度估计值（在首次观测帧的相机坐标系下）
    // -1.0: 深度未知或三角化失败
    // 正值: 有效的深度估计，单位为米
    // 该值会在优化过程中不断更新
    double estimated_depth;

    // 激光雷达深度标志
    // true: 该特征点有来自激光雷达的深度测量
    // false: 纯视觉特征点，深度需要通过三角化估计
    // 激光雷达提供的深度通常更准确，可作为强约束
    bool lidar_depth_flag;

    // 三角化求解状态标志
    // 0: 尚未尝试求解深度
    // 1: 三角化成功，深度估计可靠
    // 2: 三角化失败，深度估计不可用
    // 用于跟踪优化过程和调试
    int solve_flag;

    // ########################################
    // ##### 调试和验证信息 #####
    // ########################################

    // 真值3D位置（仅用于仿真和算法验证）
    // 在实际应用中通常不可用，主要用于：
    // - 算法精度评估
    // - 调试和性能分析
    // - 仿真环境下的对比验证
    Vector3d gt_p;

    // ########################################
    // ##### 构造函数 #####
    // ########################################

    /**
     * [构造函数]：创建新的特征点对象
     * 
     * @param _feature_id：特征点的全局唯一ID
     * @param _start_frame：特征点首次观测的帧索引
     * @param _measured_depth：初始深度测量值（来自激光雷达或其他传感器）
     * 
     * 初始化策略：
     * - 有深度测量时：设置为激光雷达约束特征点
     * - 无深度测量时：设置为纯视觉特征点，等待三角化
     */
    FeaturePerId(int _feature_id, int _start_frame, double _measured_depth)
        : feature_id(_feature_id),      // 设置特征点ID（常量）
          start_frame(_start_frame),    // 设置起始帧索引
          used_num(0),                  // 初始观测次数为0
          estimated_depth(-1.0),        // 初始深度未知
          lidar_depth_flag(false),      // 初始无激光深度标志
          solve_flag(0)                 // 初始未求解状态
    {
        // ===== 根据初始深度测量值设置深度信息 =====
        if (_measured_depth > 0)
        {
            // 有有效的深度测量值（来自激光雷达）
            estimated_depth = _measured_depth;  // 设置初始深度估计
            lidar_depth_flag = true;            // 标记为激光雷达约束特征点
            
            // 优势：
            // - 提供强深度约束，提高优化稳定性
            // - 加速收敛，减少局部最优问题  
            // - 提供尺度信息，解决视觉SLAM尺度模糊性
        }
        else
        {
            // 无有效深度测量值（纯视觉特征点）
            estimated_depth = -1;              // 深度未知，等待三角化
            lidar_depth_flag = false;          // 标记为纯视觉特征点
            
            // 后续处理：
            // - 当观测数≥2时，通过三角化估计深度
            // - 深度质量依赖于视差和观测精度
            // - 可能需要多次优化迭代来收敛
        }
    }

    // ########################################
    // ##### 成员函数声明 #####
    // ########################################

    /**
     * [功能描述]：获取特征点最后一次被观测的帧索引
     * @return：结束帧索引
     * 
     * 计算公式：endFrame = start_frame + feature_per_frame.size() - 1
     * 
     * 应用场景：
     * - 检查特征点是否在某个时间范围内被观测
     * - 计算特征点的生存时间
     * - 滑动窗口管理和边缘化决策
     */
    int endFrame();
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif