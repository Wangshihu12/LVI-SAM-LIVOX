#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

Estimator estimator;

std::condition_variable                con;
double                                 current_time = -1;
queue<sensor_msgs::ImuConstPtr>        imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;

// global variable saving the lidar odometry
deque<nav_msgs::Odometry> odomQueue;
odometryRegister         *odomRegister;

std::mutex m_buf;
std::mutex m_state;
std::mutex m_estimator;
std::mutex m_odom;

double             latest_time;
Eigen::Vector3d    tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d    tmp_V;
Eigen::Vector3d    tmp_Ba;
Eigen::Vector3d    tmp_Bg;
Eigen::Vector3d    acc_0;
Eigen::Vector3d    gyr_0;
bool               init_feature = 0;
bool               init_imu     = 1;
double             last_imu_t   = 0;

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu    = 0;
        return;
    }
    double dt   = t - latest_time;
    latest_time = t;

    double          dx = imu_msg->linear_acceleration.x;
    double          dy = imu_msg->linear_acceleration.y;
    double          dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double          rx = imu_msg->angular_velocity.x;
    double          ry = imu_msg->angular_velocity.y;
    double          rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q                  = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P       = estimator.Ps[WINDOW_SIZE];
    tmp_Q       = estimator.Rs[WINDOW_SIZE];
    tmp_V       = estimator.Vs[WINDOW_SIZE];
    tmp_Ba      = estimator.Bas[WINDOW_SIZE];
    tmp_Bg      = estimator.Bgs[WINDOW_SIZE];
    acc_0       = estimator.acc_0;
    gyr_0       = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
        measurements;

    while (ros::ok())
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        if (!(imu_buf.back()->header.stamp.toSec() >
              feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp.toSec() <
              feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(estimator, tmp_P, tmp_Q, tmp_V, header, estimator.failureCount);
    }
}

void odom_callback(const nav_msgs::Odometry::ConstPtr &odom_msg)
{
    m_odom.lock();
    odomQueue.push_back(*odom_msg);
    m_odom.unlock();
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while (!feature_buf.empty())
            feature_buf.pop();
        while (!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t   = 0;
    }
    return;
}

/**
 * [功能描述]：VINS估计器的主处理循环，融合IMU和视觉特征数据进行状态估计
 * 
 * 主要流程：
 * 1. 获取同步的IMU和图像特征数据
 * 2. 执行IMU预积分
 * 3. 处理视觉特征点并进行VINS优化
 * 4. 发布估计结果和可视化信息
 */
void process()
{
    // ========== 主处理循环 ==========
    while (ros::ok())
    {
        // ##### 第1步：获取同步测量数据 #####
        // 数据结构：每个measurement包含一组IMU数据和一帧图像特征数据
        std::vector<
            std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
                                     measurements;
        
        // 加锁并等待新的测量数据到达
        std::unique_lock<std::mutex> lk(m_buf);
        // 条件变量等待：直到getMeasurements()返回非空数据
        // getMeasurements()负责从缓冲区中提取时间同步的IMU和图像特征数据
        con.wait(lk, [&] { return (measurements = getMeasurements()).size() != 0; });
        lk.unlock();

        // 加锁保护估计器状态，确保线程安全
        m_estimator.lock();
        
        // ========== 遍历处理每个测量数据包 ==========
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second; // 图像特征点消息

            // ########################################
            // ##### 第2步：IMU预积分处理 #####
            // ########################################
            
            // IMU数据变量：线性加速度和角速度
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            
            // 遍历当前图像帧对应的所有IMU数据
            for (auto &imu_msg : measurement.first)
            {
                double t     = imu_msg->header.stamp.toSec();           // IMU时间戳
                double img_t = img_msg->header.stamp.toSec() + estimator.td; // 图像时间戳+时间偏移

                // ===== 情况1：IMU时间戳小于等于图像时间戳 =====
                if (t <= img_t)
                {
                    // 初始化当前时间
                    if (current_time < 0)
                        current_time = t;
                    
                    // 计算时间间隔
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0); // 确保时间间隔非负
                    current_time = t;    // 更新当前时间
                    
                    // 提取IMU测量值
                    dx = imu_msg->linear_acceleration.x;  // x轴线性加速度
                    dy = imu_msg->linear_acceleration.y;  // y轴线性加速度
                    dz = imu_msg->linear_acceleration.z;  // z轴线性加速度
                    rx = imu_msg->angular_velocity.x;     // x轴角速度
                    ry = imu_msg->angular_velocity.y;     // y轴角速度
                    rz = imu_msg->angular_velocity.z;     // z轴角速度
                    
                    // 执行IMU预积分处理
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                }
                else
                {
                    // ===== 情况2：IMU时间戳大于图像时间戳（需要插值） =====
                    // 当IMU频率高于相机频率时，需要在图像时刻进行IMU数据插值
                    
                    double dt_1  = img_t - current_time; // 从当前时间到图像时间的间隔
                    double dt_2  = t - img_t;            // 从图像时间到IMU时间的间隔
                    current_time = img_t;                // 更新当前时间为图像时间
                    
                    // 时间间隔检查
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    
                    // 线性插值权重计算
                    double w1 = dt_2 / (dt_1 + dt_2); // 前一个IMU数据的权重
                    double w2 = dt_1 / (dt_1 + dt_2); // 当前IMU数据的权重
                    
                    // 对IMU数据进行线性插值
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    
                    // 使用插值后的IMU数据进行预积分
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                }
            }

            // ########################################
            // ##### 第3步：VINS视觉优化处理 #####
            // ########################################
            
            // 构建特征点数据结构：map<特征点ID, vector<相机ID和8维特征向量>>
            map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> image;
            
            // 解析图像特征点消息
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                // 解码特征点ID和相机ID
                int    v          = img_msg->channels[0].values[i] + 0.5; // 全局特征点ID编码
                int    feature_id = v / NUM_OF_CAM;                       // 真实特征点ID
                int    camera_id  = v % NUM_OF_CAM;                       // 相机ID
                
                // 提取特征点的各种信息
                double x          = img_msg->points[i].x;            // 归一化相机坐标x
                double y          = img_msg->points[i].y;            // 归一化相机坐标y
                double z          = img_msg->points[i].z;            // 归一化相机坐标z（应为1）
                double p_u        = img_msg->channels[1].values[i];  // 像素坐标u
                double p_v        = img_msg->channels[2].values[i];  // 像素坐标v
                double velocity_x = img_msg->channels[3].values[i];  // 光流速度x分量
                double velocity_y = img_msg->channels[4].values[i];  // 光流速度y分量
                double depth      = img_msg->channels[5].values[i];  // 激光雷达深度信息

                ROS_ASSERT(z == 1); // 确保归一化坐标z分量为1
                
                // 构建8维特征向量：[归一化坐标(3) + 像素坐标(2) + 光流速度(2) + 深度(1)]
                Eigen::Matrix<double, 8, 1> xyz_uv_velocity_depth;
                xyz_uv_velocity_depth << x, y, z, p_u, p_v, velocity_x, velocity_y, depth;
                
                // 将特征点数据按ID分组存储
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity_depth);
            }

            // ##### 获取激光里程计初始化信息 #####
            vector<float> initialization_info;
            m_odom.lock();
            // 从激光里程计中获取当前图像时刻的位姿信息，用于VINS系统初始化
            initialization_info =
                odomRegister->getOdometry(odomQueue, img_msg->header.stamp.toSec() + estimator.td);
            m_odom.unlock();

            // ##### 执行VINS主要处理 #####
            // 融合视觉特征点、IMU预积分结果和激光里程计信息，进行非线性优化
            estimator.processImage(image, initialization_info, img_msg->header);

            // ########################################
            // ##### 第4步：结果发布和可视化 #####
            // ########################################
            
            std_msgs::Header header = img_msg->header;
            
            // 发布各种估计结果和可视化信息
            pubOdometry(estimator, header);    // 发布里程计信息（位姿、速度等）
            pubKeyPoses(estimator, header);    // 发布关键帧位姿
            pubCameraPose(estimator, header);  // 发布相机位姿
            pubPointCloud(estimator, header);  // 发布三维点云地图
            pubTF(estimator, header);          // 发布TF变换关系
            pubKeyframe(estimator);            // 发布关键帧信息
        }
        
        m_estimator.unlock(); // 解锁估计器

        // ##### 第5步：状态更新 #####
        m_buf.lock();
        m_state.lock();
        
        // 如果估计器已经进入非线性优化状态，更新系统状态
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update(); // 更新全局状态变量
            
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins");
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Odometry Estimator Started.\033[0m");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);

    readParameters(n);
    estimator.setParameter();

    registerPub(n);

    odomRegister = new odometryRegister(n);

    ros::Subscriber sub_imu =
        n.subscribe(IMU_TOPIC, 5000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_odom = n.subscribe("odometry/imu", 5000, odom_callback);
    ros::Subscriber sub_image =
        n.subscribe(PROJECT_NAME + "/vins/feature/feature", 1, feature_callback);
    ros::Subscriber sub_restart =
        n.subscribe(PROJECT_NAME + "/vins/feature/restart", 1, restart_callback);
    if (!USE_LIDAR)
        sub_odom.shutdown();

    std::thread measurement_process{process};

    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    return 0;
}