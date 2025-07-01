#include "parameters.h"
#include "keyframe.h"
#include "loop_detection.h"

#define SKIP_FIRST_CNT 10

// ########################################
// ##### 数据缓冲区：存储待处理的传感器数据 #####
// ########################################

// 图像数据缓冲队列
// 存储来自相机的原始图像消息，用于回环检测的特征提取和匹配
queue<sensor_msgs::ImageConstPtr>      image_buf;

// 特征点云数据缓冲队列  
// 存储来自特征跟踪器的特征点云消息，包含特征点的3D坐标和描述信息
queue<sensor_msgs::PointCloudConstPtr> point_buf;

// 位姿数据缓冲队列
// 存储来自VINS估计器的里程计信息，包含位置、姿态和协方差信息
queue<nav_msgs::Odometry::ConstPtr>    pose_buf;

// ########################################
// ##### 线程同步锁：保证多线程数据安全 #####
// ########################################

// 缓冲区访问保护锁
// 保护上述三个数据缓冲队列的并发访问，确保数据读写的线程安全
std::mutex m_buf;

// 处理过程保护锁  
// 保护回环检测的核心处理过程，防止多个线程同时执行回环检测算法
std::mutex m_process;

// ########################################
// ##### 核心算法对象：回环检测器 #####
// ########################################

// 回环检测器对象
// 封装了完整的回环检测算法，包括：
// - 关键帧管理和选择
// - 词袋模型匹配
// - 几何验证
// - 回环约束生成
LoopDetector loopDetector;

// ########################################
// ##### 回环检测控制参数 #####
// ########################################

// 时间间隔跳跃阈值（秒）
// 控制回环检测的时间频率，避免过于频繁的检测
// 只有当前帧与上一个关键帧的时间间隔超过此值时，才考虑进行回环检测
double SKIP_TIME = 0;

// 距离间隔跳跃阈值（米）
// 控制回环检测的空间频率，避免相邻位置的重复检测
// 只有当前位置与上一个关键帧的距离超过此值时，才考虑进行回环检测
double SKIP_DIST = 0;

// ########################################
// ##### 相机参数：用于特征处理和几何验证 #####
// ########################################

// 相机模型对象
// 包含相机的内参、畸变参数等信息，用于：
// - 图像特征点的畸变校正
// - 3D-2D投影变换
// - 几何验证中的重投影计算
camodocal::CameraPtr m_camera;

// 相机-IMU外参：平移向量
// 表示相机坐标系相对于IMU坐标系的平移偏移
// 用于坐标系转换和位姿对齐
Eigen::Vector3d tic;

// 相机-IMU外参：旋转矩阵
// 表示相机坐标系相对于IMU坐标系的旋转变换
// 用于坐标系转换和位姿对齐
Eigen::Matrix3d qic;

// ########################################
// ##### ROS配置参数 #####
// ########################################

// 项目名称
// 用于构建ROS话题命名空间，确保多个实例之间的话题隔离
std::string PROJECT_NAME;

// 图像话题名称
// 指定订阅的相机图像话题，用于接收原始图像数据
std::string IMAGE_TOPIC;

// ########################################
// ##### 调试和功能开关 #####
// ########################################

// 调试图像开关
// 控制是否发布可视化的调试图像：
// - 0: 不发布调试图像
// - 1: 发布回环匹配的可视化结果
int DEBUG_IMAGE;

// 回环闭合功能开关
// 控制是否启用回环检测功能：
// - 0: 禁用回环检测，节省计算资源
// - 1: 启用回环检测，提高建图精度
int LOOP_CLOSURE;

// 匹配图像缩放比例
// 控制用于特征匹配的图像分辨率：
// - 1.0: 使用原始分辨率
// - 0.5: 缩放到一半分辨率，提高处理速度
// - 较小的值可以提高计算效率，但可能降低匹配精度
double MATCH_IMAGE_SCALE;

// ########################################
// ##### ROS发布器：输出回环检测结果 #####
// ########################################

// 匹配图像发布器
// 发布回环检测的可视化结果图像，用于调试和监控：
// - 显示当前帧与历史帧的特征点匹配情况
// - 标注匹配的特征点对和匹配质量
ros::Publisher pub_match_img;

// 匹配消息发布器
// 发布回环检测的数量化结果消息，包含：
// - 回环帧的ID和时间戳
// - 相对位姿变换矩阵  
// - 匹配质量和置信度
ros::Publisher pub_match_msg;

// 关键位姿发布器
// 发布用于回环检测的关键帧位姿信息：
// - 关键帧的位置和姿态
// - 关键帧的选择时间戳
// - 用于后端优化的约束构建
ros::Publisher pub_key_pose;

// ########################################
// ##### 特征提取器：用于描述符计算 #####
// ########################################

// BRIEF特征描述符提取器
// 用于计算图像特征点的二进制描述符：
// - 快速的二进制特征描述符算法
// - 适合实时回环检测的计算要求
// - 支持快速的汉明距离匹配
BriefExtractor briefExtractor;

void new_sequence()
{
    m_buf.lock();
    while(!image_buf.empty())
        image_buf.pop();
    while(!point_buf.empty())
        point_buf.pop();
    while(!pose_buf.empty())
        pose_buf.pop();
    m_buf.unlock();
}

void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    if(!LOOP_CLOSURE)
        return;

    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();

    // detect unstable camera stream
    static double last_image_time = -1;
    if (last_image_time == -1)
        last_image_time = image_msg->header.stamp.toSec();
    else if (image_msg->header.stamp.toSec() - last_image_time > 1.0 || image_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence();
    }
    last_image_time = image_msg->header.stamp.toSec();
}

void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
    if(!LOOP_CLOSURE)
        return;

    m_buf.lock();
    point_buf.push(point_msg);
    m_buf.unlock();
}

void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    if(!LOOP_CLOSURE)
        return;

    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}

void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    m_process.lock();
    tic = Vector3d(pose_msg->pose.pose.position.x,
                   pose_msg->pose.pose.position.y,
                   pose_msg->pose.pose.position.z);
    qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                      pose_msg->pose.pose.orientation.x,
                      pose_msg->pose.pose.orientation.y,
                      pose_msg->pose.pose.orientation.z).toRotationMatrix();
    m_process.unlock();
}

/**
 * [功能描述]：回环检测的主处理循环，负责数据同步、关键帧选择和回环检测
 * 
 * 主要流程：
 * 1. 多传感器数据时间同步
 * 2. 关键帧筛选（时间和空间间隔控制）
 * 3. 数据格式转换和解析
 * 4. 关键帧构建和回环检测
 */
void process()
{
    // ########################################
    // ##### 第1步：功能开关检查 #####
    // ########################################
    
    // 如果回环闭合功能被禁用，直接返回
    // 这样可以节省计算资源，适用于只需要视觉里程计的场景
    if (!LOOP_CLOSURE)
        return;

    // ########################################
    // ##### 第2步：主处理循环 #####
    // ########################################
    
    while (ros::ok())
    {
        // 初始化消息指针
        sensor_msgs::ImageConstPtr image_msg = NULL;       // 图像消息
        sensor_msgs::PointCloudConstPtr point_msg = NULL;  // 特征点云消息
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;      // 位姿消息

        // ########################################
        // ##### 第3步：多传感器数据时间同步 #####
        // ########################################
        
        // 加锁保护缓冲区访问，确保线程安全
        m_buf.lock();
        
        // 检查三个缓冲区都有数据才进行处理
        if(!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            // ===== 情况1：图像时间戳晚于位姿时间戳 =====
            if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                // 丢弃过早的位姿数据，等待时间匹配的数据
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
            // ===== 情况2：图像时间戳晚于特征点时间戳 =====
            else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec())
            {
                // 丢弃过早的特征点数据，等待时间匹配的数据
                point_buf.pop();
                printf("throw point at beginning\n");
            }
            // ===== 情况3：找到时间同步的数据组合 =====
            else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() 
                && point_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec())
            {
                // 以位姿消息的时间戳为基准进行同步
                pose_msg = pose_buf.front();
                pose_buf.pop();
                
                // 清空剩余的位姿缓冲区，保持数据新鲜
                while (!pose_buf.empty())
                    pose_buf.pop();
                
                // 找到与位姿时间戳最接近的图像消息
                while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();

                // 找到与位姿时间戳最接近的特征点消息
                while (point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    point_buf.pop();
                point_msg = point_buf.front();
                point_buf.pop();
            }
        }
        m_buf.unlock(); // 解锁缓冲区

        // ########################################
        // ##### 第4步：处理同步后的数据 #####
        // ########################################
        
        if (pose_msg != NULL)
        {
            // ===== 跳过初始帧数 =====
            // 系统启动初期的数据可能不稳定，跳过前几帧
            static int skip_first_cnt = 0;
            if (skip_first_cnt < SKIP_FIRST_CNT)
            {
                skip_first_cnt++;
                continue; // 跳过当前帧，继续下一次循环
            }

            // ===== 时间频率控制 =====
            // 控制回环检测的时间间隔，避免过于频繁的处理
            static double last_skip_time = -1;
            if (pose_msg->header.stamp.toSec() - last_skip_time < SKIP_TIME)
                continue; // 时间间隔不够，跳过当前帧
            else
                last_skip_time = pose_msg->header.stamp.toSec(); // 更新上次处理时间

            // ===== 提取当前帧位姿信息 =====
            // 从位姿消息中提取位置和旋转信息
            static Eigen::Vector3d last_t(-1e6, -1e6, -1e6); // 上一关键帧位置，初始化为远距离
            
            // 提取位置向量
            Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            
            // 提取旋转矩阵（从四元数转换）
            Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z).toRotationMatrix();

            // ===== 空间距离控制 =====
            // 只有当前位置与上一关键帧距离超过阈值时，才添加新的关键帧
            if((T - last_t).norm() > SKIP_DIST)
            {
                // ########################################
                // ##### 第5步：图像格式转换 #####
                // ########################################
                
                cv_bridge::CvImageConstPtr ptr;
                
                // 处理特殊的"8UC1"编码格式
                if (image_msg->encoding == "8UC1")
                {
                    // 手动构建标准的ROS图像消息
                    sensor_msgs::Image img;
                    img.header = image_msg->header;
                    img.height = image_msg->height;
                    img.width = image_msg->width;
                    img.is_bigendian = image_msg->is_bigendian;
                    img.step = image_msg->step;
                    img.data = image_msg->data;
                    img.encoding = "mono8"; // 修正编码格式
                    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
                }
                else
                    // 直接转换为灰度图像
                    ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
                
                cv::Mat image = ptr->image; // 获取OpenCV图像

                // ########################################
                // ##### 第6步：特征点数据解析 #####
                // ########################################
                
                // 准备特征点数据容器
                vector<cv::Point3f> point_3d;        // 3D点坐标
                vector<cv::Point2f> point_2d_uv;     // 像素坐标
                vector<cv::Point2f> point_2d_normal; // 归一化相机坐标
                vector<double> point_id;             // 特征点ID

                // 遍历特征点云中的所有点
                for (unsigned int i = 0; i < point_msg->points.size(); i++)
                {
                    // ===== 提取3D坐标 =====
                    cv::Point3f p_3d;
                    p_3d.x = point_msg->points[i].x; // 3D点的x坐标
                    p_3d.y = point_msg->points[i].y; // 3D点的y坐标
                    p_3d.z = point_msg->points[i].z; // 3D点的z坐标
                    point_3d.push_back(p_3d);

                    // ===== 提取2D坐标和ID信息 =====
                    cv::Point2f p_2d_uv, p_2d_normal;
                    double p_id;
                    
                    // 从通道数据中提取各种信息
                    // channels[i].values数组包含了特征点的多种属性
                    p_2d_normal.x = point_msg->channels[i].values[0]; // 归一化相机坐标x
                    p_2d_normal.y = point_msg->channels[i].values[1]; // 归一化相机坐标y
                    p_2d_uv.x = point_msg->channels[i].values[2];     // 像素坐标u
                    p_2d_uv.y = point_msg->channels[i].values[3];     // 像素坐标v
                    p_id = point_msg->channels[i].values[4];          // 特征点唯一ID
                    
                    // 将解析的数据添加到对应容器中
                    point_2d_normal.push_back(p_2d_normal);
                    point_2d_uv.push_back(p_2d_uv);
                    point_id.push_back(p_id);
                }

                // ########################################
                // ##### 第7步：关键帧构建和回环检测 #####
                // ########################################
                
                // 全局关键帧索引计数器
                static int global_frame_index = 0;
                
                // 创建新的关键帧对象
                // 包含完整的帧信息：时间戳、索引、位姿、图像、特征点数据
                KeyFrame* keyframe = new KeyFrame(pose_msg->header.stamp.toSec(), // 时间戳
                                                  global_frame_index,              // 全局索引
                                                  T, R,                            // 位姿信息 
                                                  image,                           // 图像数据
                                                  point_3d, point_2d_uv,          // 特征点坐标
                                                  point_2d_normal, point_id);     // 归一化坐标和ID

                // ===== 执行回环检测 =====
                // 加锁保护回环检测过程，确保线程安全
                m_process.lock();
                
                // 将关键帧添加到回环检测器中
                // 参数1: 关键帧对象
                // 参数2: 是否执行回环检测（1表示执行）
                loopDetector.addKeyFrame(keyframe, 1);
                
                m_process.unlock(); // 解锁

                // ===== 可视化关键帧位姿 =====
                // 发布关键帧位姿信息，用于rviz可视化或其他模块使用
                loopDetector.visualizeKeyPoses(pose_msg->header.stamp.toSec());

                // 更新状态变量
                global_frame_index++; // 全局帧索引递增
                last_t = T;          // 更新上一关键帧位置
            }
        }

        // ########################################
        // ##### 第8步：循环控制 #####
        // ########################################
        
        // 短暂休眠，避免CPU占用过高
        // 5毫秒的休眠时间既保证了实时性，又避免了过度的CPU消耗
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins");
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Loop Detection Started.\033[0m");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);

    // Load params
    std::string config_file;
    n.getParam("vins_config_file", config_file);
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    usleep(100);

    // Initialize global params
    fsSettings["project_name"] >> PROJECT_NAME;  
    fsSettings["image_topic"]  >> IMAGE_TOPIC;  
    fsSettings["loop_closure"] >> LOOP_CLOSURE;
    fsSettings["skip_time"]    >> SKIP_TIME;
    fsSettings["skip_dist"]    >> SKIP_DIST;
    fsSettings["debug_image"]  >> DEBUG_IMAGE;
    fsSettings["match_image_scale"] >> MATCH_IMAGE_SCALE;
    
    if (LOOP_CLOSURE)
    {
        string pkg_path = ros::package::getPath(PROJECT_NAME);

        // initialize vocabulary
        string vocabulary_file;
        fsSettings["vocabulary_file"] >> vocabulary_file;  
        vocabulary_file = pkg_path + vocabulary_file;
        loopDetector.loadVocabulary(vocabulary_file);

        // initialize brief extractor
        string brief_pattern_file;
        fsSettings["brief_pattern_file"] >> brief_pattern_file;  
        brief_pattern_file = pkg_path + brief_pattern_file;
        briefExtractor = BriefExtractor(brief_pattern_file);

        // initialize camera model
        m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(config_file.c_str());
    }

    ros::Subscriber sub_image     = n.subscribe(IMAGE_TOPIC, 30, image_callback);
    ros::Subscriber sub_pose      = n.subscribe(PROJECT_NAME + "/vins/odometry/keyframe_pose",  3, pose_callback);
    ros::Subscriber sub_point     = n.subscribe(PROJECT_NAME + "/vins/odometry/keyframe_point", 3, point_callback);
    ros::Subscriber sub_extrinsic = n.subscribe(PROJECT_NAME + "/vins/odometry/extrinsic",      3, extrinsic_callback);

    pub_match_img = n.advertise<sensor_msgs::Image>             (PROJECT_NAME + "/vins/loop/match_image", 3);
    pub_match_msg = n.advertise<std_msgs::Float64MultiArray>    (PROJECT_NAME + "/vins/loop/match_frame", 3);
    pub_key_pose  = n.advertise<visualization_msgs::MarkerArray>(PROJECT_NAME + "/vins/loop/keyframe_pose", 3);

    if (!LOOP_CLOSURE)
    {
        sub_image.shutdown();
        sub_pose.shutdown();
        sub_point.shutdown();
        sub_extrinsic.shutdown();

        pub_match_img.shutdown();
        pub_match_msg.shutdown();
        pub_key_pose.shutdown();
    }

    std::thread measurement_process;
    measurement_process = std::thread(process);

    ros::spin();

    return 0;
}