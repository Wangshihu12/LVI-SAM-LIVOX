#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

// mtx lock for two threads
std::mutex mtx_lidar;

// global variable for saving the depthCloud shared between two threads
// 保存了全局坐标系下的点云
pcl::PointCloud<PointType>::Ptr depthCloud(new pcl::PointCloud<PointType>());

// global variables saving the lidar point cloud
deque<pcl::PointCloud<PointType>> cloudQueue;
deque<double>                     timeQueue;

// global depth register for obtaining depth of a feature
DepthRegister *depthRegister;

// feature publisher for VINS estimator
ros::Publisher pub_feature;
ros::Publisher pub_match;
ros::Publisher pub_restart;

// feature tracker variables
FeatureTracker trackerData[NUM_OF_CAM];
double         first_image_time;
int            pub_count        = 1;
bool           first_image_flag = true;
double         last_image_time  = 0;
bool           init_pub         = 0;

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    // 获取当前图像的时间戳
    double cur_img_time = img_msg->header.stamp.toSec();

    // 处理第一帧图像的特殊情况
    if (first_image_flag)
    {
        first_image_flag = false;
        first_image_time = cur_img_time;  // 记录第一帧图像时间
        last_image_time  = cur_img_time;  // 记录上一帧图像时间
        return; // 第一帧直接返回，不进行处理
    }
    
    // 检测相机数据流是否稳定
    // 如果两帧间隔超过1秒或时间戳倒退，说明数据流不稳定
    if (cur_img_time - last_image_time > 1.0 || cur_img_time < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        // 重置特征跟踪器状态
        first_image_flag = true;
        last_image_time  = 0;
        pub_count        = 1;
        // 发布重启信号给VINS估计器
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = cur_img_time; // 更新上一帧时间戳

    // 频率控制：根据设定的FREQ参数控制发布频率
    // 计算当前平均帧率是否小于等于设定频率
    if (round(1.0 * pub_count / (cur_img_time - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true; // 标记当前帧需要发布
        // 如果当前帧率接近设定频率，重置频率控制计数器
        if (abs(1.0 * pub_count / (cur_img_time - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = cur_img_time;
            pub_count        = 0;
        }
    }
    else
    {
        PUB_THIS_FRAME = false; // 跳过当前帧
    }

    // 图像格式转换处理
    cv_bridge::CvImageConstPtr ptr;
    // 如果输入图像编码是"8UC1"，需要转换为"mono8"格式
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header       = img_msg->header;
        img.height       = img_msg->height;
        img.width        = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step         = img_msg->step;
        img.data         = img_msg->data;
        img.encoding     = "mono8"; // 修改编码格式
        ptr              = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        // 直接转换为MONO8格式
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat show_img = ptr->image; // 用于显示的图像
    TicToc  t_r; // 计时器

    // 对每个相机进行特征提取和跟踪
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        // 如果不是第二个相机或者不是双目跟踪模式
        if (i != 1 || !STEREO_TRACK)
            // 读取图像的特定行范围（支持多相机垂直拼接）
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), cur_img_time);
        else
        {
            // 双目跟踪模式下的第二个相机处理
            if (EQUALIZE)
            {
                // 使用CLAHE算法进行直方图均衡化增强图像对比度
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                // 直接使用原图像
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        // 显示去畸变效果（调试用）
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    // 更新特征点ID，确保跨帧特征点的一致性
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break; // 所有特征点ID更新完成
    }

    // 如果当前帧需要发布特征点
    if (PUB_THIS_FRAME)
    {
        pub_count++; // 发布计数器递增
        
        // 创建特征点云消息
        sensor_msgs::PointCloudPtr  feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;          // 特征点ID通道
        sensor_msgs::ChannelFloat32 u_of_point;           // 像素u坐标通道
        sensor_msgs::ChannelFloat32 v_of_point;           // 像素v坐标通道
        sensor_msgs::ChannelFloat32 velocity_x_of_point;  // x方向光流速度通道
        sensor_msgs::ChannelFloat32 velocity_y_of_point;  // y方向光流速度通道

        // 设置消息头信息
        feature_points->header.stamp    = img_msg->header.stamp;
        feature_points->header.frame_id = "vins_body";

        // 用于记录每个相机的特征点ID，避免重复
        vector<set<int>> hash_ids(NUM_OF_CAM);
        
        // 遍历每个相机的特征点
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts       = trackerData[i].cur_un_pts;   // 去畸变后的归一化坐标
            auto &cur_pts      = trackerData[i].cur_pts;      // 当前帧像素坐标
            auto &ids          = trackerData[i].ids;          // 特征点ID
            auto &pts_velocity = trackerData[i].pts_velocity; // 特征点光流速度
            
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                // 只处理跟踪次数大于1的稳定特征点
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    
                    // 创建3D点，z坐标设为1（归一化坐标）
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    // 添加特征点信息到各个通道
                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);  // 全局唯一ID
                    u_of_point.values.push_back(cur_pts[j].x);            // 像素u坐标
                    v_of_point.values.push_back(cur_pts[j].y);            // 像素v坐标
                    velocity_x_of_point.values.push_back(pts_velocity[j].x); // x方向速度
                    velocity_y_of_point.values.push_back(pts_velocity[j].y); // y方向速度
                }
            }
        }

        // 将所有通道添加到特征点云消息中
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);

        // 从激光雷达点云中获取特征点的深度信息
        // 从全局的点云地图中提取特征点的深度信息
        pcl::PointCloud<PointType>::Ptr depth_cloud_temp(new pcl::PointCloud<PointType>());
        mtx_lidar.lock();   // 加锁保护共享的点云数据
        *depth_cloud_temp = *depthCloud;
        mtx_lidar.unlock(); // 解锁

        // 使用深度注册器获取特征点深度
        sensor_msgs::ChannelFloat32 depth_of_points =
            depthRegister->get_depth(img_msg->header.stamp, show_img, depth_cloud_temp,
                                     trackerData[0].m_camera, feature_points->points);
        feature_points->channels.push_back(depth_of_points);

        // 跳过第一帧图像的发布，因为第一帧没有光流速度信息
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_feature.publish(feature_points); // 发布特征点给VINS估计器

        // 发布带有特征点标记的图像用于可视化
        if (pub_match.getNumSubscribers() != 0)
        {
            // 将图像转换为RGB格式用于可视化
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::RGB8);
            cv::Mat stereo_img = ptr->image;

            // 为每个相机的图像区域绘制特征点
            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                // 获取当前相机对应的图像区域
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                // 在图像上绘制每个特征点
                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    if (SHOW_TRACK)
                    {
                        // 根据跟踪次数显示特征点颜色（跟踪越久颜色越绿）
                        double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                        cv::circle(tmp_img, trackerData[i].cur_pts[j], 4,
                                   cv::Scalar(255 * (1 - len), 255 * len, 0), 4);
                    }
                    else
                    {
                        // 根据深度信息显示特征点颜色
                        if (j < depth_of_points.values.size())
                        {
                            if (depth_of_points.values[j] > 0)
                                // 有深度信息：绿色圆圈
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 4,
                                           cv::Scalar(0, 255, 0), 4);
                            else
                                // 无深度信息：红色圆圈
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 4,
                                           cv::Scalar(0, 0, 255), 4);
                        }
                    }
                }
            }

            // 发布可视化图像
            pub_match.publish(ptr->toImageMsg());
        }
    }
}

void lidar_callback(const sensor_msgs::PointCloud2ConstPtr &laser_msg)
{
    // 静态变量，用于记录激光雷达帧计数，初始值为-1
    static int lidar_count = -1;
    
    // 激光雷达跳帧处理：LIDAR_SKIP = 3，表示每4帧处理1帧
    // 这样做是为了降低计算负担，因为激光雷达频率通常比相机高
    if (++lidar_count % (LIDAR_SKIP + 1) != 0)
        return;

    // ========== 第0步：监听坐标变换 ==========
    // 静态变量，避免重复创建TF监听器和变换对象
    static tf::TransformListener listener;
    static tf::StampedTransform  transform;
    
    try
    {
        // waitForTransform参数说明：
        // - "vins_world": 父坐标系（世界坐标系）
        // - "vins_body_ros": 子坐标系（机器人本体坐标系）
        // - laser_msg->header.stamp: 变换的时间戳（激光雷达数据的时间戳）
        // - ros::Duration(0.01): 最大等待时间0.01秒
        // 目的：获取从机器人本体到世界坐标系的变换矩阵 T_W_I
        listener.waitForTransform("vins_world", "vins_body_ros", laser_msg->header.stamp,
                                  ros::Duration(0.01));
        listener.lookupTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, transform);
    }
    catch (tf::TransformException ex)
    {
        // 如果获取变换失败，直接返回（不处理当前帧）
        // 注释掉错误输出是为了避免频繁打印错误信息
        // ROS_ERROR("lidar no tf");
        return;
    }

    // 从TF变换中提取位置和姿态信息
    double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
    
    // 提取平移部分（位置信息）
    xCur = transform.getOrigin().x();
    yCur = transform.getOrigin().y();
    zCur = transform.getOrigin().z();
    
    // 提取旋转部分并转换为欧拉角（姿态信息）
    tf::Matrix3x3 m(transform.getRotation());
    m.getRPY(rollCur, pitchCur, yawCur);
    
    // 构建当前时刻的变换矩阵，用于后续点云坐标变换
    Eigen::Affine3f transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

    // ========== 第1步：ROS消息转换为PCL点云 ==========
    pcl::PointCloud<PointType>::Ptr laser_cloud_in(new pcl::PointCloud<PointType>());
    // 将ROS的PointCloud2消息转换为PCL点云格式，便于后续处理
    pcl::fromROSMsg(*laser_msg, *laser_cloud_in);

    // ========== 第2步：点云降采样（节省内存） ==========
    pcl::PointCloud<PointType>::Ptr  laser_cloud_in_ds(new pcl::PointCloud<PointType>());
    // 静态变量，避免重复创建体素滤波器对象
    static pcl::VoxelGrid<PointType> downSizeFilter;
    
    // 设置体素大小为0.2x0.2x0.2米，将密集点云降采样
    // 降采样可以减少数据量，提高处理速度，同时保持点云的主要特征
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(laser_cloud_in);
    downSizeFilter.filter(*laser_cloud_in_ds);
    *laser_cloud_in = *laser_cloud_in_ds;

    // ========== 第3步：滤波筛选点云（只保留相机视野内的点） ==========
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_filter(new pcl::PointCloud<PointType>());
    
    for (int i = 0; i < (int)laser_cloud_in->size(); ++i)
    {
        PointType p = laser_cloud_in->points[i];
        
        // 筛选条件说明：
        // 1. p.x >= 0: 只保留前方的点（假设激光雷达x轴朝前）
        // 2. abs(p.y / p.x) <= 10: 限制左右视野角度（约84度视野角）
        // 3. abs(p.z / p.x) <= 10: 限制上下视野角度（约84度视野角）
        // 这样可以只保留大致在相机视野范围内的激光点，减少无效数据
        if (p.x >= 0 && abs(p.y / p.x) <= 10 && abs(p.z / p.x) <= 10)
            laser_cloud_in_filter->push_back(p);
    }
    *laser_cloud_in = *laser_cloud_in_filter;

    // ========== 第4步：激光雷达坐标系转IMU坐标系 ==========
    pcl::PointCloud<PointType>::Ptr laser_cloud_offset(new pcl::PointCloud<PointType>());
    
    // 使用外参将点云从激光雷达坐标系转换到IMU坐标系
    // L_I_TX, L_I_TY, L_I_TZ: 激光雷达相对于IMU的平移
    // L_I_RX, L_I_RY, L_I_RZ: 激光雷达相对于IMU的旋转
    Eigen::Affine3f transOffset = pcl::getTransformation(L_I_TX, L_I_TY, L_I_TZ, L_I_RX, L_I_RY, L_I_RZ);
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_offset, transOffset);
    *laser_cloud_in = *laser_cloud_offset;

    // ========== 第5步：将点云变换到全局里程计坐标系 ==========
    pcl::PointCloud<PointType>::Ptr laser_cloud_global(new pcl::PointCloud<PointType>());
    
    // 使用当前位姿将点云从IMU坐标系变换到世界坐标系
    // 这样所有历史点云都统一在同一个全局坐标系下
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_global, transNow);

    // ========== 第6步：保存新的点云到队列 ==========
    double timeScanCur = laser_msg->header.stamp.toSec();
    
    // 将变换后的全局点云和对应时间戳加入队列
    // 这些队列用于维护一个滑动窗口的点云历史
    cloudQueue.push_back(*laser_cloud_global);
    timeQueue.push_back(timeScanCur);

    // ========== 第7步：移除过旧的点云 ==========
    while (!timeQueue.empty())
    {
        // 如果最老的点云超过5秒，就将其移除
        // 这样可以控制内存使用，避免点云数据无限增长
        if (timeScanCur - timeQueue.front() > 5.0)
        {
            cloudQueue.pop_front();
            timeQueue.pop_front();
        }
        else
        {
            break; // 一旦遇到未过期的点云就停止删除
        }
    }

    // 加锁保护全局共享的点云数据，确保线程安全
    std::lock_guard<std::mutex> lock(mtx_lidar);
    
    // ========== 第8步：融合全局点云 ==========
    depthCloud->clear(); // 清空之前的全局点云
    
    // 将队列中的所有点云合并成一个大的全局点云
    // 这个全局点云包含了过去5秒内的所有激光雷达数据
    for (int i = 0; i < (int)cloudQueue.size(); ++i)
        *depthCloud += cloudQueue[i];

    // ========== 第9步：对全局点云进行最终降采样 ==========
    pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
    
    // 对融合后的大点云再次降采样，进一步控制数据量
    // 这样既保证了点云的覆盖范围，又控制了计算复杂度
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(depthCloud);
    downSizeFilter.filter(*depthCloudDS);
    *depthCloud = *depthCloudDS;
}

int main(int argc, char **argv)
{
    // initialize ROS node
    ros::init(argc, argv, "vins");
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Feature Tracker Started.\033[0m");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);
    readParameters(n);

    // read camera params
    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    // load fisheye mask to remove features on the boundry
    if (FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if (!trackerData[i].fisheye_mask.data)
            {
                ROS_ERROR("load fisheye mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    // initialize depthRegister (after readParameters())
    depthRegister = new DepthRegister(n);

    // subscriber to image and lidar
    ros::Subscriber sub_img   = n.subscribe(IMAGE_TOPIC, 5, img_callback);
    ros::Subscriber sub_lidar = n.subscribe(POINT_CLOUD_TOPIC, 5, lidar_callback);
    if (!USE_LIDAR)
        sub_lidar.shutdown();

    // messages to vins estimator
    pub_feature = n.advertise<sensor_msgs::PointCloud>(PROJECT_NAME + "/vins/feature/feature", 5);
    pub_match   = n.advertise<sensor_msgs::Image>(PROJECT_NAME + "/vins/feature/feature_img", 5);
    pub_restart = n.advertise<std_msgs::Bool>(PROJECT_NAME + "/vins/feature/restart", 5);

    // two ROS spinners for parallel processing (image and lidar)
    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    return 0;
}