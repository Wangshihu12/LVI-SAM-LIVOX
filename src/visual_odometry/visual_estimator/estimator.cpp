#include "estimator.h"

Estimator::Estimator() : f_manager{Rs}
{
    failureCount = -1;
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info   = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td                            = TD;
}

void Estimator::clearState()
{
    ++failureCount;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false, sum_of_back = 0;
    sum_of_front      = 0;
    frame_count       = 0;
    solver_flag       = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration       = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
}

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration,
                           const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0     = linear_acceleration;
        gyr_0     = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] =
            new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);

        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int      j        = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr   = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc   = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * [功能描述]：处理图像特征点数据，执行VINS系统的核心估计流程
 * 
 * @param image：图像特征点数据，包含特征点ID、相机ID和8维特征向量
 * @param lidar_initialization_info：激光雷达里程计提供的初始化信息
 * @param header：图像消息头，包含时间戳等信息
 * 
 * 主要功能：特征管理、系统初始化、非线性优化、失效检测、滑动窗口维护
 */
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &image,
                             const vector<float>    &lidar_initialization_info,
                             const std_msgs::Header &header)
{
    // ########################################
    // ##### 第1步：特征点管理和边缘化策略 #####
    // ########################################
    
    // 添加新图像特征并检查视差
    // f_manager.addFeatureCheckParallax() 返回值表示是否有足够的视差：
    // - true: 视差足够大，可以进行三角化，边缘化最旧的帧
    // - false: 视差不足，边缘化第二新的帧以保持关键帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;        // 边缘化最旧帧
    else
        marginalization_flag = MARGIN_SECOND_NEW; // 边缘化第二新帧

    // 特殊情况：如果系统处于初始化阶段且激光里程计可用
    // 强制边缘化旧帧以加速初始化过程
    if (solver_flag == INITIAL && lidar_initialization_info[0] >= 0)
        marginalization_flag = MARGIN_OLD;

    // ##### 第2步：图像帧数据存储 #####
    Headers[frame_count] = header; // 保存当前帧的消息头

    // 创建图像帧对象，包含特征点、激光里程计信息和时间戳
    ImageFrame imageframe(image, lidar_initialization_info, header.stamp.toSec());
    
    // 将当前IMU预积分结果关联到该图像帧
    imageframe.pre_integration = tmp_pre_integration;
    
    // 将图像帧按时间戳存储到历史帧容器中
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));

    // 为下一帧创建新的IMU预积分对象
    // 使用当前帧的偏差估计值作为初始值
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // ########################################
    // ##### 第3步：相机-IMU外参标定 #####
    // ########################################
    
    // ESTIMATE_EXTRINSIC == 2 表示需要在线标定旋转外参
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("calibrating extrinsic param, rotation movement is needed");
        
        if (frame_count != 0)
        {
            // 获取前一帧和当前帧之间的特征点对应关系
            vector<pair<Vector3d, Vector3d>> corres =
                f_manager.getCorresponding(frame_count - 1, frame_count);
            
            Matrix3d calib_ric; // 标定得到的旋转外参
            
            // 使用视觉和IMU数据进行旋转外参标定
            // 原理：利用特征点对应关系和IMU旋转积分进行最小二乘求解
            if (initial_ex_rotation.CalibrationExRotation(
                    corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                
                // 更新外参估计值
                ric[0]             = calib_ric;  // 估计器内部使用
                RIC[0]             = calib_ric;  // 全局参数
                ESTIMATE_EXTRINSIC = 1;          // 标记外参标定完成
            }
        }
    }

    // ########################################
    // ##### 第4步：系统状态处理 #####
    // ########################################
    
    if (solver_flag == INITIAL)
    {
        // ===== 初始化阶段处理 =====
        
        if (frame_count == WINDOW_SIZE)
        {
            // 当滑动窗口填满时尝试初始化
            bool result = false;
            
            // 初始化条件检查：
            // 1. 外参不需要在线标定 (ESTIMATE_EXTRINSIC != 2)
            // 2. 距离上次初始化尝试超过0.1秒
            if (ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
                // 执行视觉-惯性初始化
                // initialStructure() 包括：特征点三角化、尺度估计、重力对齐等
                result            = initialStructure();
                initial_timestamp = header.stamp.toSec();
            }
            
            if (result)
            {
                // ===== 初始化成功 =====
                ROS_INFO("Initialization finish!");
                
                solver_flag = NON_LINEAR; // 切换到非线性优化模式
                
                // 执行完整的非线性优化
                solveOdometry();
                
                // 维护滑动窗口
                slideWindow();
                
                // 清理失效的特征点
                f_manager.removeFailures();
                
                // 保存关键状态用于后续处理
                last_R  = Rs[WINDOW_SIZE];  // 最新帧旋转
                last_P  = Ps[WINDOW_SIZE];  // 最新帧位置
                last_R0 = Rs[0];            // 最旧帧旋转
                last_P0 = Ps[0];            // 最旧帧位置
            }
            else
            {
                // ===== 初始化失败 =====
                // 仅执行滑动窗口维护，等待下次初始化机会
                slideWindow();
            }
        }
        else
        {
            // 滑动窗口未满，继续积累帧数据
            frame_count++;
        }
    }
    else
    {
        // ===== 非线性优化阶段处理 =====
        
        // 执行主要的非线性优化求解
        // 优化状态变量：位姿、速度、IMU偏差、特征点深度等
        solveOdometry();

        // ##### 系统失效检测 #####
        if (failureDetection())
        {
            ROS_ERROR("VINS failure detection!");
            
            // 系统失效处理流程
            failure_occur = 1;    // 标记失效发生
            clearState();         // 清空所有状态
            setParameter();       // 重新设置参数
            
            ROS_ERROR("VINS system reboot!");
            return; // 直接返回，等待系统重启
        }

        // ##### 正常处理流程 #####
        
        // 维护滑动窗口
        slideWindow();
        
        // 清理失效的特征点
        f_manager.removeFailures();

        // ##### 准备VINS输出结果 #####
        key_poses.clear();
        
        // 收集滑动窗口内所有关键帧的位姿
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        // 更新最新状态用于输出和可视化
        last_R  = Rs[WINDOW_SIZE];  // 最新帧旋转矩阵
        last_P  = Ps[WINDOW_SIZE];  // 最新帧位置向量
        last_R0 = Rs[0];            // 窗口首帧旋转矩阵  
        last_P0 = Ps[0];            // 窗口首帧位置向量
    }
}

/**
 * [功能描述]：执行VINS系统的初始化，包括激光雷达初始化、视觉SFM、IMU对齐等
 * 
 * @return：初始化是否成功
 * 
 * 主要步骤：
 * 1. 优先尝试激光雷达辅助初始化
 * 2. 检查IMU激励充分性
 * 3. 执行全局SFM重建
 * 4. 求解所有帧的PnP位姿
 * 5. 视觉-惯性对齐
 */
bool Estimator::initialStructure()
{
    // ########################################
    // ##### 第1步：激光雷达辅助初始化 #####
    // ########################################
    {
        bool lidar_info_available = true;

        // 清空所有帧的关键帧标记
        // 重新确定哪些帧是关键帧
        for (map<double, ImageFrame>::iterator frame_it = all_image_frame.begin();
             frame_it != all_image_frame.end(); frame_it++)
            frame_it->second.is_key_frame = false;

        // ===== 检查滑动窗口内激光里程计信息的有效性 =====
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            // 检查激光里程计数据的连续性和一致性
            // reset_id < 0: 激光里程计数据不可用
            // reset_id不一致: 激光里程计发生了重定位，数据不连续
            if (all_image_frame[Headers[i].stamp.toSec()].reset_id < 0 ||
                all_image_frame[Headers[i].stamp.toSec()].reset_id !=
                    all_image_frame[Headers[0].stamp.toSec()].reset_id)
            {
                lidar_info_available = false;
                ROS_INFO("Lidar initialization info not enough.");
                break;
            }
        }

        // ===== 激光雷达初始化路径 =====
        if (lidar_info_available == true)
        {
            // 直接使用激光里程计提供的状态信息进行初始化
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                // 从激光里程计获取的状态信息
                Ps[i]  = all_image_frame[Headers[i].stamp.toSec()].T;    // 位置
                Rs[i]  = all_image_frame[Headers[i].stamp.toSec()].R;    // 旋转
                Vs[i]  = all_image_frame[Headers[i].stamp.toSec()].V;    // 速度
                Bas[i] = all_image_frame[Headers[i].stamp.toSec()].Ba;   // 加速度计偏差
                Bgs[i] = all_image_frame[Headers[i].stamp.toSec()].Bg;   // 陀螺仪偏差

                // 使用更新后的偏差重新传播IMU预积分
                pre_integrations[i]->repropagate(Bas[i], Bgs[i]);

                // 标记所有窗口内的帧为关键帧
                all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
            }

            // 更新重力向量（从激光里程计获取）
            g = Eigen::Vector3d(0, 0, all_image_frame[Headers[0].stamp.toSec()].gravity);

            // ===== 重置并三角化所有特征点 =====
            // 清空所有特征点的深度信息
            VectorXd dep = f_manager.getDepthVector();
            for (int i = 0; i < dep.size(); i++)
                dep[i] = -1; // -1表示深度未知
            f_manager.clearDepth(dep);

            // 基于激光里程计提供的位姿进行特征点三角化
            Vector3d TIC_TMP[NUM_OF_CAM];
            for (int i = 0; i < NUM_OF_CAM; i++)
                TIC_TMP[i].setZero(); // 相机-IMU平移外参临时设为0
            
            ric[0] = RIC[0]; // 设置相机-IMU旋转外参
            f_manager.setRic(ric);
            f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

            return true; // 激光雷达初始化成功，直接返回
        }
    }

    // ########################################
    // ##### 第2步：IMU可观测性检查 #####
    // ########################################
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d                          sum_g; // 重力加速度累积和
        
        // 计算所有帧间的平均重力加速度
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end();
             frame_it++)
        {
            double   dt    = frame_it->second.pre_integration->sum_dt;    // 时间间隔
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt; // 估计的重力加速度
            sum_g += tmp_g;
        }
        
        // 计算平均重力加速度
        Vector3d aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        
        // 计算重力加速度的方差，用于评估IMU激励是否充分
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end();
             frame_it++)
        {
            double   dt    = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        
        // 检查IMU激励是否充分
        // 方差太小说明运动不够剧烈，难以准确估计重力方向和尺度
        if (var < 0.25)
        {
            ROS_INFO("Trying to initialize VINS, IMU excitation not enough!");
            // 注意：这里只是警告，不强制返回false，允许继续尝试初始化
        }
    }

    // ########################################
    // ##### 第3步：全局SFM（Structure from Motion）#####
    // ########################################
    
    // 准备SFM所需的数据结构
    Quaterniond        Q[frame_count + 1];     // 各帧旋转四元数
    Vector3d           T[frame_count + 1];     // 各帧平移向量
    map<int, Vector3d> sfm_tracked_points;     // SFM重建的三维点
    vector<SFMFeature> sfm_f;                  // SFM特征点数据

    // 将特征管理器中的特征点转换为SFM格式
    for (auto &it_per_id : f_manager.feature)
    {
        int        imu_j = it_per_id.start_frame - 1; // 起始帧索引
        SFMFeature tmp_feature;
        tmp_feature.state = false;                     // 初始状态为未三角化
        tmp_feature.id    = it_per_id.feature_id;      // 特征点ID

        // 收集该特征点在各帧中的观测
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point; // 归一化相机坐标
            tmp_feature.observation.push_back(
                make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }

    // ===== 计算关键帧对之间的相对位姿 =====
    Matrix3d relative_R; // 相对旋转矩阵
    Vector3d relative_T; // 相对平移向量
    int      l;          // 参考帧索引
    
    // 寻找具有足够视差的帧对并计算相对位姿
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false; // 特征点不足或视差不够，初始化失败
    }

    // ===== 执行全局SFM重建 =====
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD; // SFM失败，标记边缘化策略
        return false;
    }

    // ########################################
    // ##### 第4步：为所有帧求解PnP位姿 #####
    // ########################################
    
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator      it;
    frame_it = all_image_frame.begin();
    
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // 准备PnP求解所需的变量
        cv::Mat r, rvec, t, D, tmp_r;
        
        // ===== 处理滑动窗口内的关键帧 =====
        if ((frame_it->first) == Headers[i].stamp.toSec())
        {
            // 直接使用SFM结果设置关键帧位姿
            frame_it->second.is_key_frame = true;
            // 转换坐标系：SFM坐标系 → IMU坐标系
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        
        // 处理时间戳不匹配的情况
        if ((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        
        // ===== 为非关键帧使用PnP求解位姿 =====
        // 构造PnP初始猜测
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec); // 旋转矩阵转换为旋转向量
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false; // 标记为非关键帧
        
        // 收集3D-2D对应点对
        vector<cv::Point3f> pts_3_vector; // 3D点
        vector<cv::Point2f> pts_2_vector; // 对应的2D观测
        
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                // 查找该特征点的3D位置
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    // 添加3D点
                    Vector3d    world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    
                    // 添加对应的2D观测
                    Vector2d    img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        
        // PnP求解需要的相机内参矩阵（这里使用归一化坐标，所以是单位矩阵）
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        
        // 检查3D-2D对应点数量是否足够
        if (pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        
        // 执行PnP求解
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        
        // 转换PnP结果格式
        cv::Rodrigues(rvec, r);  // 旋转向量转旋转矩阵
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose(); // 转置得到正确的旋转矩阵
        
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp); // 计算正确的平移向量
        
        // 保存位姿结果（转换到IMU坐标系）
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    // ########################################
    // ##### 第5步：视觉-惯性对齐 #####
    // ########################################
    
    // 执行视觉SFM结果与IMU预积分的对齐
    // 包括：尺度估计、重力方向对齐、速度和偏差初始化
    if (visualInitialAlign())
        return true;  // 对齐成功，初始化完成
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false; // 对齐失败，初始化失败
    }
}

/**
 * [功能描述]：执行视觉-惯性对齐，将纯视觉SFM结果与IMU数据对齐
 * 
 * @return：对齐是否成功
 * 
 * 主要任务：
 * 1. 求解视觉-惯性系统的尺度因子
 * 2. 估计重力方向和大小
 * 3. 初始化速度和IMU偏差
 * 4. 统一坐标系，完成多传感器融合初始化
 */
bool Estimator::visualInitialAlign()
{
    VectorXd x; // 优化变量：包含各帧速度和尺度因子
    
    // ########################################
    // ##### 第1步：执行视觉-惯性联合优化 #####
    // ########################################
    
    // 调用核心对齐算法，求解：
    // - 尺度因子s（视觉尺度 → IMU尺度）
    // - 重力向量g（大小和方向）
    // - 各帧速度v
    // - 陀螺仪偏差Bgs
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        ROS_INFO("solve gravity failed, try again!");
        return false; // 对齐失败，需要重新尝试初始化
    }

    // ########################################
    // ##### 第2步：更新系统状态变量 #####
    // ########################################
    
    // 将SFM得到的位姿结果复制到滑动窗口状态变量中
    for (int i = 0; i <= frame_count; i++)
    {
        // 从历史图像帧中获取SFM计算的位姿
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R; // 旋转矩阵
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T; // 位置向量
        
        // 更新滑动窗口中的位姿状态
        Ps[i] = Pi; // 位置
        Rs[i] = Ri; // 旋转
        
        // 标记所有帧为关键帧（初始化阶段所有帧都很重要）
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    // ########################################
    // ##### 第3步：重置特征点深度信息 #####
    // ########################################
    
    // 清空所有特征点的深度估计，准备重新三角化
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1; // -1表示深度未知/无效
    f_manager.clearDepth(dep);

    // ===== 基于对齐后的位姿重新三角化特征点 =====
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero(); // 相机-IMU平移外参临时设为0
    
    ric[0] = RIC[0]; // 设置相机-IMU旋转外参
    f_manager.setRic(ric);
    
    // 使用对齐后的位姿进行三角化
    // 此时的位姿已经具有正确的尺度信息
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    // ########################################
    // ##### 第4步：应用尺度因子校正 #####
    # ##########################################
    
    // 提取求解得到的尺度因子
    double s = (x.tail<1>())(0); // x向量的最后一个元素是尺度因子
    
    // ===== 重新传播IMU预积分（使用更新后的偏差） =====
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // 使用零加速度计偏差和更新后的陀螺仪偏差重新传播
        // 加速度计偏差在后续优化中再精确估计
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    
    // ===== 应用尺度因子到位置状态 =====
    // 将视觉尺度的位置转换为IMU尺度的位置
    for (int i = frame_count; i >= 0; i--)
        // 位置校正公式：P_IMU = s * P_visual - R * t_IC - (s * P0_visual - R0 * t_IC)
        // 这里简化为：P_corrected = s * P - R * TIC - (s * P0 - R0 * TIC)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);

    // ########################################
    // ##### 第5步：初始化速度状态 #####
    // ########################################
    
    int                               kv = -1; // 关键帧计数器
    map<double, ImageFrame>::iterator frame_i;
    
    // 遍历所有历史图像帧，为关键帧设置速度
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++; // 关键帧索引递增
            
            // 从优化结果x中提取该帧的速度
            // x.segment<3>(kv * 3)：提取第kv个关键帧的3维速度向量
            // 需要转换到IMU坐标系：V_IMU = R * V_camera
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    // ########################################
    // ##### 第6步：更新特征点深度 #####
    // ########################################
    
    // 对所有有效特征点应用尺度因子
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // 更新观测次数
        
        // 只处理满足条件的特征点：
        // 1. 至少被2帧观测到
        // 2. 起始帧不能太靠近窗口末尾
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
            
        // 将视觉尺度的深度转换为IMU尺度的深度
        it_per_id.estimated_depth *= s;
    }

    // ########################################
    // ##### 第7步：重力对齐和坐标系调整 #####
    // ########################################
    
    // ===== 构建重力对齐旋转矩阵 =====
    // 将估计的重力向量对齐到标准重力方向[0,0,-g]
    Matrix3d R0 = Utility::g2R(g); // 从重力向量构建旋转矩阵
    
    // 提取当前坐标系的偏航角
    double yaw = Utility::R2ypr(R0 * Rs[0]).x(); // 获取yaw角度
    
    // 构建消除偏航角的旋转矩阵，保持重力对齐的同时固定偏航角
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    
    // 更新重力向量到标准坐标系
    g = R0 * g;
    
    // ===== 应用坐标系调整到所有状态 =====
    Matrix3d rot_diff = R0; // 坐标系调整旋转矩阵
    
    // 对所有帧的状态应用坐标系旋转
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i]; // 位置
        Rs[i] = rot_diff * Rs[i]; // 旋转
        Vs[i] = rot_diff * Vs[i]; // 速度
    }
    
    // ##### 调试输出 #####
    ROS_DEBUG_STREAM("g0     " << g.transpose());           // 最终重力向量
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); // 第一帧姿态角

    return true; // 视觉-惯性对齐成功完成
}

/**
 * [功能描述]：寻找与最新帧具有足够对应点和视差的参考帧，用于SFM初始化
 * 
 * @param relative_R：输出的相对旋转矩阵
 * @param relative_T：输出的相对平移向量  
 * @param l：输出的参考帧索引
 * @return：是否成功找到合适的参考帧对
 * 
 * 选择标准：
 * 1. 足够的特征点对应关系（>20个）
 * 2. 足够的视差（平均视差>30像素）
 * 3. 能够成功求解相对位姿
 */
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // ########################################
    // ##### 遍历滑动窗口寻找最佳参考帧 #####
    // ########################################
    
    // 从滑动窗口的第一帧开始，逐帧检查与最新帧的匹配质量
    // i: 候选参考帧索引，WINDOW_SIZE: 最新帧索引
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        // ===== 第1步：获取特征点对应关系 =====
        vector<pair<Vector3d, Vector3d>> corres;
        
        // 获取第i帧与最新帧之间的特征点对应关系
        // 返回的是归一化相机坐标对
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        
        // ===== 第2步：检查对应点数量 =====
        // 至少需要20个对应点才能进行可靠的位姿估计
        // 这个阈值确保有足够的约束来求解相对位姿
        if (corres.size() > 20)
        {
            // ===== 第3步：计算平均视差 =====
            double sum_parallax = 0;     // 视差总和
            double average_parallax;     // 平均视差
            
            // 遍历所有对应点对，计算每对点的视差
            for (int j = 0; j < int(corres.size()); j++)
            {
                // 提取第i帧和最新帧中对应特征点的归一化坐标
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));   // 第i帧特征点
                Vector2d pts_1(corres[j].second(0), corres[j].second(1)); // 最新帧特征点
                
                // 计算两点之间的欧几里得距离作为视差
                // 视差越大表示相机运动越明显，三角化精度越高
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            
            // 计算所有对应点的平均视差
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            
            // ===== 第4步：视差阈值检查和位姿求解 =====
            // 条件1：average_parallax * 460 > 30
            // - 460: 近似的相机焦距（像素单位），用于将归一化坐标视差转换为像素视差
            // - 30: 像素视差阈值，确保有足够的基线长度进行准确三角化
            // - 这个条件等价于：像素视差 > 30像素
            //
            // 条件2：solveRelativeRT成功求解相对位姿
            // - 使用5点算法或8点算法求解本质矩阵
            // - 从本质矩阵分解得到相对旋转和平移
            if (average_parallax * 460 > 30 &&
                m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i; // 设置参考帧索引
                
                // 调试输出：显示选择的参考帧信息
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the "
                          "whole structure",
                          average_parallax * 460, l);
                
                return true; // 成功找到合适的参考帧对
            }
        }
    }
    
    // 如果遍历完所有帧都没有找到合适的参考帧，返回失败
    return false;
}

/**
 * [功能描述]：执行VINS系统的里程计求解，包括特征点三角化和非线性优化
 * 
 * 执行条件：
 * 1. 滑动窗口已填满（有足够的帧数据）
 * 2. 系统处于非线性优化状态（已完成初始化）
 * 
 * 主要步骤：
 * 1. 基于当前位姿估计三角化特征点
 * 2. 执行非线性bundle adjustment优化
 */
void Estimator::solveOdometry()
{
    // ########################################
    // ##### 第1步：滑动窗口填充检查 #####
    // ########################################
    
    // 检查滑动窗口是否已经填满
    // frame_count < WINDOW_SIZE 说明还在收集初始帧数据阶段
    // 只有窗口填满后才有足够的约束进行可靠的优化
    if (frame_count < WINDOW_SIZE)
        return; // 窗口未满，直接返回等待更多帧

    // ########################################
    // ##### 第2步：系统状态检查 #####
    // ########################################
    
    // 检查求解器是否处于非线性优化状态
    // solver_flag == NON_LINEAR 表示系统已经完成初始化
    // 包括：视觉-惯性对齐、尺度估计、重力估计等
    if (solver_flag == NON_LINEAR)
    {
        // ===== 第3步：特征点三角化 =====
        // 基于当前的位姿估计，重新三角化所有特征点
        // 参数说明：
        // - Ps: 滑动窗口内各帧的位置状态
        // - tic: 相机相对于IMU的平移外参
        // - ric: 相机相对于IMU的旋转外参
        // 
        // 三角化作用：
        // 1. 更新特征点的3D位置估计
        // 2. 为后续优化提供准确的点云信息
        // 3. 剔除三角化失败的无效特征点
        f_manager.triangulate(Ps, tic, ric);

        // ===== 第4步：非线性优化 =====
        // 执行基于滑动窗口的bundle adjustment优化
        // 优化变量包括：
        // - 位姿状态：位置Ps、旋转Rs、速度Vs
        // - IMU偏差：加速度计偏差Bas、陀螺仪偏差Bgs  
        // - 特征点深度：inv_depth
        // - 外参数：相机-IMU外参（如果需要在线标定）
        // - 时间偏移：相机-IMU时间同步偏差td
        //
        // 约束项包括：
        // - IMU预积分约束：连接相邻帧的运动模型
        // - 视觉重投影约束：特征点观测误差
        // - 边缘化约束：来自被边缘化帧的先验信息
        optimization();
    }
    
    // 注意：如果 solver_flag != NON_LINEAR（即系统仍在初始化阶段）
    // 则不执行优化，等待初始化完成
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0     = Utility::R2ypr(last_R0);
        origin_P0     = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(
        Quaterniond(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4], para_Pose[0][5])
            .toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff =
            Rs[0] * Quaterniond(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4], para_Pose[0][5])
                        .toRotationMatrix()
                        .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Rs[i] = rot_diff *
                Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5])
                    .normalized()
                    .toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) +
                origin_P0;

        Vs[i] =
            rot_diff * Vector3d(para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0], para_Ex_Pose[i][1], para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6], para_Ex_Pose[i][3], para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5])
                     .toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_ERROR("VINS little feature %d!", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_ERROR("VINS big IMU acc bias estimation %f, restart estimator!",
                  Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_ERROR("VINS big IMU gyr bias estimation %f, restart estimator!",
                  Bgs[WINDOW_SIZE].norm());
        return true;
    }
    if (Vs[WINDOW_SIZE].norm() > 30.0)
    {
        ROS_ERROR("VINS big speed %f, restart estimator!", Vs[WINDOW_SIZE].norm());
        return true;
    }
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5.0)
    {
        ROS_ERROR("VINS big translation, restart estimator!");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_ERROR("VINS big z translation, restart estimator!");
        return true;
    }
    Matrix3d    tmp_R   = Rs[WINDOW_SIZE];
    Matrix3d    delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double      delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / M_PI * 180.0;
    if (delta_angle > 50)
    {
        ROS_ERROR("VINS big delta_angle, moving too fast!");
        //return true;
    }
    return false;
}

/**
 * [功能描述]：VINS系统的核心优化函数，执行基于滑动窗口的Bundle Adjustment
 * 
 * 优化变量：
 * - 位姿状态：位置、旋转、速度、IMU偏差
 * - 相机外参：相机-IMU的旋转和平移
 * - 特征点深度：逆深度参数化
 * - 时间偏移：相机-IMU时间同步参数
 * 
 * 约束项：
 * - 边缘化约束：历史信息的先验约束
 * - IMU预积分约束：相邻帧间的运动约束
 * - 视觉重投影约束：特征点观测约束
 */
void Estimator::optimization()
{
    // ########################################
    // ##### 第1步：构建Ceres优化问题 #####
    // ########################################
    
    ceres::Problem       problem;        // Ceres优化问题对象
    ceres::LossFunction *loss_function;  // 鲁棒损失函数
    
    // 选择Cauchy损失函数，对外点具有鲁棒性
    // Cauchy损失函数比Huber损失函数对外点更加鲁棒
    loss_function = new ceres::CauchyLoss(1.0);

    // ########################################
    // ##### 第2步：添加位姿和速度偏差参数块 #####
    // ########################################
    
    // 为滑动窗口内的每一帧添加位姿和速度偏差参数
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 添加位姿参数块（7维：3位置 + 4四元数）
        // 使用自定义的位姿局部参数化，处理四元数的过参数化问题
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        
        // 添加速度和偏差参数块（9维：3速度 + 3加速度计偏差 + 3陀螺仪偏差）
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    // ########################################
    // ##### 第3步：添加相机外参参数块 #####
    // ########################################
    
    // 为每个相机添加外参参数块
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        // 外参也是位姿形式（旋转+平移），需要局部参数化
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        
        // 根据配置决定是否估计外参
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            // 外参固定，不参与优化
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    
    // ########################################
    // ##### 第4步：添加时间偏移参数块 #####
    // ########################################
    
    // 如果需要估计相机-IMU时间同步偏差
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);  // 1维时间偏移参数
        // 可选择固定时间偏移：problem.SetParameterBlockConstant(para_Td[0]);
    }

    // 将Eigen格式的状态变量转换为Ceres优化所需的double数组格式
    vector2double();

    // ########################################
    // ##### 第5步：添加边缘化约束 #####
    // ########################################
    
    // 如果存在上一次边缘化的先验信息
    if (last_marginalization_info)
    {
        // 构建边缘化因子，包含历史帧的先验约束信息
        // 这些约束来自于之前被边缘化掉的帧和特征点
        MarginalizationFactor *marginalization_factor =
            new MarginalizationFactor(last_marginalization_info);
        
        // 添加边缘化残差块到优化问题中
        // NULL表示不使用损失函数（边缘化约束通常是可信的）
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    // ########################################
    // ##### 第6步：添加IMU预积分约束 #####
    // ########################################
    
    // 为滑动窗口内相邻帧对添加IMU预积分约束
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;  // 下一帧索引
        
        // 如果预积分时间过长（>10秒），跳过该约束
        // 长时间预积分容易累积误差，不利于优化
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        
        // 创建IMU预积分因子
        IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
        
        // 添加IMU预积分残差块
        // 约束相邻两帧的位姿和速度偏差状态
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], 
                                 para_Pose[j], para_SpeedBias[j]);
    }

    // ########################################
    // ##### 第7步：添加视觉重投影约束 #####
    // ########################################
    
    int f_m_cnt       = 0;   // 特征点约束计数器
    int feature_index = -1;  // 特征点索引
    
    // 遍历所有管理的特征点
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();  // 更新观测次数
        
        // 筛选有效特征点：
        // 1. 至少被2帧观测到
        // 2. 起始帧不能太靠近窗口末尾（确保有足够的观测）
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++feature_index;  // 有效特征点索引递增

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;  // 帧索引
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;  // 首次观测的归一化坐标

        // 为该特征点的所有观测添加重投影约束
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)  // 跳过首次观测（作为参考）
                continue;
            
            Vector3d pts_j = it_per_frame.point;  // 当前观测的归一化坐标
            
            if (ESTIMATE_TD)
            {
                // ===== 考虑时间偏移的重投影因子 =====
                ProjectionTdFactor *f_td = new ProjectionTdFactor(
                    pts_i, pts_j, 
                    it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,  // 光流速度
                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,      // 时间偏移
                    it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());    // 像素坐标
                
                // 添加时间偏移重投影残差块
                problem.AddResidualBlock(f_td, loss_function, 
                                         para_Pose[imu_i], para_Pose[imu_j],
                                         para_Ex_Pose[0], para_Feature[feature_index], 
                                         para_Td[0]);

                // 如果特征点深度来自激光雷达，固定该深度参数
                if (it_per_id.lidar_depth_flag == true)
                    problem.SetParameterBlockConstant(para_Feature[feature_index]);
            }
            else
            {
                // ===== 标准重投影因子 =====
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                
                // 添加标准重投影残差块
                problem.AddResidualBlock(f, loss_function, 
                                         para_Pose[imu_i], para_Pose[imu_j],
                                         para_Ex_Pose[0], para_Feature[feature_index]);

                // 如果特征点深度来自激光雷达，固定该深度参数
                if (it_per_id.lidar_depth_flag == true)
                    problem.SetParameterBlockConstant(para_Feature[feature_index]);
            }
            f_m_cnt++;  // 重投影约束计数递增
        }
    }

    // ########################################
    // ##### 第8步：配置和执行优化求解 #####
    // ########################################
    
    ceres::Solver::Options options;
    
    // 线性求解器配置：DENSE_SCHUR适合Bundle Adjustment问题
    options.linear_solver_type = ceres::DENSE_SCHUR;
    
    // 信赖域策略：DOGLEG在视觉SLAM中表现良好
    options.trust_region_strategy_type = ceres::DOGLEG;
    
    // 最大迭代次数
    options.max_num_iterations = NUM_ITERATIONS;
    
    // 根据边缘化策略调整求解时间限制
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;  // 边缘化旧帧时间更紧
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;

    // 执行非线性优化求解
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 将优化结果从double数组转换回Eigen格式
    double2vector();

    // ########################################
    // ##### 第9步：边缘化处理 #####
    // ########################################
    
    if (marginalization_flag == MARGIN_OLD)
    {
        // ===== 边缘化最旧帧的处理 =====
        
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        // 处理上一次的边缘化信息
        if (last_marginalization_info)
        {
            vector<int> drop_set;  // 要丢弃的参数块索引
            
            // 找出需要边缘化掉的参数块（最旧帧的位姿和速度偏差）
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            
            // 构建新的边缘化因子
            MarginalizationFactor *marginalization_factor =
                new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        // 添加与最旧帧相关的IMU预积分约束
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info =
                    new ResidualBlockInfo(imu_factor, NULL,
                                          vector<double *>{para_Pose[0], para_SpeedBias[0],
                                                           para_Pose[1], para_SpeedBias[1]},
                                          vector<int>{0, 1});  // 边缘化前两个参数块
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        // 添加与最旧帧相关的视觉约束
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)  // 只处理从最旧帧开始观测的特征点
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(
                            pts_i, pts_j, it_per_id.feature_per_frame[0].velocity,
                            it_per_frame.velocity, it_per_id.feature_per_frame[0].cur_td,
                            it_per_frame.cur_td, it_per_id.feature_per_frame[0].uv.y(),
                            it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                            f_td, loss_function,
                            vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0],
                                             para_Feature[feature_index], para_Td[0]},
                            vector<int>{0, 3});  // 边缘化位姿和特征点深度
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                            f, loss_function,
                            vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0],
                                             para_Feature[feature_index]},
                            vector<int>{0, 3});  // 边缘化位姿和特征点深度
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        // ===== 执行边缘化计算 =====
        TicToc t_pre_margin;
        marginalization_info->preMarginalize();  // 预边缘化：准备雅可比矩阵
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();     // 正式边缘化：Schur消元
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        // ===== 更新参数块地址映射 =====
        // 由于滑动窗口向前移动，需要更新参数块的地址映射关系
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])]      = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        
        // 获取边缘化后的参数块列表
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        // 更新边缘化信息
        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info             = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else
    {
        // ===== 边缘化第二新帧的处理 =====
        
        // 检查是否需要边缘化第二新帧
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks),
                       std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                // 找出需要边缘化的第二新帧位姿参数
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] !=
                               para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                
                // 构建新的边缘化因子
                MarginalizationFactor *marginalization_factor =
                    new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                    marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            // 执行边缘化
            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            // 更新地址映射（第二新帧边缘化的情况）
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;  // 跳过被边缘化的第二新帧
                else if (i == WINDOW_SIZE)
                {
                    // 最新帧移动到第二新帧位置
                    addr_shift[reinterpret_cast<long>(para_Pose[i])]      = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    // 其他帧保持不变
                    addr_shift[reinterpret_cast<long>(para_Pose[i])]      = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }

            vector<double *> parameter_blocks =
                marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info             = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
}

/**
 * [功能描述]：管理VINS系统的滑动窗口，根据边缘化策略更新窗口内的帧数据
 * 
 * 边缘化策略：
 * - MARGIN_OLD: 边缘化最旧帧，所有数据向前滑动一位
 * - MARGIN_SECOND_NEW: 边缘化第二新帧，保留最新和最旧帧
 * 
 * 目的：维护固定大小的优化窗口，平衡计算效率和估计精度
 */
void Estimator::slideWindow()
{
    TicToc t_margin; // 滑动窗口操作计时器

    // ########################################
    // ##### 情况1：边缘化最旧帧 (MARGIN_OLD) #####
    // ########################################
    if (marginalization_flag == MARGIN_OLD)
    {
        // 保存即将被边缘化的最旧帧信息，用于边缘化计算
        double t_0  = Headers[0].stamp.toSec(); // 最旧帧时间戳
        back_R0     = Rs[0];                    // 最旧帧旋转矩阵
        back_P0     = Ps[0];                    // 最旧帧位置向量
        
        // 只有在滑动窗口已满时才执行滑动操作
        if (frame_count == WINDOW_SIZE)
        {
            // ===== 第1步：滑动窗口内所有状态变量 =====
            // 将索引1~WINDOW_SIZE的数据向前移动到索引0~WINDOW_SIZE-1
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                // 滑动旋转矩阵
                Rs[i].swap(Rs[i + 1]);

                // 滑动IMU预积分对象
                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                // 滑动IMU原始数据缓冲区
                dt_buf[i].swap(dt_buf[i + 1]);                      // 时间间隔
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]); // 线性加速度
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);       // 角速度

                // 滑动其他状态变量
                Headers[i] = Headers[i + 1];  // 消息头
                Ps[i].swap(Ps[i + 1]);        // 位置
                Vs[i].swap(Vs[i + 1]);        // 速度
                Bas[i].swap(Bas[i + 1]);      // 加速度计偏差
                Bgs[i].swap(Bgs[i + 1]);      // 陀螺仪偏差
            }
            
            // ===== 第2步：处理窗口末尾的新位置 =====
            // 将最新帧的数据复制到窗口末尾，为接收下一帧做准备
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE]      = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE]      = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE]      = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE]     = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE]     = Bgs[WINDOW_SIZE - 1];

            // ===== 第3步：重新初始化窗口末尾的IMU预积分 =====
            // 删除旧的预积分对象，避免内存泄漏
            delete pre_integrations[WINDOW_SIZE];
            
            // 创建新的预积分对象，使用当前的偏差估计值作为初始值
            pre_integrations[WINDOW_SIZE] =
                new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            // 清空窗口末尾的IMU数据缓冲区，准备接收新数据
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            // ===== 第4步：清理历史图像帧数据 =====
            // 这是内存管理的重要步骤，防止历史数据无限增长
            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                
                // 找到即将被边缘化的帧
                it_0 = all_image_frame.find(t_0);
                
                // 释放该帧的预积分内存
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

                // 清理所有比被边缘化帧更早的历史帧
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                // 从容器中移除这些历史帧
                all_image_frame.erase(all_image_frame.begin(), it_0); // 删除更早的帧
                all_image_frame.erase(t_0);                           // 删除被边缘化的帧
            }
            
            // ===== 第5步：执行特征点的滑动窗口操作 =====
            // 更新特征管理器中的帧索引，移除不再被观测的特征点
            slideWindowOld();
        }
    }
    else
    {
        // ########################################
        // ##### 情况2：边缘化第二新帧 (MARGIN_SECOND_NEW) #####
        // ########################################
        
        // 只有在滑动窗口已满时才执行操作
        if (frame_count == WINDOW_SIZE)
        {
            // ===== 第1步：将最新帧的IMU数据合并到第二新帧 =====
            // 这是因为要删除第二新帧，需要将其IMU数据转移保存
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                // 提取最新帧的IMU数据
                double   tmp_dt                  = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity    = angular_velocity_buf[frame_count][i];

                // 将这些数据添加到第二新帧的预积分中
                // frame_count-1 就是第二新帧的索引
                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration,
                                                             tmp_angular_velocity);

                // 同时更新IMU原始数据缓冲区
                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            // ===== 第2步：将最新帧的状态覆盖第二新帧 =====
            // 实际上是保留最新帧，丢弃第二新帧
            Headers[frame_count - 1] = Headers[frame_count];  // 消息头
            Ps[frame_count - 1]      = Ps[frame_count];       // 位置
            Vs[frame_count - 1]      = Vs[frame_count];       // 速度
            Rs[frame_count - 1]      = Rs[frame_count];       // 旋转
            Bas[frame_count - 1]     = Bas[frame_count];      // 加速度计偏差
            Bgs[frame_count - 1]     = Bgs[frame_count];      // 陀螺仪偏差

            // ===== 第3步：重新初始化窗口末尾的预积分 =====
            // 为下一帧的到来做准备
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] =
                new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            // 清空窗口末尾的IMU数据缓冲区
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            // ===== 第4步：执行特征点的滑动窗口操作 =====
            // 更新特征管理器，处理第二新帧被边缘化的情况
            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}