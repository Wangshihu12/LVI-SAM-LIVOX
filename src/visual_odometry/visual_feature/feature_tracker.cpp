#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

/**
 * [功能描述]：设置特征点检测掩码，确保特征点分布均匀且优先保留长期跟踪的稳定特征点
 * 
 * 掩码作用：
 * - 白色区域(255)：允许检测新特征点的区域
 * - 黑色区域(0)：禁止检测新特征点的区域
 * 
 * 策略：优先保留跟踪次数多的特征点，在其周围设置禁区避免新特征点过于密集
 */
void FeatureTracker::setMask()
{
    // ========== 第1步：初始化掩码图像 ==========
    if(FISHEYE)
        // 如果使用鱼眼相机，使用预加载的鱼眼掩码
        // 鱼眼掩码通常会屏蔽图像边缘的畸变严重区域
        mask = fisheye_mask.clone();
    else
        // 普通相机：创建全白掩码(255表示所有区域都可以检测特征点)
        // ROW: 图像高度, COL: 图像宽度, CV_8UC1: 8位单通道, cv::Scalar(255): 全白
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    
    // ========== 第2步：构建特征点优先级排序数据结构 ==========
    // 创建包含 <跟踪次数, <特征点坐标, 特征点ID>> 的向量
    // 用于按跟踪次数对特征点进行排序，优先保留长期跟踪的稳定特征点
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    // 遍历所有当前跟踪的特征点，构建排序数据
    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));
        // make_pair结构：
        // - track_cnt[i]: 第i个特征点的跟踪次数（优先级指标）
        // - forw_pts[i]: 第i个特征点在下一帧中的坐标
        // - ids[i]: 第i个特征点的唯一标识符

    // ========== 第3步：按跟踪次数降序排序 ==========
    // 使用lambda表达式自定义排序规则：跟踪次数多的特征点排在前面
    // 这样可以优先保留更稳定、跟踪时间更长的特征点
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first; // a.first > b.first 表示按跟踪次数降序排列
         });

    // ========== 第4步：清空原有特征点数据 ==========
    // 准备重新填充经过掩码筛选后的特征点数据
    forw_pts.clear();   // 清空特征点坐标容器
    ids.clear();        // 清空特征点ID容器  
    track_cnt.clear();  // 清空跟踪次数容器

    // ========== 第5步：根据掩码筛选并重建特征点列表 ==========
    // 按优先级顺序遍历排序后的特征点
    for (auto &it : cnt_pts_id)
    {
        // 检查当前特征点位置的掩码值
        // mask.at<uchar>() 获取指定坐标处的掩码值
        // it.second.first 是特征点坐标 cv::Point2f
        if (mask.at<uchar>(it.second.first) == 255)
        {
            // 掩码值为255（白色）表示该位置允许保留特征点
            
            // 将该特征点添加到筛选后的特征点列表中
            forw_pts.push_back(it.second.first);    // 添加特征点坐标
            ids.push_back(it.second.second);        // 添加特征点ID  
            track_cnt.push_back(it.first);          // 添加跟踪次数
            
            // 在掩码上以该特征点为圆心，MIN_DIST为半径画黑色实心圆
            // 参数说明：
            // - mask: 目标掩码图像
            // - it.second.first: 圆心坐标（特征点位置）
            // - MIN_DIST: 圆的半径（最小特征点间距）
            // - 0: 圆的颜色（黑色，表示禁区）
            // - -1: 线宽（-1表示填充实心圆）
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
            
            // 作用：在该特征点周围MIN_DIST范围内设置禁区
            // 确保后续检测的新特征点不会在此范围内，保持特征点分布的均匀性
        }
        // 如果掩码值为0（黑色），说明该位置在禁区内，丢弃该特征点
    }
}

/**
 * [功能描述]：将新检测到的特征点添加到跟踪系统中
 * 
 * 调用时机：在goodFeaturesToTrack检测到新特征点后调用
 * 作用：初始化新特征点的相关数据结构，纳入特征点跟踪管理系统
 * 
 * 数据结构说明：
 * - forw_pts: 存储特征点在下一帧中的像素坐标
 * - ids: 存储特征点的全局唯一标识符
 * - track_cnt: 存储每个特征点的跟踪次数（用于评估特征点质量）
 */
void FeatureTracker::addPoints()
{
    // ========== 遍历所有新检测到的特征点 ==========
    // n_pts: 通过cv::goodFeaturesToTrack检测到的新特征点容器
    // 这些是在当前帧中新发现的Shi-Tomasi角点
    for (auto &p : n_pts)
    {
        // ===== 第1步：添加特征点坐标 =====
        // 将新特征点的像素坐标添加到下一帧特征点容器中
        // p是cv::Point2f类型，包含(x,y)像素坐标
        forw_pts.push_back(p);
        
        // ===== 第2步：分配临时ID =====
        // 为新特征点分配临时ID(-1)
        // -1表示该特征点还未获得全局唯一ID
        // 真正的全局唯一ID将在后续的updateID()函数中分配
        // 这样设计是为了避免ID冲突，确保每个特征点都有唯一标识
        ids.push_back(-1);
        
        // ===== 第3步：初始化跟踪计数 =====
        // 将新特征点的跟踪次数初始化为1
        // 跟踪次数的意义：
        // - 1: 新检测到的特征点，刚开始跟踪
        // - >1: 已经跟踪多帧的稳定特征点
        // 跟踪次数越高，表示特征点越稳定，质量越好
        track_cnt.push_back(1);
    }
    
    // 函数执行后的状态：
    // - forw_pts, ids, track_cnt三个容器的大小保持一致
    // - 新特征点已经纳入跟踪系统，等待下一帧的光流跟踪
    // - 临时ID(-1)将在updateID()中被替换为全局唯一ID
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r; // 计时器，用于测量整个函数执行时间
    cur_time = _cur_time; // 保存当前图像的时间戳

    // ========== 图像预处理：直方图均衡化 ==========
    if (EQUALIZE)
    {
        // 使用CLAHE（对比度限制自适应直方图均衡化）增强图像对比度
        // 参数说明：
        // - 3.0: 对比度限制阈值，防止噪声放大
        // - cv::Size(8, 8): 网格大小，将图像分成8x8的小块分别处理
        // 优点：能够自适应地增强局部对比度，改善光照不均匀的影响
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c; // 计时器，测量CLAHE处理时间
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img; // 不进行直方图均衡化，直接使用原图

    // ========== 图像序列管理 ==========
    if (forw_img.empty())
    {
        // 第一帧图像：初始化所有图像缓存
        // prev_img: 前一帧图像
        // cur_img:  当前帧图像  
        // forw_img: 下一帧图像（在处理过程中作为目标帧）
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        // 非第一帧：更新下一帧图像
        forw_img = img;
    }

    // 清空待处理的特征点容器
    forw_pts.clear();

    // ========== 光流跟踪：跟踪现有特征点 ==========
    if (cur_pts.size() > 0)
    {
        TicToc t_o; // 光流跟踪计时器
        vector<uchar> status; // 跟踪状态：1表示成功跟踪，0表示跟踪失败
        vector<float> err;    // 跟踪误差

        // 使用Lucas-Kanade金字塔光流法跟踪特征点
        // 参数说明：
        // - cur_img, forw_img: 当前帧和下一帧图像
        // - cur_pts, forw_pts: 输入和输出特征点坐标
        // - status: 每个特征点的跟踪状态
        // - err: 每个特征点的跟踪误差
        // - cv::Size(21, 21): 搜索窗口大小
        // - 3: 金字塔层数
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        // 边界检查：移除跟踪到图像边界外的特征点
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0; // 标记为跟踪失败

        // 根据跟踪状态清理特征点相关数据
        // 只保留成功跟踪的特征点，移除跟踪失败的点
        reduceVector(prev_pts, status);   // 前一帧特征点坐标
        reduceVector(cur_pts, status);    // 当前帧特征点坐标
        reduceVector(forw_pts, status);   // 下一帧特征点坐标
        reduceVector(ids, status);        // 特征点唯一ID
        reduceVector(cur_un_pts, status); // 当前帧去畸变后的归一化坐标
        reduceVector(track_cnt, status);  // 每个特征点的跟踪次数计数

        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    // ========== 更新跟踪计数 ==========
    // 为所有成功跟踪的特征点增加跟踪次数
    // 跟踪次数越高表示该特征点越稳定，质量越好
    for (auto &n : track_cnt)
        n++;

    // ========== 特征点处理（仅在需要发布的帧中执行） ==========
    if (PUB_THIS_FRAME)
    {
        // ##### 第1步：基础矩阵筛选 #####
        // 使用RANSAC算法和基础矩阵剔除外点
        // 基于对极几何约束，移除不符合相机运动模型的特征点匹配
        rejectWithF();
        
        ROS_DEBUG("set mask begins");
        TicToc t_m; // 掩码设置计时器
        
        // ##### 第2步：设置特征点检测掩码 #####
        // 创建掩码图像，在已有特征点周围设置禁区
        // 避免新检测的特征点过于密集，确保特征点分布均匀
        setMask();
        
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t; // 特征检测计时器
        
        // ##### 第3步：检测新的特征点 #####
        // 计算还需要检测多少个新特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        
        if (n_max_cnt > 0)
        {
            // 调试检查：确保掩码的正确性
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            
            // 使用Shi-Tomasi角点检测器检测新特征点
            // 参数说明：
            // - forw_img: 输入图像
            // - n_pts: 输出的新特征点
            // - MAX_CNT - forw_pts.size(): 最大检测数量
            // - 0.01: 角点检测质量阈值（越小检测到的角点越多）
            // - MIN_DIST: 特征点之间的最小距离
            // - mask: 掩码图像，黑色区域不检测特征点
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear(); // 如果特征点数量已够，清空新检测结果

        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a; // 特征点添加计时器
        
        // ##### 第4步：添加新特征点 #####
        // 将新检测到的特征点添加到跟踪系统中
        // 为新特征点分配ID，初始化相关数据结构
        addPoints();
        
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

    // ========== 图像和特征点数据更新 ==========
    // 将当前处理结果向前推进一帧，为下次处理做准备
    prev_img = cur_img;       // 前一帧图像 ← 当前帧图像
    prev_pts = cur_pts;       // 前一帧特征点 ← 当前帧特征点
    prev_un_pts = cur_un_pts; // 前一帧去畸变坐标 ← 当前帧去畸变坐标
    cur_img = forw_img;       // 当前帧图像 ← 下一帧图像
    cur_pts = forw_pts;       // 当前帧特征点 ← 下一帧特征点

    // ##### 第5步：坐标去畸变处理 #####
    // 将像素坐标转换为去畸变后的归一化相机坐标
    // 这是为了消除镜头畸变的影响，得到理想的针孔相机模型坐标
    undistortedPoints();
    
    prev_time = cur_time; // 更新时间戳
}

/**
 * [功能描述]：使用基础矩阵和RANSAC算法剔除光流跟踪中的外点
 * 
 * 原理：基于对极几何约束，利用基础矩阵检测不符合相机运动模型的特征点匹配
 * 目的：提高特征点匹配的精度，去除跟踪错误和噪声点
 * 
 * 前提条件：至少需要8个特征点对才能计算基础矩阵
 */
void FeatureTracker::rejectWithF()
{
    // ========== 第1步：检查特征点数量 ==========
    // 基础矩阵计算需要最少8个对应点对（8点算法）
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f; // 基础矩阵计算计时器
        
        // ========== 第2步：坐标去畸变处理 ==========
        // 创建去畸变后的特征点坐标容器
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        
        // 对当前帧和下一帧的所有特征点进行去畸变处理
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p; // 临时3D坐标变量
            
            // ===== 处理当前帧特征点 =====
            // 将像素坐标转换为归一化相机坐标（去畸变）
            // liftProjective: 像素坐标 -> 归一化相机坐标（考虑畸变）
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            
            // 将归一化坐标重新投影到虚拟的理想相机平面
            // 使用统一的焦距FOCAL_LENGTH，消除不同相机内参的影响
            // 公式：u = fx * X/Z + cx, v = fy * Y/Z + cy
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            // ===== 处理下一帧特征点 =====
            // 同样的去畸变和重投影过程
            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        // ========== 第3步：使用RANSAC计算基础矩阵 ==========
        vector<uchar> status; // 内点/外点标记：1表示内点，0表示外点
        
        // 使用OpenCV的RANSAC算法计算基础矩阵
        // 参数说明：
        // - un_cur_pts, un_forw_pts: 去畸变后的特征点对
        // - cv::FM_RANSAC: 使用RANSAC算法
        // - F_THRESHOLD: 点到对极线的距离阈值（通常为1.0像素）
        // - 0.99: RANSAC的置信度（99%的概率找到正确模型）
        // - status: 输出每个点的内点/外点标记
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        
        // ========== 第4步：根据基础矩阵结果剔除外点 ==========
        int size_a = cur_pts.size(); // 记录筛选前的特征点数量
        
        // 根据status标记，只保留内点，移除外点
        // 对所有与特征点相关的数据结构进行同步更新
        reduceVector(prev_pts, status);   // 前一帧特征点坐标
        reduceVector(cur_pts, status);    // 当前帧特征点坐标
        reduceVector(forw_pts, status);   // 下一帧特征点坐标
        reduceVector(cur_un_pts, status); // 当前帧去畸变归一化坐标
        reduceVector(ids, status);        // 特征点唯一ID
        reduceVector(track_cnt, status);  // 特征点跟踪次数
        
        // ========== 第5步：输出筛选结果统计 ==========
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        // 输出格式：原始数量 -> 筛选后数量: 保留比例
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
    // 如果特征点数量不足8个，跳过基础矩阵筛选
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

/**
 * [功能描述]：将当前帧特征点的像素坐标转换为去畸变的归一化相机坐标，并计算特征点速度
 * 
 * 主要任务：
 * 1. 像素坐标 → 归一化相机坐标变换（消除畸变影响）
 * 2. 计算特征点在归一化坐标系下的运动速度
 * 3. 更新ID到坐标的映射表，便于快速查找
 * 
 * 输出结果：
 * - cur_un_pts: 当前帧归一化坐标
 * - cur_un_pts_map: ID到坐标的映射
 * - pts_velocity: 特征点速度向量
 */
void FeatureTracker::undistortedPoints()
{
    // ########################################
    // ##### 第1步：清空和初始化数据容器 #####
    // ########################################
    
    // 清空当前帧的归一化坐标容器
    // 准备存储新计算的去畸变归一化坐标
    cur_un_pts.clear();
    
    // 清空当前帧的ID到坐标映射表
    // 重新建立特征点ID与归一化坐标的对应关系
    cur_un_pts_map.clear();
    
    // 注释掉的OpenCV标准去畸变函数：
    // cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    // 这里使用更通用的相机模型方法，支持多种畸变类型

    // ########################################
    // ##### 第2步：像素坐标到归一化坐标变换 #####
    // ########################################
    
    // 遍历当前帧所有特征点，进行坐标变换
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        // ===== 提取当前特征点的像素坐标 =====
        // 将cv::Point2f格式转换为Eigen::Vector2d格式
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        
        // ===== 执行去畸变和归一化变换 =====
        Eigen::Vector3d b; // 3D射线向量（在相机坐标系中）
        
        // 使用相机模型的liftProjective函数：
        // 功能：像素坐标 → 3D射线方向（考虑畸变校正）
        // 输入：像素坐标 a(u,v)
        // 输出：对应的3D射线向量 b(X,Y,Z)
        // 该函数内部处理了：
        // 1. 去除相机内参影响
        // 2. 畸变校正（针孔、鱼眼等多种模型）
        // 3. 转换为单位球面上的射线方向
        m_camera->liftProjective(a, b);
        
        // ===== 归一化到z=1平面 =====
        // 将3D射线投影到z=1的归一化平面上
        // 公式：(X/Z, Y/Z, 1) → (X/Z, Y/Z)
        // 这是标准的归一化相机坐标，消除了焦距的影响
        cv::Point2f normalized_point(b.x() / b.z(), b.y() / b.z());
        
        // 添加到归一化坐标容器中
        cur_un_pts.push_back(normalized_point);
        
        // ===== 建立ID到坐标的映射关系 =====
        // 将特征点ID与其归一化坐标关联
        // 便于后续基于ID的快速坐标查找
        cur_un_pts_map.insert(make_pair(ids[i], normalized_point));
        
        // 调试输出（已注释）：
        // printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }

    // ########################################
    // ##### 第3步：计算特征点运动速度 #####
    // ########################################
    
    // 检查是否存在前一帧的归一化坐标数据
    if (!prev_un_pts_map.empty())
    {
        // ===== 计算时间间隔 =====
        // 当前帧与前一帧之间的时间差（秒）
        double dt = cur_time - prev_time;
        
        // 清空速度容器，准备计算新的速度
        pts_velocity.clear();
        
        // ===== 为每个特征点计算速度 =====
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            // 检查特征点是否有有效ID（-1表示新特征点，暂无ID）
            if (ids[i] != -1)
            {
                // 在前一帧映射表中查找相同ID的特征点
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                
                if (it != prev_un_pts_map.end())
                {
                    // ===== 找到匹配的前一帧位置，计算速度 =====
                    // 速度公式：v = (当前位置 - 前一位置) / 时间间隔
                    // 在归一化相机坐标系中计算，单位为 坐标单位/秒
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt; // x方向速度
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt; // y方向速度
                    
                    // 添加计算得到的速度向量
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                {
                    // ===== 前一帧中没有找到对应特征点 =====
                    // 可能原因：特征点是新出现的，或者前一帧跟踪失败
                    // 设置速度为零
                    pts_velocity.push_back(cv::Point2f(0, 0));
                }
            }
            else
            {
                // ===== 特征点ID无效（新检测的特征点） =====
                // 新特征点没有历史轨迹，无法计算速度
                // 设置速度为零
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        // ########################################
        // ##### 第4步：处理首帧情况 #####
        // ########################################
        
        // 如果前一帧映射表为空（说明是第一帧或重新初始化）
        // 所有特征点的速度都设为零
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    
    // ########################################
    // ##### 第5步：更新历史数据 #####
    // ########################################
    
    // 将当前帧的映射表保存为前一帧映射表
    // 为下一次速度计算做准备
    prev_un_pts_map = cur_un_pts_map;
}
