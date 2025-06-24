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

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
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

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
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

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
