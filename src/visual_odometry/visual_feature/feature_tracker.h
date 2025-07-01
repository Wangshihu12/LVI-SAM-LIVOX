#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camera_models/CameraFactory.h"
#include "camera_models/CataCamera.h"
#include "camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();

    void readImage(const cv::Mat &_img, double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat               mask; // 特征点检测掩码
    cv::Mat               fisheye_mask; // 鱼眼相机掩码
    cv::Mat               prev_img, cur_img, forw_img; // 上一帧、当前帧、下一帧图像
    vector<cv::Point2f>   n_pts; // 新检测的特征点
    vector<cv::Point2f>   prev_pts, cur_pts, forw_pts; // 上一帧、当前帧、下一帧特征点
    vector<cv::Point2f>   prev_un_pts, cur_un_pts; // 上一帧、当前帧未畸变特征点
    vector<cv::Point2f>   pts_velocity; // 特征点光流速度
    vector<int>           ids; // 特征点全局唯一ID
    vector<int>           track_cnt; // 特征点跟踪次数计数
    map<int, cv::Point2f> cur_un_pts_map; // 当前帧特征点ID到归一化坐标的映射, key: 特征点ID, value: 对应的归一化相机坐标
    map<int, cv::Point2f> prev_un_pts_map; // 上一帧特征点ID到归一化坐标的映射, key: 特征点ID, value: 对应的归一化相机坐标
    camodocal::CameraPtr  m_camera; // 相机模型对象
    double                cur_time; // 当前帧时间戳
    double                prev_time; // 前一帧时间戳

    static int n_id; // 全局特征点ID计数器（静态成员变量）
};

class DepthRegister
{
public:
    ros::NodeHandle n;
    // publisher for visualization
    ros::Publisher pub_depth_feature;
    ros::Publisher pub_depth_image;
    ros::Publisher pub_depth_cloud;

    tf::TransformListener listener;
    tf::StampedTransform  transform;

    const int                 num_bins = 360;
    vector<vector<PointType>> pointsArray;

    DepthRegister(ros::NodeHandle n_in) : n(n_in)
    {
        // messages for RVIZ visualization
        pub_depth_feature =
            n.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/vins/depth/depth_feature", 5);
        pub_depth_image =
            n.advertise<sensor_msgs::Image>(PROJECT_NAME + "/vins/depth/depth_image", 5);
        pub_depth_cloud =
            n.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/vins/depth/depth_cloud", 5);

        pointsArray.resize(num_bins);
        for (int i = 0; i < num_bins; ++i)
            pointsArray[i].resize(num_bins);
    }

    /**
     * [功能描述]：从激光雷达点云中为视觉特征点获取深度信息
     * 
     * @param stamp_cur：当前图像时间戳
     * @param imageCur：当前图像
     * @param depthCloud：激光雷达点云（世界坐标系）
     * @param camera_model：相机模型
     * @param features_2d：视觉特征点的归一化坐标
     * @return：返回每个特征点对应的深度值
     * 
     * 核心思想：将激光点云和视觉特征点都投影到单位球面上，通过球面邻近搜索为特征点分配深度
     */
    sensor_msgs::ChannelFloat32 get_depth(const ros::Time &stamp_cur, const cv::Mat &imageCur,
                                        const pcl::PointCloud<PointType>::Ptr &depthCloud,
                                        const camodocal::CameraPtr            &camera_model,
                                        const vector<geometry_msgs::Point32>  &features_2d)
    {
        // ########################################
        // ##### 第0.1步：初始化返回结果 #####
        // ########################################
        
        sensor_msgs::ChannelFloat32 depth_of_point;
        depth_of_point.name = "depth";  // 通道名称
        // 初始化所有特征点深度为-1（表示无深度信息）
        depth_of_point.values.resize(features_2d.size(), -1);

        // ########################################
        // ##### 第0.2步：检查点云数据有效性 #####
        // ########################################
        
        // 如果激光雷达点云为空，直接返回无深度信息
        if (depthCloud->size() == 0)
            return depth_of_point;

        // ########################################
        // ##### 第0.3步：获取相机位姿变换 #####
        // ########################################
        
        // 查找当前时刻世界坐标系到相机坐标系的变换 T_W_C
        try
        {
            // 等待并获取TF变换：从世界坐标系到相机坐标系
            listener.waitForTransform("vins_world", "vins_cam_ros", stamp_cur, ros::Duration(0.01));
            listener.lookupTransform("vins_world", "vins_cam_ros", stamp_cur, transform);
        }
        catch (tf::TransformException ex)
        {
            // 如果获取变换失败，返回无深度信息
            // ROS_ERROR("image no tf");
            return depth_of_point;
        }

        // 从TF变换中提取位姿参数
        double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
        xCur = transform.getOrigin().x();     // 位置x
        yCur = transform.getOrigin().y();     // 位置y  
        zCur = transform.getOrigin().z();     // 位置z
        tf::Matrix3x3 m(transform.getRotation());
        m.getRPY(rollCur, pitchCur, yawCur);  // 提取欧拉角

        // 构建变换矩阵
        Eigen::Affine3f transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

        // ########################################
        // ##### 第0.4步：点云坐标系变换 #####
        // ########################################
        
        // 将点云从世界坐标系转换到相机坐标系（实际是IMU坐标系）
        pcl::PointCloud<PointType>::Ptr depth_cloud_local(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*depthCloud, *depth_cloud_local, transNow.inverse());

        // ########################################
        // ##### 第0.5步：特征点投影到单位球面 #####
        // ########################################
        
        // 将归一化的2D特征点投影到单位球面上
        pcl::PointCloud<PointType>::Ptr features_3d_sphere(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)features_2d.size(); ++i)
        {
            // 构建特征点的3D方向向量（归一化坐标，z=1）
            Eigen::Vector3f feature_cur(features_2d[i].x, features_2d[i].y, features_2d[i].z);  // z始终等于1
            
            // 归一化到单位球面：将向量长度标准化为1
            feature_cur.normalize();
            
            // 转换为PCL点格式
            PointType p;
            p.x         = feature_cur(0);
            p.y         = feature_cur(1);
            p.z         = feature_cur(2);
            p.intensity = -1;  // intensity字段将用于保存深度值
            features_3d_sphere->push_back(p);
        }

        // ########################################
        // ##### 第3步：构建距离图像进行点云降采样 #####
        // ########################################
        
        // 设置角度分辨率：180度范围划分为num_bins个区间
        float bin_res = 180.0 / (float)num_bins;  // 目前只覆盖激光雷达前方空间(-90° ~ 90°)
        
        // 创建距离图像：每个像素存储该方向上最近点的距离
        cv::Mat rangeImage = cv::Mat(num_bins, num_bins, CV_32F, cv::Scalar::all(FLT_MAX));

        // 遍历本地坐标系下的所有激光点
        for (int i = 0; i < (int)depth_cloud_local->size(); ++i)
        {
            PointType p = depth_cloud_local->points[i];
            
            // 过滤不在相机视野内的点
            // p.z < 0: 点在相机后方
            // abs(p.y/p.z) > 10: 点的俯仰角过大
            // abs(p.x/p.z) > 10: 点的方位角过大
            if (p.z < 0 || abs(p.y / p.z) > 10 || abs(p.x / p.z) > 10)
                continue;
            
            // ===== 计算点在距离图像中的行索引 =====
            // 俯仰角计算：从底部到顶部，0° ~ 180°
            float row_angle = atan2(-p.y, sqrt(p.x * p.x + p.z * p.z)) * 180.0 / M_PI + 90.0;
            int row_id = round(row_angle / bin_res);
            
            // ===== 计算点在距离图像中的列索引 =====  
            // 方位角计算：从左到右，0° ~ 360°
            float col_angle = atan2(p.z, -p.x) * 180.0 / M_PI;
            int   col_id    = round(col_angle / bin_res);
            
            // 检查索引是否越界
            if (row_id < 0 || row_id >= num_bins || col_id < 0 || col_id >= num_bins)
                continue;
            
            // ===== 距离图像更新策略：保留最近的点 =====
            float dist = pointDistance(p);  // 计算点到原点的距离
            if (dist < rangeImage.at<float>(row_id, col_id))
            {
                rangeImage.at<float>(row_id, col_id) = dist;      // 更新最近距离
                pointsArray[row_id][col_id]          = p;         // 保存对应的点
            }
        }

        // ########################################
        // ##### 第4步：从距离图像提取降采样点云 #####
        // ########################################
        
        pcl::PointCloud<PointType>::Ptr depth_cloud_local_filter2(new pcl::PointCloud<PointType>());
        // 遍历距离图像，提取有效点
        for (int i = 0; i < num_bins; ++i)
        {
            for (int j = 0; j < num_bins; ++j)
            {
                // 如果该像素有有效距离值，添加对应的点
                if (rangeImage.at<float>(i, j) != FLT_MAX)
                    depth_cloud_local_filter2->push_back(pointsArray[i][j]);
            }
        }
        *depth_cloud_local = *depth_cloud_local_filter2;
        
        // 发布降采样后的点云用于可视化
        publishCloud(&pub_depth_cloud, depth_cloud_local, stamp_cur, "vins_cam_ros");

        // ########################################
        // ##### 第5步：激光点云投影到单位球面 #####
        // ########################################
        
        pcl::PointCloud<PointType>::Ptr depth_cloud_unit_sphere(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)depth_cloud_local->size(); ++i)
        {
            PointType p     = depth_cloud_local->points[i];
            float     range = pointDistance(p);  // 计算点到原点的距离
            
            // 归一化到单位球面
            p.x /= range;
            p.y /= range;
            p.z /= range;
            p.intensity = range;  // 将原始距离保存在intensity字段中
            depth_cloud_unit_sphere->push_back(p);
        }
        
        // 如果球面点云太少，返回无深度信息
        if (depth_cloud_unit_sphere->size() < 10)
            return depth_of_point;

        // ########################################
        // ##### 第6步：构建KD树用于邻近搜索 #####
        // ########################################
        
        // 使用球面激光点云构建KD树，用于快速邻近点搜索
        pcl::KdTreeFLANN<PointType>::Ptr kdtree(new pcl::KdTreeFLANN<PointType>());
        kdtree->setInputCloud(depth_cloud_unit_sphere);

        // ########################################
        // ##### 第7步：使用KD树为特征点查找深度 #####
        // ########################################
        
        vector<int>   pointSearchInd;      // 搜索到的邻近点索引
        vector<float> pointSearchSqDis;    // 搜索到的邻近点距离平方
        
        // 设置搜索距离阈值：基于角度分辨率计算
        float dist_sq_threshold = pow(sin(bin_res / 180.0 * M_PI) * 5.0, 2);
        
        // 为每个特征点搜索邻近的激光点
        for (int i = 0; i < (int)features_3d_sphere->size(); ++i)
        {
            // 搜索3个最近邻点
            kdtree->nearestKSearch(features_3d_sphere->points[i], 3, pointSearchInd, pointSearchSqDis);
            
            // 检查是否找到3个邻近点且距离在阈值内
            if (pointSearchInd.size() == 3 && pointSearchSqDis[2] < dist_sq_threshold)
            {
                // ===== 恢复3个邻近点的3D坐标 =====
                float           r1 = depth_cloud_unit_sphere->points[pointSearchInd[0]].intensity;
                Eigen::Vector3f A(depth_cloud_unit_sphere->points[pointSearchInd[0]].x * r1,
                                depth_cloud_unit_sphere->points[pointSearchInd[0]].y * r1,
                                depth_cloud_unit_sphere->points[pointSearchInd[0]].z * r1);

                float           r2 = depth_cloud_unit_sphere->points[pointSearchInd[1]].intensity;
                Eigen::Vector3f B(depth_cloud_unit_sphere->points[pointSearchInd[1]].x * r2,
                                depth_cloud_unit_sphere->points[pointSearchInd[1]].y * r2,
                                depth_cloud_unit_sphere->points[pointSearchInd[1]].z * r2);

                float           r3 = depth_cloud_unit_sphere->points[pointSearchInd[2]].intensity;
                Eigen::Vector3f C(depth_cloud_unit_sphere->points[pointSearchInd[2]].x * r3,
                                depth_cloud_unit_sphere->points[pointSearchInd[2]].y * r3,
                                depth_cloud_unit_sphere->points[pointSearchInd[2]].z * r3);

                // ===== 计算射线与三角形平面的交点 =====
                // 特征点的单位方向向量
                Eigen::Vector3f V(features_3d_sphere->points[i].x, features_3d_sphere->points[i].y,
                                features_3d_sphere->points[i].z);

                // 计算三角形ABC的法向量
                Eigen::Vector3f N = (A - B).cross(B - C);
                
                // 使用射线-平面交点公式计算深度
                // 平面方程: N·P = N·A  
                // 射线方程: P = s*V
                // 交点: s = (N·A) / (N·V)
                float s = (N(0) * A(0) + N(1) * A(1) + N(2) * A(2)) /
                        (N(0) * V(0) + N(1) * V(1) + N(2) * V(2));

                // ===== 深度值合理性检查 =====
                float min_depth = min(r1, min(r2, r3));  // 三个邻近点的最小深度
                float max_depth = max(r1, max(r2, r3));  // 三个邻近点的最大深度
                
                // 检查深度变化是否过大或深度值是否合理
                if (max_depth - min_depth > 2 || s <= 0.5)
                {
                    continue;  // 跳过不合理的深度值
                }
                else if (s - max_depth > 0)
                {
                    s = max_depth;  // 限制在最大深度
                }
                else if (s - min_depth < 0)
                {
                    s = min_depth;  // 限制在最小深度
                }
                
                // ===== 更新特征点的3D坐标和深度 =====
                // 将特征点从单位球面恢复到笛卡尔坐标系
                features_3d_sphere->points[i].x *= s;
                features_3d_sphere->points[i].y *= s;
                features_3d_sphere->points[i].z *= s;
                
                // 保存深度值：这里的深度是z坐标（激光雷达的x轴对应相机的z轴）
                features_3d_sphere->points[i].intensity = features_3d_sphere->points[i].z;
            }
        }

        // 发布带有深度信息的特征点云用于可视化
        publishCloud(&pub_depth_feature, features_3d_sphere, stamp_cur, "vins_cam_ros");

        // ########################################
        // ##### 第8步：更新返回结果 #####
        // ########################################
        
        // 将计算得到的深度值更新到返回结果中
        for (int i = 0; i < (int)features_3d_sphere->size(); ++i)
        {
            // 只保留深度大于3米的有效深度值
            if (features_3d_sphere->points[i].intensity > 3.0)
                depth_of_point.values[i] = features_3d_sphere->points[i].intensity;
        }

        // ########################################
        // ##### 第9步：可视化处理（可选） #####
        // ########################################
        
        // 如果有订阅者需要深度图像可视化
        if (pub_depth_image.getNumSubscribers() != 0)
        {
            vector<cv::Point2f> points_2d;      // 2D投影点
            vector<float>       points_distance; // 对应的距离值

            // 将3D激光点投影到图像平面
            for (int i = 0; i < (int)depth_cloud_local->size(); ++i)
            {
                // 3D点坐标
                Eigen::Vector3d p_3d(depth_cloud_local->points[i].x, depth_cloud_local->points[i].y,
                                    depth_cloud_local->points[i].z);
                Eigen::Vector2d p_2d;
                
                // 使用相机模型将3D点投影到图像平面
                camera_model->spaceToPlane(p_3d, p_2d);

                points_2d.push_back(cv::Point2f(p_2d(0), p_2d(1)));
                points_distance.push_back(pointDistance(depth_cloud_local->points[i]));
            }

            // 创建可视化图像
            cv::Mat showImage, circleImage;
            cv::cvtColor(imageCur, showImage, cv::COLOR_GRAY2RGB);  // 转换为彩色图像
            circleImage = showImage.clone();
            
            // 在图像上绘制彩色深度点
            for (int i = 0; i < (int)points_2d.size(); ++i)
            {
                float r, g, b;
                getColor(points_distance[i], 50.0, r, g, b);  // 根据距离获取颜色
                cv::circle(circleImage, points_2d[i], 0, cv::Scalar(r, g, b), 5);
            }
            
            // 混合原图像和深度点图像
            cv::addWeighted(showImage, 1.0, circleImage, 0.7, 0, showImage);

            // 发布可视化图像
            cv_bridge::CvImage bridge;
            bridge.image                             = showImage;
            bridge.encoding                          = "rgb8";
            sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
            imageShowPointer->header.stamp           = stamp_cur;
            pub_depth_image.publish(imageShowPointer);
        }

        return depth_of_point;
    }

    void getColor(float p, float np, float &r, float &g, float &b)
    {
        float inc = 6.0 / np;
        float x   = p * inc;
        r         = 0.0f;
        g         = 0.0f;
        b         = 0.0f;
        if ((0 <= x && x <= 1) || (5 <= x && x <= 6))
            r = 1.0f;
        else if (4 <= x && x <= 5)
            r = x - 4;
        else if (1 <= x && x <= 2)
            r = 1.0f - (x - 1);

        if (1 <= x && x <= 3)
            g = 1.0f;
        else if (0 <= x && x <= 1)
            g = x - 0;
        else if (3 <= x && x <= 4)
            g = 1.0f - (x - 3);

        if (3 <= x && x <= 5)
            b = 1.0f;
        else if (2 <= x && x <= 3)
            b = x - 2;
        else if (5 <= x && x <= 6)
            b = 1.0f - (x - 5);
        r *= 255.0;
        g *= 255.0;
        b *= 255.0;
    }
};