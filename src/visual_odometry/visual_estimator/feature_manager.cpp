#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

/**
 * [功能描述]：统计特征管理器中有效特征点的数量
 * 
 * @return：满足条件的有效特征点数量
 * 
 * 有效特征点的定义：
 * 1. 至少被2帧观测到（可以进行三角化）
 * 2. 起始帧不能太靠近滑动窗口末尾（确保有足够观测）
 */
int FeatureManager::getFeatureCount()
{
    // ########################################
    // ##### 初始化计数器 #####
    // ########################################
    
    int cnt = 0;  // 有效特征点计数器

    // ########################################
    // ##### 遍历所有管理的特征点 #####
    // ########################################
    
    // 遍历特征管理器中的所有特征点对象
    for (auto &it : feature)
    {
        // ===== 更新特征点的观测次数统计 =====
        // used_num记录该特征点在多少帧中被观测到
        // feature_per_frame.size()返回该特征点的观测帧数
        it.used_num = it.feature_per_frame.size();

        // ########################################
        // ##### 检查特征点有效性条件 #####
        // ########################################
        
        // 条件1：it.used_num >= 2
        // 含义：特征点至少被2帧观测到
        // 原因：只有被多帧观测的特征点才能进行三角化，获得可靠的3D坐标
        //
        // 条件2：it.start_frame < WINDOW_SIZE - 2  
        // 含义：特征点的起始观测帧不能太靠近滑动窗口的末尾
        // 原因：
        // - WINDOW_SIZE是滑动窗口大小（通常为10）
        // - WINDOW_SIZE - 2 = 8，即特征点必须在第8帧之前开始被观测
        // - 这样可以确保特征点有足够的观测历史，提供稳定的约束
        // - 避免只在最新几帧中出现的"年轻"特征点，它们可能不够稳定
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;  // 满足条件的特征点计数器递增
        }
    }
    
    // 返回有效特征点的总数
    return cnt;
}


/**
 * [功能描述]：添加新帧的特征点观测并检查是否有足够视差
 * 
 * @param frame_count：当前帧索引
 * @param image：特征点数据，格式为 map<特征点ID, vector<相机ID和8维特征向量>>
 * @param td：相机-IMU时间偏移
 * @return：是否有足够视差（true: 视差足够，边缘化旧帧; false: 视差不足，边缘化第二新帧）
 * 
 * 核心逻辑：
 * 1. 管理特征点的生命周期（新建/更新观测）
 * 2. 处理激光雷达深度信息
 * 3. 计算平均视差决定边缘化策略
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &image, double td)
{
    // 调试输出：当前输入的特征点数量和已管理的特征点数量
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    
    // ########################################
    // ##### 初始化视差计算相关变量 #####
    // ########################################
    
    double parallax_sum = 0;    // 视差总和
    int parallax_num = 0;       // 参与视差计算的特征点数量
    last_track_num = 0;         // 上一帧成功跟踪的特征点数量

    // ########################################
    // ##### 第1步：遍历所有输入特征点 #####
    // ########################################
    
    // 遍历当前帧的所有特征点观测
    for (auto &id_pts : image)
    {
        // ===== 构建特征点帧观测对象 =====
        // 从8维特征向量中提取信息并创建FeaturePerFrame对象
        // id_pts.second[0].second: 8维特征向量 [归一化坐标(3) + 像素坐标(2) + 光流速度(2) + 深度(1)]
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        // ===== 查找特征点是否已存在 =====
        int feature_id = id_pts.first;  // 特征点的全局唯一ID
        
        // 在已管理的特征点列表中搜索当前ID
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {return it.feature_id == feature_id;});

        if (it == feature.end())
        {
            // ########################################
            // ##### 情况1：新特征点首次观测 #####
            // ########################################
            
            // 创建新的特征点对象
            // 参数：特征点ID, 起始帧索引, 初始深度估计
            feature.push_back(FeaturePerId(feature_id, frame_count, f_per_fra.depth));
            
            // 添加当前帧的观测到新创建的特征点中
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        else if (it->feature_id == feature_id)
        {
            // ########################################
            // ##### 情况2：已有特征点的新观测 #####
            // ########################################
            
            // 为已存在的特征点添加新的帧观测
            it->feature_per_frame.push_back(f_per_fra);
            
            // 更新成功跟踪的特征点计数
            last_track_num++;
            
            // ===== 激光雷达深度信息处理 =====
            // 有时特征点首次观测时没有深度信息，后续帧可能获得激光雷达深度
            // 注意：如果相机运动很快，用当前图像深度初始化特征点深度可能不够准确
            if (f_per_fra.depth > 0 && it->lidar_depth_flag == false)
            {
                // 更新特征点的深度估计
                it->estimated_depth = f_per_fra.depth;      // 设置估计深度
                it->lidar_depth_flag = true;                // 标记有激光雷达深度
                it->feature_per_frame[0].depth = f_per_fra.depth;  // 更新首次观测的深度
            }
        }
    }

    // ########################################
    // ##### 第2步：视差计算前的条件检查 #####
    // ########################################
    
    // 检查是否满足视差计算的基本条件：
    // 1. frame_count < 2: 前两帧没有足够的时间基线
    // 2. last_track_num < 20: 成功跟踪的特征点太少，视差不可靠
    if (frame_count < 2 || last_track_num < 20)
        return true;  // 条件不满足时，默认返回true（边缘化旧帧）

    // ########################################
    // ##### 第3步：计算平均视差 #####
    // ########################################
    
    // 遍历所有管理的特征点，计算视差
    for (auto &it_per_id : feature)
    {
        // ===== 筛选参与视差计算的特征点 =====
        // 条件1: it_per_id.start_frame <= frame_count - 2
        //        特征点至少在前第二帧就开始被观测
        // 条件2: it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1
        //        特征点至少被观测到前一帧
        // 总结：选择在前第二帧和前一帧之间都有观测的特征点
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            // 计算该特征点在前第二帧和前一帧之间的补偿视差
            // compensatedParallax2(): 考虑了旋转补偿的视差计算，更准确
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    // ########################################
    // ##### 第4步：视差判断和边缘化决策 #####
    // ########################################
    
    if (parallax_num == 0)
    {
        // 如果没有特征点参与视差计算，默认边缘化旧帧
        return true;
    }
    else
    {
        // ===== 计算并输出平均视差 =====
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        
        // ===== 视差判断和边缘化策略 =====
        // 平均视差 >= MIN_PARALLAX: 视差足够大
        //   - 说明相机运动明显，有足够的基线长度
        //   - 适合进行三角化和优化
        //   - 返回true: 边缘化最旧帧，保留当前帧作为新的关键帧
        //
        // 平均视差 < MIN_PARALLAX: 视差不足
        //   - 说明相机运动缓慢或静止
        //   - 当前帧信息量不够，不适合作为关键帧
        //   - 返回false: 边缘化第二新帧，保留时间跨度更大的帧组合
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

/**
 * [功能描述]：获取两帧之间的特征点对应关系
 * 
 * @param frame_count_l：左帧（参考帧）的索引
 * @param frame_count_r：右帧（目标帧）的索引
 * @return：特征点对应关系列表，每个pair包含同一特征点在两帧中的归一化坐标
 * 
 * 应用场景：
 * - 相对位姿估计（Essential Matrix）
 * - 基础矩阵计算（Fundamental Matrix）  
 * - 相机外参标定
 * - SFM初始化
 */
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    // ########################################
    // ##### 初始化对应关系容器 #####
    // ########################################
    
    // 存储特征点对应关系的容器
    // 每个pair的结构：<左帧观测, 右帧观测>
    // Vector3d格式：[归一化x, 归一化y, 1.0]
    vector<pair<Vector3d, Vector3d>> corres;

    // ########################################
    // ##### 遍历所有管理的特征点 #####
    // ########################################
    
    // 遍历特征管理器中的所有特征点
    for (auto &it : feature)
    {
        // ===== 检查特征点在两帧中是否都有观测 =====
        // 条件1: it.start_frame <= frame_count_l
        //        特征点在左帧（或之前）开始被观测
        // 条件2: it.endFrame() >= frame_count_r  
        //        特征点至少被观测到右帧
        // 结论：只有在两帧中都有观测的特征点才能建立对应关系
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            // ===== 初始化观测向量 =====
            Vector3d a = Vector3d::Zero();  // 左帧中的观测
            Vector3d b = Vector3d::Zero();  // 右帧中的观测
            
            // ===== 计算特征点在观测序列中的索引 =====
            // 由于特征点的观测存储在feature_per_frame数组中
            // 需要将全局帧索引转换为该特征点观测序列中的相对索引
            
            // 左帧在该特征点观测序列中的索引
            // 公式：相对索引 = 全局帧索引 - 特征点起始帧索引
            int idx_l = frame_count_l - it.start_frame;
            
            // 右帧在该特征点观测序列中的索引
            int idx_r = frame_count_r - it.start_frame;

            // ===== 提取特征点在两帧中的观测数据 =====
            // feature_per_frame[idx].point 存储的是归一化相机坐标 (x, y, 1)
            a = it.feature_per_frame[idx_l].point;  // 左帧观测
            b = it.feature_per_frame[idx_r].point;  // 右帧观测
            
            // ===== 构建对应关系并添加到结果中 =====
            // make_pair创建特征点对应关系：<左帧观测, 右帧观测>
            corres.push_back(make_pair(a, b));
        }
        // 如果特征点在两帧中没有完整观测，跳过该特征点
    }
    
    // 返回所有有效的特征点对应关系
    return corres;
}

/**
 * [功能描述]：根据优化结果更新特征点的深度估计
 * 
 * @param x：优化后的逆深度参数向量，每个元素对应一个有效特征点的逆深度值
 * 
 * 处理流程：
 * 1. 过滤掉不满足条件的特征点
 * 2. 将逆深度转换为真实深度
 * 3. 根据深度正负性标记求解状态
 */
void FeatureManager::setDepth(const VectorXd &x)
{
    // ########################################
    // ##### 初始化逆深度参数索引 #####
    // ########################################
    
    // 逆深度参数向量的索引计数器
    // 只有满足条件的特征点才会分配逆深度参数，因此需要单独计数
    int feature_index = -1;
    
    // ########################################
    // ##### 遍历所有管理的特征点 #####
    // ########################################
    
    for (auto &it_per_id : feature)
    {
        // ===== 第1步：更新观测次数统计 =====
        // 统计该特征点在多少帧中被观测到
        // used_num用于后续的有效性判断
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        
        // ===== 第2步：特征点有效性检查 =====
        // 跳过不满足以下条件的特征点：
        // 条件1：used_num >= 2 (至少被2帧观测，才能进行三角化)
        // 条件2：start_frame < WINDOW_SIZE - 2 (起始帧不能太靠近窗口末尾)
        // 这些条件确保特征点有足够的观测约束进行可靠的深度估计
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // ===== 第3步：逆深度转换为深度 =====
        // 递增索引，获取对应的逆深度参数
        // VINS使用逆深度参数化：rho = 1/depth，提高数值稳定性
        it_per_id.estimated_depth = 1.0 / x(++feature_index);

        // ===== 第4步：根据深度值设置求解标志 =====
        if (it_per_id.estimated_depth < 0)
        {
            // 深度为负：物理不合理，标记为求解失败
            // solve_flag = 2 表示深度估计异常，该特征点可能需要重新三角化
            it_per_id.solve_flag = 2;
        }
        else
        {
            // 深度为正：物理合理，标记为求解成功
            // solve_flag = 1 表示深度估计正常，可以用于后续优化
            it_per_id.solve_flag = 1;
        }
    }
}

/**
 * [功能描述]：移除深度求解失败的特征点，清理无效数据
 * 
 * 移除条件：solve_flag == 2（深度估计异常的特征点）
 * 
 * 技术要点：
 * - 使用双迭代器技术确保删除过程中迭代器的有效性
 * - 及时清理异常特征点，避免影响后续优化精度
 */
void FeatureManager::removeFailures()
{
    // ########################################
    // ##### 安全遍历并删除异常特征点 #####
    // ########################################
    
    // 使用双迭代器模式进行安全删除：
    // - it: 当前处理的特征点迭代器
    // - it_next: 下一个特征点迭代器（预先递增，防止删除时迭代器失效）
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        // ===== 第1步：预先递增下一个迭代器 =====
        // 在删除当前元素之前，先保存下一个有效位置
        // 这样即使当前迭代器因删除而失效，it_next仍然有效
        it_next++;
        
        // ===== 第2步：检查特征点求解状态 =====
        // solve_flag状态说明：
        // - 0: 未求解
        // - 1: 求解成功（深度为正值）
        // - 2: 求解失败（深度为负值或其他异常）
        if (it->solve_flag == 2)
        {
            // ===== 第3步：删除异常特征点 =====
            // 从特征管理器中彻底移除该特征点
            // 包括其完整的观测历史和相关数据
            feature.erase(it);
        }
    }
}

/**
 * [功能描述]：更新特征点深度并清除激光雷达深度标志，切换到纯视觉深度估计模式
 * 
 * @param x：优化后的逆深度参数向量
 * 
 * 与setDepth()的区别：
 * - setDepth()：正常更新深度，保持激光雷达标志不变
 * - clearDepth()：更新深度 + 强制清除激光雷达标志，转为纯视觉模式
 * 
 * 应用场景：系统重初始化、激光雷达数据异常、纯视觉模式切换
 */
void FeatureManager::clearDepth(const VectorXd &x)
{
    // ########################################
    // ##### 初始化逆深度参数索引 #####
    // ########################################
    
    // 逆深度参数向量的索引计数器
    // 只对满足条件的有效特征点分配逆深度参数
    int feature_index = -1;
    
    // ########################################
    // ##### 遍历所有管理的特征点 #####
    // ########################################
    
    for (auto &it_per_id : feature)
    {
        // ===== 第1步：更新观测次数统计 =====
        // 重新计算该特征点的观测帧数
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        
        // ===== 第2步：特征点有效性检查 =====
        // 筛选条件与setDepth()完全相同：
        // - used_num >= 2: 至少被2帧观测，可进行三角化
        // - start_frame < WINDOW_SIZE - 2: 起始帧位置合理，有足够观测历史
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // ===== 第3步：逆深度转换为深度 =====
        // 从优化结果中提取逆深度参数并转换为真实深度
        // 使用逆深度参数化提高数值稳定性：depth = 1/rho
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        
        // ===== 第4步：清除激光雷达深度标志 =====
        // 关键操作：强制设置激光雷达深度标志为false
        // 目的：
        // 1. 放弃激光雷达提供的深度信息
        // 2. 完全依赖视觉三角化的深度估计
        // 3. 统一深度来源，避免多源深度冲突
        it_per_id.lidar_depth_flag = false;
    }
}

/**
 * [功能描述]：提取所有有效特征点的逆深度参数向量，用于Ceres优化
 * 
 * @return：逆深度参数向量，每个元素对应一个有效特征点的逆深度值（1/depth）
 * 
 * 功能特点：
 * 1. 只包含满足条件的有效特征点
 * 2. 自动处理异常深度值，确保参数合理性
 * 3. 为非线性优化提供良好的初始值
 */
VectorXd FeatureManager::getDepthVector()
{
    // ########################################
    // ##### 第1步：创建逆深度参数向量 #####
    // ########################################
    
    // 创建向量存储逆深度参数，向量大小等于有效特征点数量
    // getFeatureCount()统计满足优化条件的特征点总数
    VectorXd dep_vec(getFeatureCount());
    
    // 逆深度参数索引计数器，对应dep_vec中的位置
    int feature_index = -1;
    
    // ########################################
    // ##### 第2步：遍历所有管理的特征点 #####
    // ########################################
    
    for (auto &it_per_id : feature)
    {
        // ===== 更新观测次数统计 =====
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        
        // ===== 特征点有效性检查 =====
        // 跳过不满足优化条件的特征点：
        // - used_num >= 2: 至少被2帧观测，支持三角化
        // - start_frame < WINDOW_SIZE - 2: 起始帧位置合理，有足够观测约束
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // ===== 第3步：逆深度参数设置 =====
        // 递增索引，处理下一个有效特征点
        ++feature_index;
        
        // 检查深度估计的有效性并设置逆深度参数
        if (it_per_id.estimated_depth > 0)
        {
            // 情况1：深度估计正常（正值）
            // 直接计算逆深度：rho = 1/depth
            // 逆深度参数化提供更好的数值稳定性和线性化特性
            dep_vec(feature_index) = 1.0 / it_per_id.estimated_depth;
        }
        else
        {
            // 情况2：深度估计异常（负值、零值或未初始化）
            // 使用默认初始深度的逆深度作为安全回退值
            // INIT_DEPTH通常设置为较大值（如5米），对应较小的逆深度
            // 注释说明：Ceres优化后的深度可能为负值，用默认值重新初始化
            dep_vec(feature_index) = 1.0 / INIT_DEPTH;
        }
    }
    
    // 返回构建完成的逆深度参数向量
    return dep_vec;
}

/**
 * [功能描述]：对没有深度信息的特征点进行三角化，从多帧观测中恢复3D位置
 * 
 * @param Ps[]：各帧IMU的位置向量
 * @param tic[]：相机相对IMU的平移向量  
 * @param ric[]：相机相对IMU的旋转矩阵
 * 
 * 算法原理：
 * 使用多视角几何的DLT（直接线性变换）方法，通过SVD求解线性方程组
 * 每个观测提供2个约束方程，多帧观测构成超定方程组
 */
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    // ########################################
    // ##### 遍历所有管理的特征点 #####
    // ########################################
    
    for (auto &it_per_id : feature)
    {
        // ===== 第1步：特征点有效性检查 =====
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        
        // 跳过不满足三角化条件的特征点：
        // - used_num >= 2: 至少需要2帧观测才能进行三角化
        // - start_frame < WINDOW_SIZE - 2: 确保有足够的观测约束
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // ===== 第2步：跳过已有深度估计的特征点 =====
        // 如果特征点已经有可靠的深度估计（来自优化或激光雷达），则信任第一次估计
        // 避免重复计算，提高效率并保持深度一致性
        if (it_per_id.estimated_depth > 0)
            continue;

        // ===== 第3步：初始化三角化变量 =====
        // imu_i: 特征点首次观测帧的索引  
        // imu_j: 当前处理帧的索引（从首次观测帧开始）
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        // 断言检查：当前实现只支持单目相机
        ROS_ASSERT(NUM_OF_CAM == 1);
        
        // 构建SVD求解的系数矩阵A
        // 矩阵大小：(2 * 观测帧数) x 4
        // 每帧提供2个约束方程，求解4维齐次坐标
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;  // 约束方程的行索引

        // ########################################
        // ##### 第4步：设置参考帧投影矩阵 #####
        // ########################################
        
        // 构建第一帧（参考帧）的投影矩阵P0 = [I|0]
        Eigen::Matrix<double, 3, 4> P0;
        
        // 计算第一帧相机在世界坐标系中的位置和姿态
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];  // 相机位置
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];              // 相机姿态
        
        // 参考帧设为标准形式：P0 = [I|0]（相机坐标系作为世界坐标系）
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();  // 旋转部分：单位矩阵
        P0.rightCols<1>() = Eigen::Vector3d::Zero();     // 平移部分：零向量

        // ########################################
        // ##### 第5步：构建多视角约束方程 #####
        // ########################################
        
        // 遍历特征点在各帧中的观测
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;  // 递增到下一观测帧

            // ===== 计算当前帧相对参考帧的位姿变换 =====
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];  // 当前帧相机位置
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];              // 当前帧相机姿态
            
            // 计算相对变换（从参考帧到当前帧）
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);  // 相对平移
            Eigen::Matrix3d R = R0.transpose() * R1;         // 相对旋转

            // ===== 构建当前帧的投影矩阵 =====
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();           // 旋转部分
            P.rightCols<1>() = -R.transpose() * t;     // 平移部分

            // ===== 获取归一化特征点坐标 =====
            // 将观测点归一化为单位向量，消除尺度影响
            Eigen::Vector3d f = it_per_frame.point.normalized();

            // ===== 构建DLT约束方程 =====
            // 基于共线约束：特征点、投影中心、图像点三点共线
            // 交叉乘积为零：f × (P * X) = 0，展开得到线性约束
            
            // 第一个约束：f[0] * P.row(2) - f[2] * P.row(0) = 0
            // 对应x方向的共线约束
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            
            // 第二个约束：f[1] * P.row(2) - f[2] * P.row(1) = 0  
            // 对应y方向的共线约束
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            // 跳过参考帧（自身约束无意义）
            if (imu_i == imu_j)
                continue;
        }
        
        // 验证约束方程数量正确性
        ROS_ASSERT(svd_idx == svd_A.rows());
        
        // ########################################
        // ##### 第6步：SVD求解齐次线性方程组 #####
        // ########################################
        
        // 使用SVD分解求解Ax = 0的最小二乘解
        // 解对应于A的最小奇异值对应的右奇异向量
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV)
                                   .matrixV().rightCols<1>();
        
        // 从齐次坐标恢复欧几里得坐标
        // svd_V = [X, Y, Z, W]^T，欧几里得坐标 = [X/W, Y/W, Z/W]^T
        // 这里只需要深度信息：depth = Z/W
        double svd_method = svd_V[2] / svd_V[3];

        // ########################################
        // ##### 第7步：更新深度估计 #####
        // ########################################
        
        // 保存三角化得到的深度估计
        it_per_id.estimated_depth = svd_method;
        
        // ===== 三角化失败检查 =====
        // 如果深度为负值（不符合物理约束），使用默认深度
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }
    }
}

/**
 * [功能描述]：移除被标记为外点的有效特征点，清理优化过程中识别出的异常数据
 * 
 * 外点识别来源：
 * - Bundle Adjustment中的鲁棒估计器标记
 * - 重投影误差过大的特征点
 * - 几何一致性检查失败的特征点
 * 
 * 处理策略：只删除仍在使用且被确认为外点的特征点
 */
void FeatureManager::removeOutlier()
{
    // ########################################
    // ##### 调试断点（开发阶段使用） #####
    // ########################################
    
    // ROS调试断点，用于开发和调试阶段暂停程序执行
    // 在生产环境中通常会被注释或移除
    ROS_BREAK();
    
    // ########################################
    // ##### 初始化删除过程 #####
    // ########################################
    
    // 有效特征点索引计数器
    // 用于统计处理过程中遇到的有效特征点数量
    int i = -1;
    
    // ########################################
    // ##### 安全遍历并删除外点特征 #####
    // ########################################
    
    // 使用双迭代器模式确保删除过程中迭代器的安全性
    // - it: 当前处理的特征点迭代器
    // - it_next: 下一个特征点迭代器（预先递增，防止删除时失效）
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        // ===== 第1步：预先保存下一个迭代器位置 =====
        // 在可能删除当前元素之前，先获取下一个有效位置
        // 确保即使当前迭代器被删除，循环仍能正常继续
        it_next++;
        
        // ===== 第2步：更新有效特征点计数 =====
        // 特殊的计数方式：只对有观测数据的特征点计数
        // used_num != 0 表示特征点仍在使用中（有观测数据）
        // 使用 += 布尔值的技巧：true=1, false=0
        i += it->used_num != 0;
        
        // ===== 第3步：外点检查和删除 =====
        // 删除条件需要同时满足两个条件：
        // 条件1：it->used_num != 0  (特征点仍在使用，有观测数据)
        // 条件2：it->is_outlier == true  (被标记为外点)
        if (it->used_num != 0 && it->is_outlier == true)
        {
            // 从特征管理器中删除该外点特征
            // 包括其完整的观测历史和所有相关数据
            feature.erase(it);
        }
    }
}

/**
 * [功能描述]：处理滑动窗口边缘化时的特征点管理，移除最旧帧并转换深度坐标系
 * 
 * @param marg_R：被边缘化帧的相机旋转矩阵（世界坐标系到相机坐标系）
 * @param marg_P：被边缘化帧的相机位置向量（世界坐标系中的位置）
 * @param new_R：新第0帧的相机旋转矩阵
 * @param new_P：新第0帧的相机位置向量
 * 
 * 核心任务：
 * 1. 更新所有特征点的起始帧索引
 * 2. 处理从被边缘化帧开始的特征点的深度坐标系转换
 * 3. 清理观测不足的特征点
 */
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, 
                                           Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    // ########################################
    // ##### 安全遍历所有特征点 #####
    // ########################################
    
    // 使用双迭代器模式进行安全删除
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;  // 预先保存下一个位置，防止删除时迭代器失效

        // ########################################
        // ##### 情况1：非第0帧起始的特征点 #####
        // ########################################
        
        if (it->start_frame != 0)
        {
            // 简单情况：特征点不是从被边缘化的第0帧开始
            // 只需要将起始帧索引前移1位，适应新的窗口编号
            it->start_frame--;
        }
        else
        {
            // ########################################
            // ##### 情况2：第0帧起始的特征点（复杂处理） #####
            // ########################################
            
            // ===== 第1步：提取旧帧中的深度信息 =====
            // 获取特征点在第0帧（即将被边缘化帧）中的归一化坐标
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            double depth = -1;  // 初始化深度为无效值
            
            // 深度信息优先级：激光雷达深度 > 估计深度
            if (it->feature_per_frame[0].depth > 0)
            {
                // 优先使用激光雷达提供的深度信息（更精确）
                depth = it->feature_per_frame[0].depth;
            }
            else if (it->estimated_depth > 0)
            {
                // 次选使用视觉估计的深度信息
                depth = it->estimated_depth;
            }

            // ===== 第2步：移除被边缘化帧的观测 =====
            // 删除第0帧的观测记录，因为该帧即将从滑动窗口中移除
            it->feature_per_frame.erase(it->feature_per_frame.begin());

            // ===== 第3步：检查剩余观测数量 =====
            if (it->feature_per_frame.size() < 2)
            {
                // 观测数不足：删除整个特征点
                // 特征点需要至少2帧观测才能进行有效的三角化和优化
                feature.erase(it);
                continue;  // 跳到下一个特征点
            }
            else
            {
                // ########################################
                // ##### 第4步：深度坐标系转换 #####
                // ########################################
                
                // ===== 步骤4.1：旧相机坐标系中的3D点 =====
                // 将归一化坐标 × 深度 = 相机坐标系中的3D坐标
                Eigen::Vector3d pts_i = uv_i * depth;
                
                // ===== 步骤4.2：转换到世界坐标系 =====
                // 旧相机坐标系 → 世界坐标系
                // P_world = R_marg * P_camera + t_marg
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                
                // ===== 步骤4.3：转换到新相机坐标系 =====
                // 世界坐标系 → 新相机坐标系
                // P_new_camera = R_new^T * (P_world - t_new)
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                
                // 提取新坐标系下的深度（Z坐标）
                double dep_j = pts_j(2);

                // ########################################
                // ##### 第5步：更新深度估计 #####
                // ########################################
                
                // ===== 情况5.1：新第0帧有激光雷达深度 =====
                if (it->feature_per_frame[0].depth > 0)
                {
                    // 优先使用激光雷达深度，精度更高
                    it->estimated_depth = it->feature_per_frame[0].depth;
                    it->lidar_depth_flag = true;  // 标记为激光雷达深度
                } 
                // ===== 情况5.2：使用坐标变换后的深度 =====
                else if (dep_j > 0)
                {
                    // 使用坐标变换计算的深度（正值，物理合理）
                    it->estimated_depth = dep_j;
                    it->lidar_depth_flag = false;  // 标记为视觉估计深度
                } 
                // ===== 情况5.3：深度异常处理 =====
                else 
                {
                    // 深度为负或零，物理不合理，使用默认初始深度
                    it->estimated_depth = INIT_DEPTH;
                    it->lidar_depth_flag = false;
                }
            }
        }
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}