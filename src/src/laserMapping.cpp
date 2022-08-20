#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;           //雷达最大的探测距离--avia450m,velodyne100m
const float MOV_THRESHOLD = 1.5f;   //移动阈值 1.5 float

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;            //BoxPointType是一个结构体（数据类型）里面有两个数组属性
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;//std::deque容器：存的都是点云类型的指针，每一个指针都是一帧点云的初始点的指针
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());             //点云指针类型
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;    //特征点数组 （未用）

pcl::VoxelGrid<PointType> downSizeFilterSurf;   //创建体素滤波器    //x.setInputCloud(cloud) 输入点云； x.setLeafSize(0.5f,0.5f,0.5f)设置体素； x.filter(cloud_)过滤后的点云
pcl::VoxelGrid<PointType> downSizeFilterMap;    //此滤波器针对整个地图； 上一个滤波器针对去畸变后的当前帧点云

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);    //x轴点-局部坐标系下    （2，0，0）
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);   //x轴点-世界坐标系下    （2，0，0）
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;  //测量值
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;// 构建这个类的一个对象 //待优化状态、状态维数、加速度与角速度  //（状态、噪声维度、输入状态、观测状态、）
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;                    //nav_msgs::下有五种消息类型：Path、Odometry、MapMetaData、OccupancyGrid、GridCells
                                        //Path的结构：std_msgs/Header header       -> uint32 seq; time stamp; string frame_id;
                                        //           geometry_msgs/PoseStamped [] -> std_msgs/Header header
                                        //           geometry_msgs/Pose pose      -> float64 x; float64 y; float64 z; float64 w;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

//智能指针————指向的数据类型是<Preprocess> <ImuProcess>
shared_ptr<Preprocess> p_pre(new Preprocess());//new一个类的对象，用智能指针接收
shared_ptr<ImuProcess> p_imu(new ImuProcess());//p_imu指向ImuProcess这个类对象，里面有好多属性和方法：属性有加速度计、陀螺仪测量值，偏置等等

void SigHandle(int sig)//中断处理函数，ctrl+c后，就运行这个函数
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)       //（2，0，0，   2，0，0）x轴点 局部坐标系，，，x轴点 全局坐标系 //全局坐标系用来装东西
{
    V3D p_body(pi[0], pi[1], pi[2]);    //2，0，0
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos); //转到全局坐标系

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history); //获得被删除的点云
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

//根据激光雷达视场角分割场景图
BoxPointType LocalMap_Points;           //用于计算BOX大小   //盒点类型
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();                 //装有BoxPointType类型的vector容器
    kdtree_delete_counter = 0;          //kd树删除计数器
    kdtree_delete_time = 0.0;           //kd树删除时间
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);    //imu坐标系转到世界坐标系（参数1：x轴点-局部坐标系， 参数2：x轴点-世界坐标系）（2，0，0，     2，0，0）//什么用啊？
    V3D pos_LiD = pos_lid;              //pos_lid是当前20ms中尾部的雷达位姿————三维向量
    if (!Localmap_Initialized){         //若局部地图没初始化
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;    //最小的顶点(0) = 雷达位姿的(x) - 盒子的一半长度
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;    //最大的顶点(0) = 雷达位姿的(x) + 盒子的一半长度    //初始位姿 - 200（200用立方体个数度量？）
        }
        Localmap_Initialized = true;    //初始化完毕---设置了局部地图的最大最小点
        return; //----------------------//初始化完毕直接退出
    }
    float dist_to_map_edge[3][2];       //到地图边缘的距离-3行2列
    bool need_move = false;             //需要移动局部地图
    for (int i = 0; i < 3; i++){         //分别计算当前位姿距离地图边缘最大点和最小点的距离
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);//第1列 = 雷达位姿 - 局部地图的最小点位置 //计算当前位姿距离地图边缘有多远
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);//第2列 = 雷达位姿 - 局部地图的最大点位置
        // 当前位姿离地图边缘的距离<1.5倍探测距离(任意一轴)，则需要移动局部地图
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    //cout << "******cube_len = " << cube_len << " //MOV_ = " << MOV_THRESHOLD << " //DET_ = " << DET_RANGE << endl;
    if (!need_move) return; //---------如果不需要更新局部地图，则结束此函数----上面的目的就是判断，当前离局部地图边缘是不是太近了，近到一定阈值，就需要移动局部地图（更换视角）
                            //和着进来就判断需不需要移动局部地图，不需要直接退
    //需要移动局部地图
    BoxPointType New_LocalMap_Points, tmp_boxpoints;    //构建新的局部地图，临时局部地图 的最大最小点
    New_LocalMap_Points = LocalMap_Points;              //旧的局部地图 给到 新的局部地图 变量中
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));//velodyne是(315,500)
    //cout << "mov_dis = " << mov_dist << endl;
    //cout << "cube_len = " << cube_len << " //MOV_ = " << MOV_THRESHOLD << " //DET_ = " << DET_RANGE << endl;
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;                //旧的局部地图 给到 临时局部地图 变量中
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){   //如果是离最小顶点近
            New_LocalMap_Points.vertex_max[i] -= mov_dist;          //新局部地图的最大顶点 = 旧局部地图最大顶点 - 要移动的距离
            New_LocalMap_Points.vertex_min[i] -= mov_dist;          //新局部地图的最小顶点 = 旧局部地图最小顶点 - 要移动的距离
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);                    //装入容器中--need remove
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){//如果是离最大顶点近
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;          //把新的局部地图 最大最小点 更新到 当前局部地图点

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);   //删除盒内的点，从ikdtree中删除点
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

//sensor_msgs::POintCloud2 点云数据处理回调
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();                                      //缓存器上锁
    scan_count ++;                                          //扫描次数+1
    double preprocess_start_time = omp_get_wtime();         //定义预处理开始时间
    if (msg->header.stamp.toSec() < last_timestamp_lidar)   //header.stamp.toSec()把时间戳转化成浮点型格式
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    //-----
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());     //定义个点云组 ，用ptr指向他
    p_pre->process(msg, ptr);           //注意：回调函数里做的点云处理   //预处理，包含数据读取  //构造函数里就根据不同雷达类型，做对应的处理，比如怎么取点，怎么取特征，最后追加到缓存器中
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

//livox_ros_driver::CustomMsg 点云数据处理回调
double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) //回调20ms走一次，也就是说msg是整个20ms的所有点云，不分帧，imu分帧
{
    mtx_buffer.lock();      //----------------------------------------lock
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    
    if (msg->header.stamp.toSec() < last_timestamp_lidar)//检查lidar时间戳
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();   //更新时间戳
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )    //缓存器不为空但时间没同步
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty()) //雷达和imu的时间偏移大于1  //这里面的函数要么不走，走只走一次
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());            //创建点云
    p_pre->process(msg, ptr); //传入的ptr是空的，msg是一帧点云     //预处理，包含点云读取   //ptr将会指向pcl形式的点云且是有效点云
    lidar_buffer.push_back(ptr);  //存入好多ptr，每个ptr都是一个             //把这些转换后的且有效的点云，放入雷达缓存器中。至此，雷达类型与算法无关****（重要）****不同激光雷达如何处理，看前面就行了
    time_buffer.push_back(last_timestamp_lidar);               //把当前20ms这组的雷达的时间戳放在缓存器中
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;//仅用来输出日志了
    mtx_buffer.unlock();    //----------------------------------------unlock
    sig_buffer.notify_all();
}

//imu数据回调
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) //20ms中所有的imu数据都方缓存器中  ***//msg_in是一个ptr，imubuffer是好多个ptr，回调走的时候，ros缓存队列存了好多个ptr，把所有ptr都放imubuffer中，也就是会连续执行好多次回调直到ros队列没有缓存了
{                                                                   //每次调用ros::spinOnce()都会执行与消息队列中缓存的信息数量相同次数的回调函数，只要回调函数执行够快的话，就能清空队列
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)  //abs绝对值函数
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)         //检查imu时间戳，保证当前帧大于上一帧时间
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;
    //cout << "in cbk: msg = \n" << msg->linear_acceleration << endl;
    imu_buffer.push_back(msg);  //ros：：spinOnce()一次性执行多次回调，把20ms的所有imu帧都放到了buffer中
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

//时间同步
double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)              //20ms到了，把数据从缓存器中拿出来，做时间同步
{
    if (lidar_buffer.empty() || imu_buffer.empty()) //若雷达缓存器与imu缓存器为空，则改变标志位，不做处理
    {
        return false;
    }

    /*** push a lidar scan ***/ //单处理雷达————计算雷达结束的时间
    if(!lidar_pushed)//lidar_pushed：当前帧雷达数据已出缓存器标志   //如果雷达数据没有出缓存器，则。。。
    {
        meas.lidar = lidar_buffer.front();//deque(缓存器是deque容器) 里的front函数---获取第一个元素（buffer里都是指向点云的指针）//缓存器里的每个指针都是一次20ms的扫描
        meas.lidar_beg_time = time_buffer.front();//20ms中最早点的时间
        if (meas.lidar->points.size() <= 1) // time too little //第一帧（front）有效点云数量太少
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;//时间叠加
            ROS_WARN("Too few input point cloud!\n");//这帧点云数量太少
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)    //注意：这里的curvature曲率，里面装的是时间——meas由buffer给的，buffer由preprocess给的，preprocess中做的格式转换
        {       //如果 该帧雷达最后的点的时间 还没到 整体扫描时间的一半
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;//那就强制让雷达结束时间 = 雷达开始时间 + 扫描时间（补上来，结束时间）
            //没有将扫描次数+1（不视为一次完整的扫描）
        }
        else    //雷达点又不少，结束的时间也在后半段
        {
            scan_num ++;    //将这20ms视为一次完整的扫描————只用来求平均扫描时间了————平均扫描时间又只用来上两个情况了//***一次扫描20ms，没有帧的概念。就直接是20ms中的所有点云————在缓存器中的一个指针所指，每次处理只拿一个指针，并且找出这期间的imu。imu有帧的概念
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);  //结束时间 = 开始时间+最后的点的时间（这时间戳得学习一下）（第二个参数应该是此帧第一个点到最后一个点的持续时间）
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;//更新平均扫描时间
        }
        //更新数据
        meas.lidar_end_time = lidar_end_time;
        lidar_pushed = true;    //雷达数据已从缓存器中出来
    }

    //20ms中的最后一帧，保证雷达采样时间晚于imu采样时间 //这个时间判断不是按帧来的，是按雷达的20ms所有点的最后点的时间来的 //采点频率240khz
    if (last_timestamp_imu < lidar_end_time)    //若20ms中最后一个imu采样时间 < 20ms中雷达采样的时间    //一定是imu时间包住雷达时间，因为还要做反向传播呢
    {
        return false;
    }//如果雷达采样包不住imu采样，下面也不用处理了
    //——————————————————————————————————————————————————————————————————————————————————————————————上为雷达，下为imu——————————
    /*** push imu data, and pop from imu buffer ***/    //把imu数据从imu缓存器中拿出来
    double imu_time = imu_buffer.front()->header.stamp.toSec();//把第一个20ms的imu采样时间，赋值给imu_time //imu_buffer.front()是一个指针，指向的旗下的时间戳，在转换成秒
    meas.imu.clear();       //把上一次处理时，容器中的数据清空（用一个碗，再盛碗饭）
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))//imu开始的时间必须比这20ms的雷达时间小 且 imu缓存器不为空，才行*****取出这之间的imu数据
    {
        cout << "xxx" << endl;
        imu_time = imu_buffer.front()->header.stamp.toSec();    //根上面那语句重复了！上面应该只定义的！
        if(imu_time > lidar_end_time) {cout << "imutime > lidar_end_time" << endl;break;}                    //一旦雷达的时间包不住imu，直接break//是不是有点多余，已经在while处判断了阿
        //cout << "in sync: imu_buffer.front = \n " << imu_buffer.front() << endl;
        meas.imu.push_back(imu_buffer.front());                 //把20ms中的第一个追加到观测量上（imu观测量是deque容器） **** —————— 是先进入缓存器，再传入 此帧观测量
        imu_buffer.pop_front();//第一个数据丢掉***************遍历的好方法（如果只用一次的话），就第一个赋值出来，然后弹出，while循环，这样第二个就变成第一个了。继续赋值出来
    }//while循环把imu数据一次性全赋值出来了，但是雷达的没有while循环，这是为什么呢？因为！！雷达不分帧，是20ms的一大片点云，而imu是20ms中的多个imu帧（但是反向又需要时间戳，其实点云的点自带时间戳）

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;       //更新标志为位
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)   //Nearest_Points并不是nearpoint，是滤波后的点云 vector <pointvector>
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min; //floor 返回小于x的整数，10.5，返回10
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)   //  用于初始化
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/ //最近点搜索和残差计算
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);     //第一个转是转换到局部IMU坐标系，第二个转是转到世界坐标系
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/ //计算雅各比矩阵H 和观测向量
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;       //avia的话：offset_R_L_I = 1，旋转 = 1
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec); //conjugate共扼矩阵
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);  //12维，后6维是外参估计
        }
        else
        {       //block。从i,0元素算，算1行12列，一横倘
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;          //6维
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity ;
    }
    solve_time += omp_get_wtime() - solve_start_;
}
ros::Publisher pub_pre_odometry;
ros::Publisher pub_pre_path;
nav_msgs::Path IMU_path;
ros::Publisher pub_pre_pointcloud;
ros::Publisher pub_pointcloud;
void publish_pre_imu( esekfom::esekf<state_ikfom, 12, input_ikfom> &state)
{
    Eigen::Quaterniond q = Eigen::Quaterniond(state.get_x().rot);

    nav_msgs::Odometry imu_pre_odometry;
    imu_pre_odometry.header.frame_id = "world";
    imu_pre_odometry.child_frame_id = "/body";
    imu_pre_odometry.pose.pose.position.x = state.get_x().pos.x();
    imu_pre_odometry.pose.pose.position.y = state.get_x().pos.y();
    imu_pre_odometry.pose.pose.position.z = state.get_x().pos.z();

    //std::cout << "pose:\n" << imu_pre_odometry.pose.pose.position << std::endl;

    imu_pre_odometry.pose.pose.orientation.w = q.w();
    imu_pre_odometry.pose.pose.orientation.x = q.x();
    imu_pre_odometry.pose.pose.orientation.y = q.y();
    imu_pre_odometry.pose.pose.orientation.z = q.z();
    pub_pre_odometry.publish(imu_pre_odometry);

    geometry_msgs::PoseStamped imu_pre_path;
    imu_pre_path.header.stamp = ros::Time().now();
    imu_pre_path.header.frame_id = "world";
    imu_pre_path.pose.position.x = state.get_x().pos.x();
    imu_pre_path.pose.position.y = state.get_x().pos.y();
    imu_pre_path.pose.position.z = state.get_x().pos.z();
    imu_pre_path.pose.orientation.x = q.x();
    imu_pre_path.pose.orientation.y = q.y();
    imu_pre_path.pose.orientation.z = q.z();
    imu_pre_path.pose.orientation.w = q.w();
    IMU_path.header.frame_id = "world";
    IMU_path.poses.push_back(imu_pre_path);
    pub_pre_path.publish(IMU_path);  
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    //读取配置参数（ROS参数传递）
    //launch文件中: <rosparam command="load" file="$(find examle_pkg)/example.yaml" /> 即可通过rosparam导入.yaml文件参数
    nh.param<bool>("publish/path_en",path_en, true);                //将yaml文件中 publish/path_en 的值赋给变量path_en,没有参数时默认true  //是否发布路径的topic
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);    //是否发布当前正在扫描的点云topic
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);  //是否在全局帧点云扫描中降低点云数量
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true); //是否发布经过运动畸变矫正注册到IMU坐标系的点云topic
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);            //滤波最大迭代此书
    nh.param<string>("map_file_path",map_file_path,"");             //地图的保存路径
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");  //将yaml文件中lid_topic 赋值给lid_topic变量，无时默认是/livox/lidar
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");   //imu的topic名
    nh.param<bool>("common/time_sync_en", time_sync_en, false);     //是否需要时间同步，只有当外部未进行时间同步时，设为true
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);  //降采样时的体素大小 //角点
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);  //降采样时的体素大小 //平面点
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);    //降采样时的体素大小 //地图
    nh.param<double>("cube_side_length",cube_len,200);              //地图的局部区域长度（论文中有）//实际取的1000--在launch中写了 200 m？1000m？
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);           //雷达的最大探测范围 avia = 450m，velodyne = 100m
    nh.param<double>("mapping/fov_degree",fov_deg,180);             //雷达的视场角
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);                //IMU陀螺仪的协方差
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);                //IMU加速度计的协方差
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);         //IMU陀螺仪偏置的协方差
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);         //IMU加速度计偏置的协方差
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);       //最小距离阈值（过滤到0-blind范围内的点云）
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);//雷达类型
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);      //雷达扫描的线数
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);//
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);    //雷达扫描频率
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);  //采样间隔，每隔2个点取一个点
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);//是否提取特征呢改（默认不提取特征）
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);   //是否输出调试log信息
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);//是否进行自动外参标定
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);     //是否保存pcd地图文件
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);      //每个pcd文件保存多少雷达帧（-1表示所有雷达帧保存到一个pcd文件中）
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());//外参T
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());//外参R
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";    //全局坐标系(rviz,个传感器信息统一，以便全部显示)
    
    /*** variables definition ***/  //变量的定义
    int effect_feat_num = 0, frame_num = 0; // 有效的特征点数量  // 雷达总的帧数
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        //平移向量 //旋转向量 //每帧平均处理的总时间 //每帧icp的平均实际时间 //每帧匹配平均时间   //每帧ikd tree处理的平均时间 //每帧计算的平均时间 //每帧计算的平均时间（H恒定时）
    bool flg_EKF_converged, EKF_stop_flg = 0;       //扩展卡尔曼滤波收敛标志（未用到该变量）//扩展卡尔曼滤波停止标志（未用到该变量）
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);//视场角度（未用）
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);//视场角度的半值cos值（未用）

    _featsArray.reset(new PointCloudXYZI());

    //menset函数，将某一块内存全部设定为指定值————通常用于初始化
    memset(point_selected_surf, true, sizeof(point_selected_surf));//把point_selected_surf变量的内存全设定为true
    memset(res_last, -1000.0f, sizeof(res_last));           //把数组res_last内元素的值全部设置为 -1000.0f，数组res_last用于点到平面间的残差距离
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);//定义体素滤波器参数，体素边长为filter_size_surf_min = 0.5
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);   //提前设定好----0.5m
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);               //由yaml文件读进来的extrinT给到Lidar_T_wrt_IMU，再给到p_imu
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);

    //设置imu参数，p_imu为shared_ptr
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU); //换言之，就是把yaml中的参数，给到这里的具体变量 //传递IMU与雷达的外参数
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));     //设置陀螺仪协方差
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));     //设置加速度计协方差
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));  //设置陀螺仪偏置协方差
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));  //设置加速度计偏置协方差

    double epsi[23] = {0.001};  //收敛阈值
    fill(epsi, epsi+23, 0.001); //把epsi数组中的23个数全填写未0.01 //初始化重复？//stl::fill()赋值函数，高级版for循环赋值给数组——————fill(修改元素的头，修改元素的尾，修改元素的值)
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);    //扩展卡尔曼滤波初始化
            //获得状态转移矩阵，测量值与估计值之差对状态的偏导，测量值与估计值之差对噪声的偏导，构造优化函数并求解该函数的雅克比矩阵，最大迭代次数，要满足的迭代误差阈值

    /*** debug record ***/  //调试记录日志=================
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;
    //=====================================================

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);                          //点云的订阅器sub_pcl，订阅点云的topic，放到lidar缓存器中 //因为20ms处理一次，先放缓存器里，到时间就处理
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);             //imu订阅器sub_imu，订阅imu的topic，放imu缓存器里
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>       //发布正在扫描的点云topic
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>  //发布经过运动畸变矫正后，注册到imu坐标系的点云topic
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>     //代码中没用到
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>        //代码中没用到
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>              //发布当前里程计信息topic
            ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path>                  //发布里程计总路径
            ("/path", 100000);
    pub_pre_odometry = nh.advertise<nav_msgs::Odometry>("/imu_Odo", 10000);
    pub_pre_path = nh.advertise<nav_msgs::Path>("/imu_path", 10000);
    pub_pre_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("/pre_pointcloud", 100000);
    pub_pointcloud = nh.advertise<sensor_msgs::PointCloud2>("/my_practice_pointcloud", 100000);
//------------------------------------------------------------------------------------------------------数据处理部分---------------------------
    signal(SIGINT, SigHandle);              //中断处理函数（比如有ctrl+c，则执行SigHandle）
    ros::Rate rate(5000);                   //20ms处理一次，20ms看作是雷达的一次扫描
    bool status = ros::ok();
    int test_times = 0;
    //--------------------------------主循环
    while (status)//-----------------一次while，是一次20ms的处理
    {
        if (flg_exit) break;                                //如果有中断，则直接退出循环
        //在做回调之前，消息都存在ros的消息队列缓存器中————回调函数把消息从队列中取出来放在buffer里，并做一些预处理
        ros::spinOnce();      //执行回调的函数(20ms做一次)，与订阅、rosok，rate等连用 //若没有消息，则会造成线程阻塞（所以回调来了的时候，处理消息完毕后有一个唤醒线程操作）
        //回调函数不宜太慢，若处理时间大于传感器发布周期，可能会丢失数据 
        //多个传感器，比如imu缓存器5个数据，雷达缓存器5个数据，雷达的回调处理太慢，第二次处理时，已经不是紧接着的那五个数据了，是溢出之后的数据了，而imu还是第二次的5个数据，就不同步了
        //ros消息的回调处理函数 //ros::spin() 或 ros::spinOnce()；前者调用后，程序不继续执行；后者调用后继续执行下面的程序
        
        //将点云数据和IMU数据从缓存器中取出，进行时间对齐，保存到Measure中
        if(sync_packages(Measures)) //把数据都从缓存器里拿出来了    //这玩意是全局变量 //此时！Measures是空的，是空碗，进取接水
        {
            if (flg_first_scan)//如果激光雷达是第一次扫描，就把这一次的时间，定义为 第一次 的雷达时间与imu时间  //不懂？？
            {
                //记录下雷达第一次的扫描时间
                first_lidar_time = Measures.lidar_beg_time; //Measures中属性赋值给first_lidar_time，first_lidar_time再给到p_imu->first_lidar_time
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false; //修改标志位 ———— 非第一次输入
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;//IKD树建图以及相关算法求解所用时间记录

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            //publish未去畸变的点云
            PointCloudXYZI pcl_pre = *(Measures.lidar);
            sensor_msgs::PointCloud2 pcl_pre_ros;
            pcl::toROSMsg(pcl_pre, pcl_pre_ros);
            pcl_pre_ros.header.frame_id = "world";
            pub_pre_pointcloud.publish(pcl_pre_ros);

            //p_imu是指向imuProcess类的对象的指针，下面有成员函数Process()
            p_imu->Process(Measures, kf, feats_undistort);      //imu预处理（包含点云畸变处理）————初始化/前向传播  //Measures是一整个20ms的所有观测量
            //publish_pre_imu(kf);
            state_point = kf.get_x();                           //获取x 姿态（获取先验）
            test_times++;
            cout << "第" << test_times << "个while-------------------------------" << endl;
            if(0)
            {
                cout << "imu:" << endl;
                cout << "rot = \n" << state_point.rot << "\npos = " << state_point.pos << "\nvel = " << state_point.vel << "\n ba:bg=" << state_point.ba << " " << state_point.bg << endl;
                cout << "------" << endl;
            }

            //publish去畸变后的点云
            sensor_msgs::PointCloud2 pcl_ros;
            pcl::toROSMsg(*feats_undistort, pcl_ros);
            pcl_ros.header.frame_id = "world";
            pub_pointcloud.publish(pcl_ros);

            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; //激光雷达的位姿 //imu位姿+imu旋转+外参  //这里只是先定义出来吧

            if (feats_undistort->empty() || (feats_undistort == NULL))  //去畸变后没有点了
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            //EKF初始化完成---------------------------------
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();                         //对激光雷达视场中的地图进行分割

            /*** downsample the feature points in a scan ***/   //降采样----体素滤波三部曲：设定体素，输入点云，过滤点云
            downSizeFilterSurf.setInputCloud(feats_undistort);  //体素滤波——输入待滤波的点云  //对去畸变后的点云降采样（为什么不先降采样，再取畸变呢，岂不是更快）
            downSizeFilterSurf.filter(*feats_down_body);        //滤波————滤波后的点云为*feats_down_body，feats_down_body是指向点云的指针，这里解引用掉，body是局部坐标系
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();   //降采样后，当前去畸变后的点云中还有多少个点---点的数量

            /*** initialize the map kdtree ***/             //初始化建图的KD树结构-------------------------
            if(ikdtree.Root_Node == nullptr)                            //若还没有根节点
            {
                if(feats_down_size > 5)                                 //若滤波后的点数大于5个
                {
                    ikdtree.set_downsample_param(filter_size_map_min);  //设置IKDTree降采样参数（多大的立方体）---0.5m
                    feats_down_world->resize(feats_down_size);          //世界坐标系下降采样点的空间大小重置--初始化
                    for(int i = 0; i < feats_down_size; i++)            //坐标系转换
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));//将IMU坐标系中降采样的点转到世界坐标系下
                    }
                    ikdtree.Build(feats_down_world->points);            //根据世界坐标系下的降采样的点，构建IKdTree
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();                   //获取ikdtree中有效点的数目——————仅用来调试
            kdtree_size_st = ikdtree.size();                            //kdtree中的点数量——————仅用来日志输出
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/   //ICP和卡尔曼滤波器更新
            if (feats_down_size < 5)                                    //若降采样后的点云少于5个
            {
                ROS_WARN("No point, skip this scan!\n");                //跳过本次扫描处理
                continue;
            }
            
            normvec->resize(feats_down_size);                           //根据这次降采样后的点云数量，重新设置  //重新设置了两点云（根据滤波后的点数）
            feats_down_world->resize(feats_down_size);                  //

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);       //状态中Imu与雷达的旋转偏差 转换为 欧拉角   //仅仅用来日志输出
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;//计算并输出状态处理结果，到日志文件

            //是否显示iKdTree中的地图点
            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }
            pointSearchInd_surf.resize(feats_down_size);                //对搜索点索引数组以及最近点数组大小进行更新
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/     //开始迭代状态估计————迭代更新
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);//卡尔曼迭代函数 前者针对一个特定系统修改的迭代误差状态EKF更新，后者求解时间
            state_point = kf.get_x();//更新后的状态 // *****更新的最终目的————得到精准的定位结果
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;//激光雷达位姿——————更新完毕
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];//相应的四元数

            if(0)
            {
                cout << "esikf:" << endl;
                cout << "rot = \n" << state_point.rot << "\npos = " << state_point.pos << "\nvel = " << state_point.vel << "\n ba:bg=" << state_point.ba << " " << state_point.bg << endl;
            }


            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);//发布里程计  ———— pubOdomAftMapped是发布者，OdomAftMapped是里程计数据

            /*** add the feature points to map kdtree ***/  //将特征点添加到地图kdtree中
            t3 = omp_get_wtime();
            map_incremental();                  //iKdTree的增量式地图
            t5 = omp_get_wtime();
            publish_pre_imu(kf);
            /******* Publish points *******/    //发布点云
            if (path_en)                         publish_path(pubPath);                     //发布路径
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);    //当前扫描点
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);//去畸变后imu坐标系下的点云
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/   //调试变量输出 //为了输出的日志
            if (runtime_pos_log)    //在《各个结构所需参数获取》部分进行设置
            {
                frame_num ++;                                                                                               //统计帧数
                kdtree_size_end = ikdtree.size();                                                                           //统计ikdtree的大小
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;                    //每帧点云在框架中计算所用平均时间
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;    //每帧点云ICP迭代所用时间
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;                     //每帧点云进行匹配所用平均时间
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;        //每帧点云进行ikdtree增量操作所用平均时间
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;      //总的求解时间平均值
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;                             //当前帧所用总时间
                s_plot2[time_log_counter] = feats_undistort->points.size();     //去畸变后的特征点数量
                s_plot3[time_log_counter] = kdtree_incremental_time;            //当前帧ikdtree增量时间
                s_plot4[time_log_counter] = kdtree_search_time;                 //当前帧kdtree搜索用的时间
                s_plot5[time_log_counter] = kdtree_delete_counter;              //当前帧kdtree删减计数器
                s_plot6[time_log_counter] = kdtree_delete_time;                 //当前帧kdtree删减所用时间
                s_plot7[time_log_counter] = kdtree_size_st;                     //当前帧添加特征点到地图之前，kdtree的大小
                s_plot8[time_log_counter] = kdtree_size_end;                    //当前帧添加特征点到地图之后，kdtree的大小
                s_plot9[time_log_counter] = aver_time_consu;                    //上面计算出的平均计算时间
                s_plot10[time_log_counter] = add_point_size;                    //kdtree中插入的点数量
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();   //20ms sleep 此时imu、雷达不断做回调，把数据放到缓存器
    }
    //-------------------------------------------------------------------------------------------主循环结束--------------------------------------------


    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
