#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)  //10帧

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess  //用于IMU处理的类
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();   //构造
  ~ImuProcess();  //析构
  
  void Reset();   //重置
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);  
  void set_extrinsic(const V3D &transl, const M3D &rot);  
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;  //白噪声的协方差（）
  //                            观测量                                                  滤波器                           点云
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

  ofstream fout_imu;
  V3D cov_acc;        //加速度————协方差
  V3D cov_gyr;        //陀螺仪————协方差
  V3D cov_acc_scale;  //加速度————比例误差————协方差
  V3D cov_gyr_scale;  //陀螺仪————比例误差————协方差
  V3D cov_bias_gyr;   //陀螺仪————偏置————协方差
  V3D cov_bias_acc;   //加速度————偏置————协方差
  double first_lidar_time;  //雷达第一次的采样时间

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);    //imu初始化（观测状态，滤波器，N）
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);  //点云去畸变

  PointCloudXYZI::Ptr cur_pcl_un_;        //点云指针
  sensor_msgs::ImuConstPtr last_imu_;     //上一帧imu信息
  deque<sensor_msgs::ImuConstPtr> v_imu_; //imu信息容器
  vector<Pose6D> IMUpose;                 //imu——（偏移时间、加速度、角速度、P、V、Q）
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;        //外参旋转
  V3D Lidar_T_wrt_IMU;        //外参平移
  V3D mean_acc;               //平均的加速度值
  V3D mean_gyr;               //平均的陀螺仪值
  V3D angvel_last;            //上一组imu最后帧的加速度数据
  V3D acc_s_last;
  double start_timestamp_;    //开始的时间
  double last_lidar_end_time_;//20ms中雷达的最后一次采样时间
  int    init_iter_num = 1;   //帧数
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;//imu是否需要初始化
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)    //构造函数，同时继承了类的三个属性————是第一帧imu，需要初始化imu，开始时间戳-1
{
  //构建需要的变量
  init_iter_num = 1;                          //初始迭代次数
  Q = process_noise_cov();                    //四元数（12维）
  cov_acc       = V3D(0.1, 0.1, 0.1);         //加速度协方差（3维）
  cov_gyr       = V3D(0.1, 0.1, 0.1);         //角速度协方差（3维）
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);//加速度偏置（3维）
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);//角速度偏置（3维）
  mean_acc      = V3D(0, 0, -1.0);            //观测 加速度（3维）
  mean_gyr      = V3D(0, 0, 0);               //观测 角速度（3维）
  angvel_last     = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}  //析构

void ImuProcess::Reset() //重置imu————恢复变量为初始值
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;  //初始化缓存器帧数
  v_imu_.clear();         //清空imu缓存器 //没用上啊
  IMUpose.clear();        //清空imu位姿容器
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler) //把比例系数给到imu对象上
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

//1、初始化重力、陀螺偏差、acc和陀螺仪协方差
//2、将加速度测量值标准化为单位重力
//静止初始化
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N) //N记录着，在这次imu观测中，算到了第几帧imu了
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance   //1、初始化重力、陀螺偏置、acc和陀螺仪协方差
   ** 2. normalize the acceleration measurenments to unit gravity **/   //2、将加速度归一化到单位重力
  V3D cur_acc, cur_gyr; //创建三维向量用于记录一帧的imu加速度角速度
  if (b_first_frame_)   //若是第一帧
  {
    Reset();            //把相关变量设置为初始值
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;  //IMU第一帧 线加速度 参数传递 vector3 //.front是deque中的第一个数据
    const auto &gyr_acc = meas.imu.front()->angular_velocity;     //IMU第一帧 角加速度 参数传递
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;    //参数传递----平均---用于后面计算平均加速度和角速度  mean_acc是全局变量
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;    //参数传递----mean_gyr这两个变量 是imu deque容器中的一个值
    first_lidar_time = meas.lidar_beg_time;         //获取第一帧激光雷达开始的时间
  }
  //计算方差    //注意：遍历观测量————遍历的是meas，是缓存器放这里的，外面判断IMU_init处理了多少帧 大于10才行，根据第三个参数init_iter_num，这里N++会直接影响init_iter_num，因为是 地址传递！
  for (const auto &imu : meas.imu)  //目的是：通过遍历，从此次观测的imu的第一帧，推到最后一帧，最终的到此次观测的imu的均值 及 协方差  //遍历meas中的imu数据 meas.imu是deque容器 这个&的imu是一帧imu的数据
  {
    const auto &imu_acc = imu->linear_acceleration; //把当前帧imu加速度给到imu_acc
    const auto &gyr_acc = imu->angular_velocity;    //把当前帧imu角速度给到gyr_acc
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;     //格式转换——>向量形式 cur_acc是当前测量的
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    //更新均值
    /*简单介绍一下这个求平均
      更新后的均值 = 之前的均值 + 此次测量与均值的差/总帧数（此帧数据和均值差了x，把差平均，加到均值上）
    */ 
    mean_acc      += (cur_acc - mean_acc) / N;  
    mean_gyr      += (cur_gyr - mean_gyr) / N;  
    
    //更新协方差 
    //A.cwiseProduct(B)对应系数运算————A第一个元素 * B第一个元素...————A、B必须同一类型，返回值也是该类型的（仅做点对点的系数运算）
    //A.cwiseProduct(B) 矩阵只能进行矩阵运算，数只能进行数的运算，不能将矩阵和数 相加 ————但有时需要同时进行矩阵和数的运算，可以用这个函数（否则代码长，把矩阵转数组，做运算在加回去，麻烦！）
    //mean_acc、cur_acc、cov_acc、cov_gyr都是Eigen::Matrix<double, 3, 1>
    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);  // (N-1)*(cov/N  +  (x-u)/ N)
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);  // （之前协方差均值+当前帧协方差）*传递了多少帧
    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;
    N ++;
  }
  state_ikfom init_state = kf_state.get_x();                  //创建初始状态————混合流形，借助 IKFoM (Iterated Kalman Filter on Manifold) 工具包
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);//求出初始化的重力 = 加速度方向 * 重力大小
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;  //因为现在是静止初始化，所以当前角速度平均值，正是陀螺仪偏置  //求出初始化陀螺仪偏置
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;    //Vector3类型   //传递初始化参数
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;    //SO3类型       //传递初始化参数
  kf_state.change_x(init_state);                //KF的变化量    //初始化完毕，把初始化状态 转换 成一般状态？

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();  //误差状态协方差初始值————论文公式8
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = 0.00001; 
  kf_state.change_P(init_P);                                  //误差状态协方差变化量
  last_imu_ = meas.imu.back();                                //把最后一个IMU数据 转换成上一次的imu数据（当下一帧来的时候用）

}

//imu前向传播，雷达反向传播--> 去畸变     //观测量                                //状态                                //去畸变后的点云
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)//kf_state——论文中 3-c-0 误差状态
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  //将最后一帧尾部的imu添加到当前帧头部的imu
  auto v_imu = meas.imu;      //deque容器类型
  v_imu.push_front(last_imu_);//把上一个20ms的最后的imu ， 赋给这一帧的第一个
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();   //此组 imu数据的 第一帧imu的时间  //没用上
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();    //此组 imu数据的 最后一帧imu的时间
  const double &pcl_beg_time = meas.lidar_beg_time;                   //点云的第一个时间（从测量值中读取出来的）
  const double &pcl_end_time = meas.lidar_end_time;                   //点云的最后一个时间
  /*** sort point clouds by offset time ***/      //按照偏移时间对点云进行排序
  pcl_out = *(meas.lidar);                        //反向传播去畸变时用   //meas.lidar是指针————做了解引用给到pcl_out(meas作为地址传递————还是const——1、动但不想动meas 2、常量动不了)
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);      //从begin，到end，把点云每个点按照时间排序 //排序函数（内部实现采用快速排序+插入排序+堆排序）

  /*** Initialize IMU pose ***/ //imu初始位姿***************这两句重要，获取状态（上一组的最后的状态），把这个状态追加到IMUpose中——————利用了上一个状态的PVQ哦
  state_ikfom imu_state = kf_state.get_x(); //get x 就是获取状态 给到imu_state
  IMUpose.clear();  //IMUpose一共六维（偏移时间(double 1*1)、加速度(1*3)、角速度(1*3)、V(1*3)、P(1*3)、R(3*3)）****
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));  //设定初始位姿
  //IMUpose是容器，好多个Pose，这里push_back只追加一个 ———— 就是把第一个追加进来————先把初始的定了，因为初始的可能是上一组末尾的imu

  /*** forward propagation at each imu point ***/ //前向传播
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;//角速度平均值，加速度平均值，IMU加速度，IMU速度，IMU的位姿//后三个为推断值
  M3D R_imu;        //误差（3*3）
  double dt = 0;    //时间增量
  input_ikfom in;   //kf的输入 //input_ikfom下面就两属性，加速度、陀螺仪角速度 ———— 均是三维向量
  //离散中值法，前向传播***************************//看下面，为了防止不越界，到end-1就结束了？ //最后一帧的imu没做传播了
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) //20ms中的每一个imu采样值都进行处理
  {
    auto &&head = *(it_imu);      //当前帧imu给到head
    auto &&tail = *(it_imu + 1);  //后一个给到tail
    
    //判断时间先后，不符合直接continue  
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;//tail(imu)的时间戳要小于 雷达的最后一次采样时间（雷达包住imu）
    
    //采用中值 把两帧平均输入到变量
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);      //角速度平均值，前一个的各种速度+后一个的各种速度 /2 = 平均速度
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);//加速度平均值，前一个的各种速度+后一个的各种速度 /2 = 平均速度

    //通过重力数值对加速度进行一下倍数的微调？
    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

   //如果imu开始时刻早于上次雷达的最晚时刻（因为将上次最后一个imu插入到下一个的开头了，所以会有这种情况）———————— 计算dt
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      //从上次雷达时刻末尾开始传播（而不从上组imu末尾帧算了，因为雷达末尾帧更接近），计算与此次imu结尾之间的时间差
      dt = tail->header.stamp.toSec() - last_lidar_end_time_; //dt时间增量 = imu尾部时间 - 雷达最后一次采样时间
    }
    else
    {
      //否则 两个imu时刻之间的间隔
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();//否则 时间增量 = 两imu的时间差
    }
    
    in.acc = acc_avr;           //in是kf的输入       //加速度平均值 给到kf的in变量         // in：此帧测量的imu数据
    in.gyro = angvel_avr;                           //角速度平均值 给到kf输入的陀螺仪gyro
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;       
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;       //diagonal.() 对角线
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;  //论文公式8
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);  //（为什么没有返回值，1、做的是地址传递！2、确实不用返回值，目的是计算P，在predict计算完p，p就在那里（P是工具包中的全局变量、不是局部变量））//预测噪声协方差Q 输入（时间增量dt，Q是、白噪声协方差，输入(加速度平均值、角速度平均值)）//这predict是用的IKFoM工具包 ************** //里面做了论文公式8
    //-------------------------------------------------------------//predict函数里面动了x_  下面直接get_x就能获得imu传播出来的状态？
    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;  //两帧之间的角速度 去掉 陀螺偏置
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba); //两帧平均加速度 去掉 加速度偏置 （更准确）
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];//在加上每个轴的重力   //加速度再 去掉 重力加速度
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;  //此组此帧imu的时间 离 此组雷达开始时间  = 偏移时间 存入到IMUpose中，反向传播时拿出来用
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));//预测的东西追加到IMUpose容器中（第一个参数是偏移时时间）
  }//开始进行下一采样的预测——>也就是20ms中的一次前向传播完毕------------------（但最后一帧imu还没做处理）----------------------------------------

  /*** calculated the pos and attitude prediction at the frame-end ***/ //计算帧尾位姿与姿态的预测值
  //判断雷达结束时间是否晚于imu
  //计算雷达末尾姿态
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;//若雷达最后时间大于imu最后时间，则note=1，否则为0 （条目运算符）
  dt = note * (pcl_end_time - imu_end_time);        //这个dt是激光雷达帧尾时间 - imu帧尾时间 //中间的预测值时间差用IMU自己的，结束时预测值时间差用IMU和激光雷达的//因为最后是以雷达，不要计算imu-imu了，imu-lidar-imu。算imu-lidar更准
  kf_state.predict(dt, Q, in);                      //传播出最后一帧的误差协方差
  
  imu_state = kf_state.get_x();                     //*******重要————前向传播的最终目的，得到imu的PVQ
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;              //参数更新  //记录IMU和激光雷达结束时刻的时间

  /*** undistort each lidar point (backward propagation) ***/  //反向传播去畸变 ****
  //反向去畸变时，需要知道20ms中，每个雷达采样时刻的IMU的预测位姿
  if (pcl_out.points.begin() == pcl_out.points.end()) return; //什么意思
  auto it_pcl = pcl_out.points.end() - 1;     //都是指针哦      //20ms点云的倒数第二个点（时间排序的）//地址传递？最终动的是pcl_out 在process外是feats_undistort

  //从后往前迭代 先从imu坐标下处理，再转回雷达坐标系——————imu和雷达*****************不懂，必要时可以再看
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) //循环两两帧  //第一层循环是不断将两imu做前移动***
  {
    auto head = it_kp - 1;//——————————————————IMU指针——————————————head-tail ； it_kp-1 - it_kp
    auto tail = it_kp;    //head 倒数第二，tail 倒数第一
    //这里的head和tali都是imuPose，auto也是imuPose的类型，下面有自定义的属性
    R_imu<<MAT_FROM_ARRAY(head->rot);         //旋转矩阵 Q
    vel_imu<<VEC_FROM_ARRAY(head->vel);       //速度 V
    pos_imu<<VEC_FROM_ARRAY(head->pos);       //位姿 P

    acc_imu<<VEC_FROM_ARRAY(tail->acc);       //加速度 ———— 使用的是tail的数据
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);    //角速度 ———— 使用的是tail的数据

   //   imu偏移时间（当前imu距离开始时的时间） ---时间就是dt时间
   //   ～～～～～～～～～～---
   //     |      |      |     | imu
   //   ——————————————————————— t
   //   | | | | | | | | | | | | lidar point (not points)
   //   ～～～～～～～～～～～～
   //         点的时间
    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)  //循环往前挪指针————指针指向此组雷达的每帧点云  //it_pcl为循环指针，在for循环前已经声明过了，不必初始化了
    {            //for的执行条件————点的时间大于了imu的偏移时间（这个点时间上一直在最近的imu后）                                //第二层for循环是把两个imu之间的点 依次做 坐标变换
      dt = it_pcl->curvature / double(1000) - head->offset_time;//

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      // 变换到“结束帧”，仅使用旋转
      // 注意：补偿方向与帧的移动方向相反
      // 所以如果我们想补偿时间戳i到e的一个点
      // P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei)    T_ei是在全局坐标系中的表示
      //按时间戳的差值 进行插值 //fast lio 论文公式（10）***
      M3D R_i(R_imu * Exp(angvel_avr, dt));//点所在时刻的旋转
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
                                                                                                          //lidar to imu 的rot     lidar to imu 的pos
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

//这个函数可以做三个事情，但一次只能做一个：1、meas.imu为空，直接退出函数 / 2、做imu初始化，求imu初始化值，偏置、重力、设定好初始PVQ(q是计算出来的) / 3、不是初始组的imu，做前向传播
void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;          //用于计时
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};               //若里面没有数据，则不用继续了
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)                           //若imu需要初始化，则进行初始化（默认需要）
  {
    /// The very first lidar frame  //imu初始化————获取一些参数（参数传递）//静止初始化10帧（2s之内完全可以达到效果）
    IMU_init(meas, kf_state, init_iter_num);    //（测量值，待优化状态，帧数）//主要是计算IMU相关初始值：偏置、重力、协方差等等**********************************

    imu_need_init_ = true;
    last_imu_   = meas.imu.back();              //20ms中最后的imu

    state_ikfom imu_state = kf_state.get_x();   //*******重要————初始化的最终目的，得到imu的PVQ————但是这里只是为了打印
    //（imu数据是由sync_packages()，把缓存器中的数据放进了观测量，）
    if (init_iter_num > MAX_INI_COUNT)      //如果 静止初始化函数 处理的帧数超过了10帧（或者说 缓存器传过来了超过10帧imu数据） ———— 否则还需要初始化
    {
      //初始化有效
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2); //pow(x,y) = x^y //————————————————👇cov_acc这有什么用呢 
      imu_need_init_ = false;     //更改标志位，不需要初始化了

      cov_acc = cov_acc_scale;    //比例因子误差给到协方差？ ———— 由参数文件传到这里的//——————☝cov_acc上面刚赋值完，下面又赋值————可以运行，打印出来看看
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; \nstate.bias_g_covarience: %.4f %.4f %.4f; \nacc covarience: %.8f %.8f %.8f; \ngry covarience: %.8f %.8f %.8f",\
                imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      //fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);  //输出日志
      std::cout << "bias_g = " << mean_gyr << std::endl; 
    }

    return; //出Process函数————此帧不用做其他处理了
  }

  //前向传播、反向传播 去畸变       //去畸变的点云指针
  UndistortPcl(meas, kf_state, *cur_pcl_un_);   //包括了IMU的前向传播和反向传播，利用IMU的数据将点云数据都投影到当前帧最后时刻*****************

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
