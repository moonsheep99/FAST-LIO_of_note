#include "preprocess.h"

#define RETURN0     0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess()
  :feature_enabled(0), lidar_type(AVIA), blind(0.01), point_filter_num(1)   //构造函数——初始化变量
{
  inf_bound = 10;
  N_SCANS   = 6;    //不单独服务特征提取
  SCAN_RATE = 10;   //不单独服务特征提取
  group_size = 8;
  disA = 0.01;
  disA = 0.1; // B?——————应该是disB，要不不出错？disB没赋值直接用
  p2l_ratio = 225;
  limit_maxmid =6.25;
  limit_midmin =6.25;
  limit_maxmin = 3.24;
  jump_up_limit = 170.0;
  jump_down_limit = 8.0;
  cos160 = 160.0;
  edgea = 2;
  edgeb = 0.1;
  smallp_intersect = 172.5;
  smallp_ratio = 1.2;
  given_offset_time = false;

  jump_up_limit = cos(jump_up_limit/180*M_PI);
  jump_down_limit = cos(jump_down_limit/180*M_PI);
  cos160 = cos(cos160/180*M_PI);
  smallp_intersect = cos(smallp_intersect/180*M_PI);
}

Preprocess::~Preprocess() {}

void Preprocess::set(bool feat_en, int lid_type, double bld, int pfilt_num) //没用上
{
  feature_enabled = feat_en;
  lidar_type = lid_type;
  blind = bld;
  point_filter_num = pfilt_num;
}

/****************************************************************************
 * @brief Livox激光雷达点云预处理函数
 * 
 * @param msg livox激光雷达点云数据，格式为livox_ros_driver::CustomMsg
 * @param pcl_out 输出处理后的点云数据，格式为pcl::PointCloud<pcl::PointXYZINormal>
 */
void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)  //构造函数 ———— livox雷达走这里
{  
  avia_handler(msg);
  *pcl_out = pl_surf; //所有点云中有效的点云 —— pl_surf,用地址的解引用，第二个参数的地址内的东西 已经变成pl_surf
}

void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)//构造函数 ———— oster、velondyne雷达走这里
{
  switch (time_unit)
  {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;   //velodyne
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }

  switch (lidar_type)
  {
  case OUST64:
    oust64_handler(msg);
    break;

  case VELO16:
    velodyne_handler(msg);
    break;
  
  default:
    printf("Error LiDAR Type");
    break;
  }
  *pcl_out = pl_surf;
}


/**
 * @brief 对Livox激光雷达点云数据进行预处理
 * 
 * @param msg livox激光雷达点云数据，格式为livox_ros_driver::CustomMsg
 */

//livox driver发布的customMsg格式的点云的格式
// Header header             # ROS standard message header
// uint64 timebase          # The time of first point       //此帧的第一个点的时间
// uint32 point_num      # Total number of pointclouds      //总点云数量
// uint8 lidar_id               # Lidar device id number    //雷达设备的id
// uint8[3] rsvd                 # Reserved use             //？
// CustomPoint[] points    # Pointcloud data                //点云数据（类似容器）

//最主要是做点云的类型转换（默认不用特征，否则在加点云特征提取、判断）
void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg) //msg没用ros的消息类型，用的livox_ros_driver中的消息类型  //一直是地址传递
{                                                                     //注：msg是const形式
  pl_surf.clear();  // 清除之前的点云缓存
  pl_corn.clear();  // PointCloudXYZI类型点云哦
  pl_full.clear();
  int plsize = msg->point_num;    //获取此帧点云的数量
  // cout<<"plsie: "<<plsize<<endl;

  pl_corn.reserve(plsize);    //reverse按照数量设置点云大小（分配内存空间）———— 点云PointCloudXYZI的本质还是容器，所以resize也是容器的成员函数
  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  for(int i=0; i<N_SCANS; i++)    //为什么不一起清，而是一条一条的清除  //livox avia有两种扫描模式，非重复性和重复性的（官网有动态扫描图）
  {
    pl_buff[i].clear();
    pl_buff[i].reserve(plsize);   //以点云总数量设置每条线的内存
  }
  uint valid_num = 0;             //无符号整数————有效点数量
  
  //特征提取（默认不进行特征提取）
  if (feature_enabled)
  {
    for(uint i=1; i<plsize; i++)//分别对每个点云进行处理  //for循环从i=1开始， 而不是0哦，因为下面用到了i-1的点
    {
      //只取线数在0_N_SCANS内 并且回波次序为0或1的点云（avia雷达最大能回波三条）//0x30是16进制的0 //（回波次序=0或1）且在线数内 的点
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        //把点云从livox数据类型转换为pcl的形式
        pl_full[i].x = msg->points[i].x;  //点云的三轴坐标
        pl_full[i].y = msg->points[i].y;
        pl_full[i].z = msg->points[i].z;
        pl_full[i].intensity = msg->points[i].reflectivity; //点云强度
        pl_full[i].curvature = msg->points[i].offset_time / float(1000000); //使用曲率作为每个点的时间

        bool is_new = false;  //  满文件夹里就这一个
        // 只有当前点和上一点的间距足够大（>1e-7时），才将当前点认为是有用的点，分别加入到对应line的pl_buff队列中 //任何一轴太近都不行
        if((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7) //1e-7 = 1*10的-7次方
            || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
            || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7))
        {
          pl_buff[msg->points[i].line].push_back(pl_full[i]); //追加到点云的缓存器中
          //缓存器中第i条线(点在livox类型下的第几条线，就也放在缓存器中的第几条线).追加进来（每个点依次追加）
        }
      }
    }       //把此组所有点云搬到了pcl的数据类型下——————————————————完毕
    //定义参数
    static int count = 0;
    static double time = 0.0;
    count ++;
    double t0 = omp_get_wtime();    //计时用

    // 对每个line中的激光雷达分别进行处理
    for(int j=0; j<N_SCANS; j++)
    {
      if(pl_buff[j].size() <= 5) continue;// 如果该line中的点云过小，则继续处理下一条line
      pcl::PointCloud<PointType> &pl = pl_buff[j];  //缓存器中第j条线的所有点 构成的点云  给到pl（取地址的方式）
      plsize = pl.size();                           //这条线上的点云数量
      vector<orgtype> &types = typess[j];           //感觉没啥用阿，创建一个容器来存储点的其他属性（点的类型、前后点的类型（正常？丢失？）、）
      types.clear();                                //清空
      types.resize(plsize);                         //分配内存
      plsize--;                                     //因为：比如有三个点，但是是0.1.2，所以先处理是2（3-1）
      for(uint i=0; i<plsize; i++)                  //还是上面例子，三个点，但是i < 2 ，所以第3个点没做，所以for循环完了把第三个点（最后一个点）做了
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y); //第j条线上的第i个点（xy上到雷达中心的距离）
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = sqrt(vx * vx + vy * vy + vz * vz);           //dista：当前点与后一个点的距离
      }
      types[plsize].range = sqrt(pl[plsize].x * pl[plsize].x + pl[plsize].y * pl[plsize].y);  //把最后一个点也做了（为什么不放在for循环里一起做了）
      //
      give_feature(pl, types);//做特征提取（这一条线上的点云，点们的其他属性）
      // pl_surf += pl;
    }
    time += omp_get_wtime() - t0;
    printf("Feature extraction time: %lf \n", time / count);
  }
  else      //若不做特征配准 ———— 走这里
  {
    for(uint i=1; i<plsize; i++)    // 分别对每个点云进行处理
    {
      // 只取线数在0~N_SCANS内并且回波次序为0或者1的点云  //知乎livox官方有说，//激光雷达实际使用经验
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        valid_num ++;// 有效的点云数+1
        // 等间隔降采样   point_filter_num = 2 
        if (valid_num % point_filter_num == 0)  // %：运算取余  //激光雷达实际使用经验————需不需要隔点取点
        {
          pl_full[i].x = msg->points[i].x;
          pl_full[i].y = msg->points[i].y;
          pl_full[i].z = msg->points[i].z;
          pl_full[i].intensity = msg->points[i].reflectivity;
          pl_full[i].curvature = msg->points[i].offset_time / float(1000000); // //使用曲率作为每个点的时间 curvature unit: ms  //偏移时间哦！***

          //只有当前点和上一点的间距足够大（>1e-7时）同时不在盲区里，才将当前点认为是有用的点，分别加入到对应line的pl_buff队列中 //任何一轴太近都不行
          if((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7) 
              || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
              || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7)  //这个10^-7 就是激光雷达实际使用经验，需要这个雷达实际使用，他的点是多密
              && (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z > (blind * blind)))
          {
            pl_surf.push_back(pl_full[i]);  //把有用的点追加到pl_surf点云容器中
          }
        }
      }
    }
  }
}

void Preprocess::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  if (feature_enabled)
  {
    for (int i = 0; i < N_SCANS; i++)
    {
      pl_buff[i].clear();
      pl_buff[i].reserve(plsize);
    }

    for (uint i = 0; i < plsize; i++)
    {
      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      if (range < (blind * blind)) continue;
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.3;
      if (yaw_angle >= 180.0)
        yaw_angle -= 360.0;
      if (yaw_angle <= -180.0)
        yaw_angle += 360.0;

      added_pt.curvature = pl_orig.points[i].t * time_unit_scale;
      if(pl_orig.points[i].ring < N_SCANS)
      {
        pl_buff[pl_orig.points[i].ring].push_back(added_pt);
      }
    }

    for (int j = 0; j < N_SCANS; j++)
    {
      PointCloudXYZI &pl = pl_buff[j];
      int linesize = pl.size();
      vector<orgtype> &types = typess[j];
      types.clear();
      types.resize(linesize);
      linesize--;
      for (uint i = 0; i < linesize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = vx * vx + vy * vy + vz * vz;
      }
      types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
      give_feature(pl, types);
    }
  }
  else
  {
    double time_stamp = msg->header.stamp.toSec();
    // cout << "===================================" << endl;
    // printf("Pt size = %d, N_SCANS = %d\r\n", plsize, N_SCANS);
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
      if (i % point_filter_num != 0) continue;

      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      
      if (range < (blind * blind)) continue;
      
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;//Normal结构体表示给定点所在样本曲面上的法线方向,以及对应曲率的测量值
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.curvature = pl_orig.points[i].t * time_unit_scale; // curvature unit: ms

      pl_surf.points.push_back(added_pt);
    }
  }
  // pub_func(pl_surf, pub_full, msg->header.stamp);
  // pub_func(pl_surf, pub_corn, msg->header.stamp);
}

void Preprocess::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    pl_surf.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * SCAN_RATE;       // scan angular velocity
    std::vector<bool> is_first(N_SCANS,true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);      // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0)//假如提供了每个点的时间戳
    {
      given_offset_time = true;   //标志为更新
    }
    else                          //没有提供每个点的时间戳
    {
      given_offset_time = false;
      double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;//寻找终止yaw
      double yaw_end  = yaw_first;
      int layer_first = pl_orig.points[0].ring;
      for (uint i = plsize - 1; i > 0; i--)
      {
        if (pl_orig.points[i].ring == layer_first)
        {
          yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
          break;
        }
      }
    }

    if(feature_enabled)
    {
      for (int i = 0; i < N_SCANS; i++)
      {
        pl_buff[i].clear();
        pl_buff[i].reserve(plsize);
      }
      
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        int layer  = pl_orig.points[i].ring;
        if (layer >= N_SCANS) continue;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale; // units: ms

        if (!given_offset_time)//假如没有提供时间pianyi ?
        {
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;
          if (is_first[layer])//给每条线第一个赋予yaw角 时间归0
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }
          //对每个点计算时间
          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }
          //新得到的点的时间不能早于之前的最后一个点的时间
          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        pl_buff[layer].points.push_back(added_pt);
      }

      for (int j = 0; j < N_SCANS; j++)
      {
        PointCloudXYZI &pl = pl_buff[j];
        int linesize = pl.size();
        if (linesize < 2) continue;   //点太少就跳过
        vector<orgtype> &types = typess[j];
        types.clear();
        types.resize(linesize);
        linesize--;
        for (uint i = 0; i < linesize; i++)
        {
          types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
          vx = pl[i].x - pl[i + 1].x;
          vy = pl[i].y - pl[i + 1].y;
          vz = pl[i].z - pl[i + 1].z;
          types[i].dista = vx * vx + vy * vy + vz * vz;
        }
        types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
        give_feature(pl, types);
      }
    }
    else
    {
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;
        
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale;  // curvature unit: ms // cout<<added_pt.curvature<<endl;

        if (!given_offset_time)
        {
          int layer = pl_orig.points[i].ring;
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

          if (is_first[layer])
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }

          // compute offset time
          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }

          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
          if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
          {
            pl_surf.points.push_back(added_pt);
          }
        }
      }
    }
}

/**
 * @brief 对于每条line的点云提取特征
 * 
 * @param pl  pcl格式的点云 输入进来一条扫描线上的点
 * @param types  点云的其他属性
 */
void Preprocess::give_feature(pcl::PointCloud<PointType> &pl, vector<orgtype> &types) //均是地址传递
{
  int plsize = pl.size();
  int plsize2;
  if(plsize == 0)
  {
    printf("something wrong\n");
    return;
  }
  uint head = 0;

  while(types[head].range < blind)  //从第一个点开始，看看第几个点开始，点才不再盲区中，下面从这个点开始（前面盲区中的点直接排除）
  {
    head++;
  }

  // Surf
  //x = a？b：c  条件运算符：若a为true，则x = b，否则x = c
  plsize2 = (plsize > group_size) ? (plsize - group_size) : 0;//判断此条线上的点够不够8个点 不够的话就=0，不做处理了

  Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero());
  Eigen::Vector3d last_direct(Eigen::Vector3d::Zero());

  uint i_nex = 0, i2;
  uint last_i = 0; uint last_i_nex = 0;
  int last_state = 0;//为1代表上个状态为平面 否则为0
  int plane_type;

  //判断面点
  for(uint i=head; i<plsize2; i++)  //此条线大于8个的前提下（是多少个就做多少循环（依次处理））//从第一个不在盲区中的点开始，把此条线上的所有点都做了
  {
    if(types[i].range < blind)  // 点在xy平面上是在盲区中，不做这个点，直接下次循环
    {
      continue;
    }

    i2 = i;
    plane_type = plane_judge(pl, types, i, i_nex, curr_direct);//类型返回0 1 2  //参数：此条线上的点云，点的其他类型，第几个点，i_nex，法向量
    
    if(plane_type == 1)//返回1一般默认是平面
    {
      //设置确定的平面点和可能的平面点
      for(uint j=i; j<=i_nex; j++)
      { 
        if(j!=i && j!=i_nex)
        {
          //把起始点和终止点之间的所有点设置为确定的平面点
          types[j].ftype = Real_Plane;
        }
        else
        {
          //把起始点和终止点设置为可能的平面点
          types[j].ftype = Poss_Plane;
        }
      }
      //最开始last_state=0直接跳过
      //之后last_state=1
      //如果之前状态是平面则判断当前点是处于两平面边缘的点还是较为平坦的平面的点 
      if(last_state==1 && last_direct.norm()>0.1)
      {
        double mod = last_direct.transpose() * curr_direct;
        if(mod>-0.707 && mod<0.707)
        {
          //两个面法向量夹角在45度和135度之间 认为是两平面边缘上的点
          types[i].ftype = Edge_Plane;
        }
        else
        {
          //否则认为是真正的平面点
          types[i].ftype = Real_Plane;
        }
      }
      
      i = i_nex - 1;
      last_state = 1;
    }
    else
    {
      i = i_nex;
      last_state = 0;
    }

    last_i = i2;
    last_i_nex = i_nex;
    last_direct = curr_direct;
  }
  //判断边缘点
  plsize2 = plsize > 3 ? plsize - 3 : 0;
  for(uint i=head+3; i<plsize2; i++)
  {
    //点不能在盲区 或者 点必须属于正常点(?)和可能的平面点
    if(types[i].range<blind || types[i].ftype>=Real_Plane)
    {
      continue;
    }
    
    //该点与前后点的距离不能挨的太近
    if(types[i-1].dista<1e-16 || types[i].dista<1e-16)
    {
      continue;
    }

    Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z);
    Eigen::Vector3d vecs[2];

    for(int j=0; j<2; j++)
    {
      int m = -1;
      if(j == 1)
      {
        m = 1;
      }
      //若当前的前/后一个点在盲区内（4m)
      if(types[i+m].range < blind)
      {
        if(types[i].range > inf_bound)//若其大于10m
        {
          types[i].edj[j] = Nr_inf;//赋予该点Nr_inf(跳变较远)
        }
        else
        {
          types[i].edj[j] = Nr_blind;//赋予该点Nr_blind(在盲区)
        }
        continue;
      }

      vecs[j] = Eigen::Vector3d(pl[i+m].x, pl[i+m].y, pl[i+m].z);
      vecs[j] = vecs[j] - vec_a;//前/后点指向当前点的向量
      //若雷达坐标系原点为O 当前点为A 前/后一点为M和N
      //则下面OA点乘MA/（|OA|*|MA|）
      //得到的是cos角OAM的大小

      types[i].angle[j] = vec_a.dot(vecs[j]) / vec_a.norm() / vecs[j].norm();
      if(types[i].angle[j] < jump_up_limit)//cos(170)
      {
        types[i].edj[j] = Nr_180;//M在OA延长线上
      }
      else if(types[i].angle[j] > jump_down_limit)
      {
        types[i].edj[j] = Nr_zero;//M在OA上
      }
    }

    types[i].intersect = vecs[Prev].dot(vecs[Next]) / vecs[Prev].norm() / vecs[Next].norm();

    //前一个点是正常点(?) && 下一个点在激光线上 && 当前点与后一个点的距离大于0.0225m && 当前点与后一个点的距离大于当前点与前一个点距离的四倍
    //这种边缘点像是7字形这种的边缘？
    if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_zero && types[i].dista>0.0225 && types[i].dista>4*types[i-1].dista)
    {
      if(types[i].intersect > cos160)//角MAN要小于160度 不然就平行于激光了
      {
        if(edge_jump_judge(pl, types, i, Prev))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    //与上面类似
    //前一个点在激光束上 && 后一个点正常 && 前一个点与当前点的距离大于0.0225m && 前一个点与当前点的距离大于当前点与后一个点距离的四倍
    else if(types[i].edj[Prev]==Nr_zero && types[i].edj[Next]== Nr_nor && types[i-1].dista>0.0225 && types[i-1].dista>4*types[i].dista)
    {
      if(types[i].intersect > cos160)
      {
        if(edge_jump_judge(pl, types, i, Next))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    //前面的是正常点 && (当前点到中心距离>10m并且后点在盲区)
    else if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_inf)
    {
      if(edge_jump_judge(pl, types, i, Prev))
      {
        types[i].ftype = Edge_Jump;
      }
    }
    //(当前点到中心距离>10m并且前点在盲区) && 后面的是正常点
    else if(types[i].edj[Prev]==Nr_inf && types[i].edj[Next]==Nr_nor)
    {
      if(edge_jump_judge(pl, types, i, Next))
      {
        types[i].ftype = Edge_Jump;
      }
     
    }
    //前后点都不是正常点
    else if(types[i].edj[Prev]>Nr_nor && types[i].edj[Next]>Nr_nor)
    {
      if(types[i].ftype == Nor)
      {
        types[i].ftype = Wire;  //程序中应该没使用 当成空间中的小线段或者无用点了
      }
    }
  }

  plsize2 = plsize-1;
  double ratio;
  for(uint i=head+1; i<plsize2; i++)
  {
    if(types[i].range<blind || types[i-1].range<blind || types[i+1].range<blind)
    {
      continue;
    }

    if(types[i-1].dista<1e-8 || types[i].dista<1e-8)
    {
      continue;
    }

    if(types[i].ftype == Nor)
    {
      if(types[i-1].dista > types[i].dista)
      {
        ratio = types[i-1].dista / types[i].dista;
      }
      else
      {
        ratio = types[i].dista / types[i-1].dista;
      }

      if(types[i].intersect<smallp_intersect && ratio < smallp_ratio)
      {
        if(types[i-1].ftype == Nor)
        {
          types[i-1].ftype = Real_Plane;
        }
        if(types[i+1].ftype == Nor)
        {
          types[i+1].ftype = Real_Plane;
        }
        types[i].ftype = Real_Plane;
      }
    }
  }

  int last_surface = -1;
  for(uint j=head; j<plsize; j++)
  {
    if(types[j].ftype==Poss_Plane || types[j].ftype==Real_Plane)
    {
      if(last_surface == -1)
      {
        last_surface = j;
      }
    
      if(j == uint(last_surface+point_filter_num-1))
      {
        PointType ap;
        ap.x = pl[j].x;
        ap.y = pl[j].y;
        ap.z = pl[j].z;
        ap.intensity = pl[j].intensity;
        ap.curvature = pl[j].curvature;
        pl_surf.push_back(ap);

        last_surface = -1;
      }
    }
    else
    {
      if(types[j].ftype==Edge_Jump || types[j].ftype==Edge_Plane)
      {
        pl_corn.push_back(pl[j]);
      }
      if(last_surface != -1)
      {
        PointType ap;
        for(uint k=last_surface; k<j; k++)
        {
          ap.x += pl[k].x;
          ap.y += pl[k].y;
          ap.z += pl[k].z;
          ap.intensity += pl[k].intensity;
          ap.curvature += pl[k].curvature;
        }
        ap.x /= (j-last_surface);
        ap.y /= (j-last_surface);
        ap.z /= (j-last_surface);
        ap.intensity /= (j-last_surface);
        ap.curvature /= (j-last_surface);
        pl_surf.push_back(ap);
      }
      last_surface = -1;
    }
  }
}

//发布livox点云
void Preprocess::pub_func(PointCloudXYZI &pl, const ros::Time &ct)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "livox";
  output.header.stamp = ct;
}

//平面判断
int Preprocess::plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct)
{
  double group_dis = disA*types[i_cur].range + disB;
  group_dis = group_dis * group_dis;
  double two_dis;
  vector<double> disarr;//前后点距离数组
  disarr.reserve(20);
  //距离小 点与点之间较近 先取够8个点
  for(i_nex=i_cur; i_nex<i_cur+group_size; i_nex++)
  {
    if(types[i_nex].range < blind)
    {
      curr_direct.setZero();//距离雷达原点太小了将法向量设置为零向量
      return 2;
    }
    disarr.push_back(types[i_nex].dista);//存储当前点与后一个点的距离
  }
  
  for(;;)  //距离远 点与点距离较远 看看后续的点有没有满足条件的
  {
    if((i_cur >= pl.size()) || (i_nex >= pl.size())) break;

    if(types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;
    }

    //最后的i_nex点到i_cur点的距离
    vx = pl[i_nex].x - pl[i_cur].x;
    vy = pl[i_nex].y - pl[i_cur].y;
    vz = pl[i_nex].z - pl[i_cur].z;
    two_dis = vx*vx + vy*vy + vz*vz;
    if(two_dis >= group_dis)//距离i_cur点太远了就直接break
    {
      break;
    }
    disarr.push_back(types[i_nex].dista);
    i_nex++;
  }

  double leng_wid = 0;
  double v1[3], v2[3];
  for(uint j=i_cur+1; j<i_nex; j++)
  {
    //假设i_cur点为A j点为B i_nex点为C
    //向量AB
    if((j >= pl.size()) || (i_cur >= pl.size())) break;
    v1[0] = pl[j].x - pl[i_cur].x;
    v1[1] = pl[j].y - pl[i_cur].y;
    v1[2] = pl[j].z - pl[i_cur].z;

    //向量AB叉乘向量AC   
    v2[0] = v1[1]*vz - vy*v1[2];
    v2[1] = v1[2]*vx - v1[0]*vz;
    v2[2] = v1[0]*vy - vx*v1[1];

    //物理意义是组成的ABC组成的平行四边形面积的平方(为|AC|*h，其中为B到线AC的距离)
    double lw = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
    if(lw > leng_wid)
    {
      leng_wid = lw;//寻找最大面积的平方(也就是寻找距离AC最远的B)
    }
  }

  //|AC|*|AC|/(|AC|*|AC|*h*h)<225
  //也就是h>1/15 B点到AC的距离要大于0.06667m
  //太近了不好拟合一个平面
  if((two_dis*two_dis/leng_wid) < p2l_ratio)
  {
    curr_direct.setZero();//太近了法向量直接设置为0
    return 0;
  }

  //把两点之间的距离 按从大到小排个顺序
  uint disarrsize = disarr.size();
  for(uint j=0; j<disarrsize-1; j++)
  {
    for(uint k=j+1; k<disarrsize; k++)
    {
      if(disarr[j] < disarr[k])
      {
        leng_wid = disarr[j];
        disarr[j] = disarr[k];
        disarr[k] = leng_wid;
      }
    }
  }

  //这里可能还是觉得太近了
  if(disarr[disarr.size()-2] < 1e-16)
  {
    curr_direct.setZero();
    return 0;
  }

  //目前还不太懂为什么给AVIA单独弄了一种，其实我觉得全都可以采用上面这种判定方式（虽然FAST-LIO默认不开特征提取）
  if(lidar_type==AVIA)
  {
    double dismax_mid = disarr[0]/disarr[disarrsize/2];
    double dismid_min = disarr[disarrsize/2]/disarr[disarrsize-2];

    //点与点之间距离变化太大的时候 可能与激光束是平行的 就也舍弃了
    if(dismax_mid>=limit_maxmid || dismid_min>=limit_midmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  else
  {
    double dismax_min = disarr[0] / disarr[disarrsize-2];
    if(dismax_min >= limit_maxmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  
  curr_direct << vx, vy, vz;
  curr_direct.normalize();//法向量归一化
  return 1;
}

//边缘判断
bool Preprocess::edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir)
{
  if(nor_dir == 0)
  {
    if(types[i-1].range<blind || types[i-2].range<blind)//前两个点不能在盲区
    {
      return false;
    }
  }
  else if(nor_dir == 1)
  {
    if(types[i+1].range<blind || types[i+2].range<blind)//后两个点不能在盲区
    {
      return false;
    }
  }

  //下面分别对i-2 i-1和i i+1两种情况时点与点间距进行了判断
  double d1 = types[i+nor_dir-1].dista;
  double d2 = types[i+3*nor_dir-2].dista;
  double d;

  //将大小间距进行调换 大在前 小在后
  if(d1<d2)
  {
    d = d1;
    d1 = d2;
    d2 = d;
  }

  d1 = sqrt(d1);
  d2 = sqrt(d2);

  //edgea=2 edgeb=0.1
  if(d1>edgea*d2 || (d1-d2)>edgeb)
  {
    //假如间距太大 可能怕是被遮挡？就不把它当作线点
    return false;
  }
  
  return true;
}
