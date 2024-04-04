#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>




typedef unsigned char BYTE;
// Structure for a point in 3D space
struct Point {
  float x = 0.0f, y = 0.0f, z = 0.0f;
  int r = 0, g = 0, b = 0;
  float point_size = 0.0f;
  int count = 0;
};

struct Point_rgb {
  int r = 0, g = 0, b = 0;
  int count = 0;
  float depth_val = 1.0f;
};


// Structure for the gaze position and direction
struct Gaze {
  Point position;
  Point direction;
};

struct Vector3d {
  float x, y, z;
};

struct Id_Size {
  int id;
  float size;
};

void writeToLogFile_gpu(const std::string& logMessage) {
    std::ofstream logFile("../../Logs/log.txt", std::ios_base::app);

    if (logFile.is_open()) {
        logFile << logMessage << std::endl;
        logFile.close();
    } else {
        std::cerr << "Unable to open log file." << std::endl;
    }
}

uint64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

uint64_t NDKGetTime() {
    //struct timespec res;
    //clock_gettime(CLOCK_REALTIME, &res);
    //double t = res.tv_sec + (double)res.tv_nsec / 1e9f;

    //float t = FPlatformTime::Seconds()*1000;
    
    uint64_t t = timeSinceEpochMillisec();
    return t;
}

// // Kernel function to classify points in the point cloud
// // into different angle ranges relative to the gaze
// __global__ void classifyPoints(Point* points, Gaze gaze, int* classification, int numPoints) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   //printf("id %d", idx);

//   if (idx >= numPoints) {
//     return;
//   }

//   // Calculate the angle between the gaze direction and the current point
//   Point point = points[idx];
//   float dx = point.x - gaze.position.x;
//   float dy = point.y - gaze.position.y;
//   float dz = point.z - gaze.position.z;
//   float angle = acos(
//     (dx * gaze.direction.x + dy * gaze.direction.y + dz * gaze.direction.z) /
//     sqrt((dx * dx + dy * dy + dz * dz) * (gaze.direction.x * gaze.direction.x + gaze.direction.y * gaze.direction.y + gaze.direction.z * gaze.direction.z))
//   );

//   //printf("angle %f", angle);
//   // Classify the point based on its angle to the gaze
//   if (angle < 2.5 * M_PI / 180.0) {
//     classification[idx] = 1;
//   } else if (angle < 3.75 * M_PI / 180.0) {
//     classification[idx] = 2;
//   } else if (angle < 8.75 * M_PI / 180.0) {
//     classification[idx] = 3;
//   } else if (angle < 15.0 * M_PI / 180.0) {
//     classification[idx] = 4;
//   } else {
//     classification[idx] = 5;
//   }
// }




// Read the points from a PLY ASCII file
int readPLY(const char* filename, Point* points, int maxPoints) {
  // Open the file
  printf("open");
  FILE* file = fopen(filename, "r");
  if (!file) {
    return -1;
  }
  

  // Read the header to find the number of points
  char line[1024];
  int numPoints = 0;
  while (fgets(line, 1024, file)) {
    if (strncmp(line, "element vertex", 14) == 0) {
      sscanf(line + 14, "%d", &numPoints);
      if (numPoints > maxPoints) {
        numPoints = maxPoints;
      }
      break;
    }
  }

  while (fgets(line, 1024, file)) {
    if (strncmp(line, "end_header", 10) == 0) {
      break;
    }
  }

  // Read the points
  for (int i = 0; i < numPoints; i++) {
    //printf("good %d\n", i);
    if (!fgets(line, 1024, file)) {
      break;
    }
    sscanf(line, "%f %f %f %d %d %d", &points[i].x, &points[i].y, &points[i].z, &points[i].r, &points[i].g, &points[i].b);
    //printf("%f %f %f\n", points[i].x, points[i].y, points[i].z);

    // Apply transformations to the point
    points[i].x = points[i].x * 0.181731f - 39.1599f;
    points[i].y = points[i].y * 0.181731f + 3.75652f;
    points[i].z = points[i].z * 0.181731f - 46.6228f;
    points[i].x /= 100.0;
    points[i].y /= 100.0;
    points[i].z /= 100.0;
    
  }

  fclose(file);
  return numPoints;
}

__device__ void atomicMinn(float *const addr, const float val) {
  if (*addr <= val) return;

  unsigned int *const addr_as_ui = (unsigned int *)addr;
  unsigned int old = *addr_as_ui, assumed;
  do {
    assumed = old;
    if (__uint_as_float(assumed) <= val) break;
    old = atomicCAS(addr_as_ui, assumed, __float_as_uint(val));
  } while (assumed != old);
}

__global__ void calculate_depth(Point* points, Gaze gaze, float* depth_list, int numPoints) {
    // Get the index of the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the thread is within the range of points
    if (idx >= numPoints) return;



    // Get the current point
    Point point = points[idx];

    Point point_relative;
    point_relative.x = point.x - gaze.position.x;
    point_relative.y = point.y - gaze.position.y;
    point_relative.z = point.z - gaze.position.z;


    // calculate the component of point_relative vector along the gaze_direction
    float d = point_relative.x * gaze.direction.x + point_relative.y * gaze.direction.y + point_relative.z * gaze.direction.z;

    depth_list[idx] = d;
}

__global__ void logPolarTransformKernel(Point* points, int numPoints, float radius_min, int r_bins, int theta_bins, int phi_bins, Point_rgb* logPolarBuffer, Gaze gaze, Vector3d rightVector, Vector3d upVector, float rate_adapt)
{
    // Get the index of the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Make sure the thread is within the range of points
    if (idx >= numPoints) return;
    //printf("gaze: %f %f %f\n", gaze.position.x, gaze.position.y, gaze.position.z);



    // Get the current point
    Point point = points[idx];

    Point point_relative;
    point_relative.x = point.x - gaze.position.x;
    point_relative.y = point.y - gaze.position.y;
    point_relative.z = point.z - gaze.position.z;


    // calculate the component of point_relative vector along the gaze_direction
    float d = point_relative.x * gaze.direction.x + point_relative.y * gaze.direction.y + point_relative.z * gaze.direction.z;

    //float d_scaled = (d - d_min) / (d_max - d_min);

    // project point onto the plane
    Point point_on_plane;
    //point_on_plane.x = point_relative.x - d * gaze.direction.x;
    //point_on_plane.y = point_relative.y - d * gaze.direction.y;
    //point_on_plane.z = point_relative.z - d * gaze.direction.z;

    point_on_plane.x = point_relative.x;
    point_on_plane.y = point_relative.y;
    point_on_plane.z = point_relative.z;

    point_on_plane.x /= d;
    point_on_plane.y /= d;
    point_on_plane.z /= d;



    float x = point_on_plane.x * rightVector.x + point_on_plane.y * rightVector.y + point_on_plane.z * rightVector.z;
    //np.dot(x_axis, projected_point)

    float y = point_on_plane.x * upVector.x + point_on_plane.y * upVector.y + point_on_plane.z * upVector.z;

    float theta = atan2(y, x);
    float radius = max(sqrt(x*x + y*y), radius_min);
    // apply logarithmic scaling to radius
    float log_r = log(radius/radius_min);
		float base = (theta_bins + M_PI) / (theta_bins - M_PI);



    // determine buffer indices
    int r_index = (int)(log_r / log(base)/ rate_adapt) ;
    int theta_index = (int)(theta_bins * (theta + M_PI) / (2 * M_PI));
    //int phi_index = (int)((phi_bins-1) * d_scaled);
    int phi_index = (int)(0);
    // Check if indices are within range
    if (r_index >= r_bins || theta_index >= theta_bins || phi_index >= phi_bins) return;

    //logPolarBuffer[r_index * theta_bins * phi_bins + theta_index * phi_bins + phi_index].count = 1; 
    //logPolarBuffer[r_index * theta_bins * phi_bins + theta_index * phi_bins + phi_index].r = point.r; 
    //logPolarBuffer[r_index * theta_bins * phi_bins + theta_index * phi_bins + phi_index].g = point.g; 
    //logPolarBuffer[r_index * theta_bins * phi_bins + theta_index * phi_bins + phi_index].b = point.b; 
    int tmp_position = r_index * theta_bins * phi_bins + theta_index * phi_bins + phi_index;
    // Use atomic operation to update the log-polar buffer
    atomicMinn(&logPolarBuffer[tmp_position].depth_val, d);
    //atomicAdd(&logPolarBuffer[tmp_position].r, point.r);
    //atomicAdd(&logPolarBuffer[tmp_position].g, point.g);
    //atomicAdd(&logPolarBuffer[tmp_position].b, point.b);

    //printf("d_scaled: %f\n", d_scaled);

    if (d == logPolarBuffer[tmp_position].depth_val){
        logPolarBuffer[tmp_position].r = point.r;
        logPolarBuffer[tmp_position].g = point.g;
        logPolarBuffer[tmp_position].b = point.b;
    }


}

__global__ void setlogPolar(int numPoints, Point_rgb* logPolarBuffer)
{
    // Get the index of the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the thread is within the range of points
    if (idx >= numPoints) return;

    logPolarBuffer[idx].depth_val = 100.0f;
}

__global__ void logPolarToCartesianKernel(Point_rgb* logPolarBuffer, int r_bins, int theta_bins, int phi_bins, Point* points, int numPoints, float radius_min, float d_min, float d_max, Gaze gaze, Vector3d rightVector, Vector3d upVector, float rate_adapt)
{
    // Get the index of the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure the thread is within the range of points
    if (idx >= numPoints) return;

    // Compute the indices of the current point in the log-polar buffer
    int r_index = idx / (theta_bins * phi_bins);
    int theta_index = (idx / phi_bins) % theta_bins;
    int phi_index = idx % phi_bins;

    // Check if the point is valid
    if (logPolarBuffer[idx].depth_val == 100.0f){
        // Set the point coordinates and color
        points[idx].x = 0.0f;
        points[idx].y = 0.0f;
        points[idx].z = 0.0f;
        points[idx].r = 0;
        points[idx].g = 0;
        points[idx].b = 0;
        points[idx].count = 0;
        return;
    } 

    //Convert log-polar coordinates to spherical coordinates
    float base = (theta_bins + M_PI) / (theta_bins - M_PI);
    //float r = radius_min * std::pow(r_max / radius_min, (double)r_index / r_bins);
    float r = radius_min * std::pow(base, (float)r_index * rate_adapt);
    float theta = (float) (2 * M_PI) * (float(theta_index) / float(theta_bins)) - M_PI;
    //float phi = M_PI*phi_index / phi_bins;


    float x_back = r * cos(theta);
    float y_back = r * sin(theta);


          
    Vector3d new_dir;
    new_dir.x = gaze.direction.x + x_back * rightVector.x + y_back * upVector.x;
    new_dir.y = gaze.direction.y + x_back * rightVector.y + y_back * upVector.y;
    new_dir.z = gaze.direction.z + x_back * rightVector.z + y_back * upVector.z;

    float float_d_scale = logPolarBuffer[idx].depth_val;






    // convert d_scaled to d
    //float d_back = d_min + float_d_scale * (d_max - d_min);
    float d_back = float_d_scale;

    // convert back to Cartesian coordinates
    Point cartesian_point;
    cartesian_point.x = gaze.position.x + d_back * new_dir.x;
    cartesian_point.y = gaze.position.y + d_back * new_dir.y;
    cartesian_point.z = gaze.position.z + d_back * new_dir.z;


    // Compute the average color of the points in the cell
    int count = logPolarBuffer[idx].count;
    int red = logPolarBuffer[idx].r;
    int green = logPolarBuffer[idx].g;
    int blue = logPolarBuffer[idx].b;

    // Set the point coordinates and color
    points[idx].x = cartesian_point.x;
    points[idx].y = cartesian_point.y;
    points[idx].z = cartesian_point.z;
    points[idx].r = red;
    points[idx].g = green;
    points[idx].b = blue;
    points[idx].point_size = d_back * (r - radius_min * std::pow(base, (float)(r_index-1) * rate_adapt));
    points[idx].count = 1;


}

__device__ int distance(int x1, int y1, int z1, int x2, int y2, int z2) {
  return abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2);
}

__global__ void fill_empty_buffers_kernel(Point_rgb *temp_log_polar_buffer, Point_rgb *temp_log_polar_buffer_filled, int r_bins, int theta_bins, int phi_bins, float radius_min, float base, float point_size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total_size = r_bins * theta_bins * phi_bins;

  if (idx >= total_size) return;

  int r = idx / (theta_bins * phi_bins);
  int theta = (idx / phi_bins) % theta_bins;
  int phi = idx % phi_bins;
  
  
  Point_rgb *current_point = &temp_log_polar_buffer_filled[r * theta_bins * phi_bins + theta * phi_bins + phi];

  Point_rgb *nearest_point = nullptr;

  Point_rgb *current_buffer_point = &temp_log_polar_buffer[r * theta_bins * phi_bins + theta * phi_bins + phi];
  if (current_buffer_point->depth_val != 100.0f) {
    current_point->r = current_buffer_point->r;
    current_point->g = current_buffer_point->g;
    current_point->b = current_buffer_point->b;
    current_point->depth_val = current_buffer_point->depth_val;
    current_point->count = 1;
    return;
  }
  int min_distance = INT_MAX;
  int r_diff = 0;
  int theta_diff = 0;
  for (int temp_r_diff = 0; temp_r_diff < r_bins-r; temp_r_diff++){
    //float temp_distance = radius_min * std::pow(base, (float)temp_r_diff+r+1) - radius_min * std::pow(base, (float)temp_r_diff+r);
    float temp_distance = radius_min * std::pow(base, (float)(temp_r_diff+r)) - radius_min * std::pow(base, (float)(r));
    if (temp_distance > 1.25 * point_size){
      r_diff = temp_r_diff;
      break;
    }
  }
  float temp_theta_distance = 2 * M_PI * radius_min * std::pow(base, (float)r) / float(theta_bins);

  theta_diff = int(1.25 * point_size / float(temp_theta_distance));


  if (r_diff <= 1){
    r_diff = 0;
  }
  if (theta_diff <= 1){
    theta_diff = 0;
  }
  //int r2_max = (int)(log(tan(0.75 * M_PI / 180.0) / radius_min ) / log(base));
  //min(r_bins, r + r_diff) 
  for (int r2 = min(r_bins, r + r_diff); r2 >= max(0, r - r_diff); r2--) {
    // Nan Todo
    for (int theta2 = max(0, theta-theta_diff); theta2 < min(theta_bins, theta + theta_diff); ++theta2) {
      for (int phi2 = 0; phi2 < phi_bins; ++phi2) {
        if (r2 < r && nearest_point == nullptr){
          return;
        }
        Point_rgb *candidate_point = &temp_log_polar_buffer[r2 * theta_bins * phi_bins + theta2 * phi_bins + phi2];
        if (candidate_point->depth_val != 100.0f) {
          int dist = distance(r, theta, phi, r2, theta2, phi2);
          if (dist < min_distance) {
            min_distance = dist;
            nearest_point = candidate_point;
          }
        }
      }
    }
  }

  if (nearest_point != nullptr) {
    current_point->r = nearest_point->r;
    current_point->g = nearest_point->g;
    current_point->b = nearest_point->b;
    current_point->depth_val = nearest_point->depth_val;
    current_point->count = 1;
  }else{
    current_point->r = 0;
    current_point->g = 0;
    current_point->b = 0;
    current_point->depth_val = 100.0f;
    current_point->count = 0;
  }
}

void normalizeVector(Vector3d &v) {
    float magnitude = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

    if (magnitude != 0) {
        v.x /= magnitude;
        v.y /= magnitude;
        v.z /= magnitude;
    }
}

void crossProduct(Vector3d A, Vector3d B, Vector3d &C) {
   C.x = A.y * B.z - A.z * B.y;
   C.y = -(A.x * B.z - A.z * B.x);
   C.z = A.x * B.y - A.y * B.x;
   normalizeVector(C);
}

__global__ void processPointsKernel(BYTE *pp_pc, int nPoints, Point *points) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < nPoints) {
        //printf("index: %f\n", index);
        int pDataOffset = 15 + index * sizeof(short) * 3;
        int cDataOffset = 15 + nPoints * sizeof(short) * 3 + index * sizeof(char) * 3;
        short pp[3];
        memcpy(pp, pp_pc + pDataOffset, sizeof(short) * 3);
        //short* pp = (short*)(pp_pc + pDataOffset);
        float pp_float0 = (float)pp[0] / 1000.0f;
        float pp_float1 = (float)pp[1] / 1000.0f;
        float pp_float2 = (float)-pp[2] / 1000.0f;
        points[index].x = pp_float0;
        points[index].y = pp_float1;
        points[index].z = pp_float2;
        // points[index].x = (float)pp[0] / 1000.0f;
        // points[index].y = (float)pp[1] / 1000.0f;
        // points[index].z = -(float)pp[2] / 1000.0f;
        char pc[3];
        memcpy(pc, pp_pc + cDataOffset, sizeof(char) * 3);
        //char* pc = (char*)(pp_pc + cDataOffset);
        points[index].r = (int)pc[0];
        points[index].g = (int)pc[1];
        points[index].b = (int)pc[2];
    }
}

//extern "C"
//std::vector<int> get_fovea(BYTE * pp_pc, int dataLen, Gaze gaze, Point* d_points, int* d_classification){
extern "C"
void get_fovea(BYTE * pp_pc, int dataLen, Gaze gaze, Point* d_points, std::vector<Point> &ret_points_log_polar_inner, std::vector<Point> &ret_points_log_polar_outer, bool dynamic_skip, float rate_adapt, bool aug){
//std::vector<Point> get_fovea(BYTE * pp_pc, int dataLen, Gaze gaze, Point* d_points){
  //rate_adapt = 2.0f; //NANWU
  // if (rate_adapt > 1.5f){
  //   rate_adapt = 1.414f;
  // }
  uint64_t t0 = NDKGetTime();
  std::ostringstream oss;
  std::string logMessage;

  const int maxPoints = 1500000;
  //int* classification = (int*)malloc(maxPoints * sizeof(int));
  //printf("gaze: %f %f %f\n", gaze.position.x, gaze.position.y, gaze.position.z);


  int nPoints = (dataLen-15) / (sizeof(short)+sizeof(char)) / 3;
  //printf("nPoints: %d\n", nPoints);


  // char * pointBuf = new char[nPoints * sizeof(short) * 3];
  // char * colorBuf = new char[nPoints * sizeof(char) * 3];

  // int pDataSize = sizeof(short) * nPoints * 3;
  // memcpy(pointBuf, pp_pc+15, pDataSize);

  // int cDataSize = sizeof(char) * nPoints * 3;
  // memcpy(colorBuf, pp_pc+15+pDataSize, cDataSize);

  Point* points = (Point*)malloc(nPoints * sizeof(Point));
  // int pOffset = 0;
  // int cOffset = 0;


    // Allocate GPU memory
  BYTE *d_pp_pc;
  cudaMalloc((void**)&d_pp_pc, dataLen * sizeof(BYTE));

  // Copy input data to GPU memory
  cudaMemcpy(d_pp_pc, pp_pc, dataLen * sizeof(BYTE), cudaMemcpyHostToDevice);

  // Launch the CUDA kernel
  int blockSize1 = 1024;
  int gridSize = (nPoints + blockSize1 - 1) / blockSize1;
  processPointsKernel<<<gridSize, blockSize1>>>(d_pp_pc, nPoints, d_points);

  // Copy the result from GPU memory to host memory
  //cudaMemcpy(points, d_points, nPoints * sizeof(Point), cudaMemcpyDeviceToHost);


  // #pragma omp parallel for
  // for (int index = 0; index < nPoints; index++) {
  //   int pDataOffset = 15 + index * sizeof(short) * 3;
  //   int cDataOffset = 15 + nPoints * sizeof(short) * 3 + index * sizeof(char) * 3;
  //   // short * pp = (short *)(pointBuf + pOffset);
    
  //   short* pp = (short*)(pp_pc + pDataOffset);
  //   points[index].x = pp[0] / 1000.0f;
  //   points[index].y = pp[1] / 1000.0f;
  //   points[index].z = -pp[2] / 1000.0f;


  //   //pOffset += sizeof(short) * 3;

  //   // char * pc = (char *)(colorBuf + cOffset);
  //   char* pc = (char*)(pp_pc + cDataOffset);
  //   points[index].r = pc[0];
  //   points[index].g = pc[1];
  //   points[index].b = pc[2];


  //   //cOffset += sizeof(char) * 3;

  // }
  // delete [] pointBuf;
	// delete [] colorBuf;

  uint64_t t1 = NDKGetTime();
  //std::cout << "read data time used:" << t1-t0 << std::endl;

  
  oss << "GPU read data time used:" << t1 - t0 << std::endl;

  logMessage = oss.str();

  oss.str("");
  oss.clear();

  writeToLogFile_gpu(logMessage);

  const int r_bins = (int)(400/rate_adapt); //420
  const int theta_bins = (int)(285/rate_adapt); //315
  const int phi_bins = 1;
  float point_size = 0.0018;
  int max_points_save = r_bins * theta_bins * phi_bins;
  // Allocate memory on the GPU for the temp_log_polar_buffer
  Point_rgb* temp_log_polar_buffer = (Point_rgb*)malloc(sizeof(Point) * r_bins * theta_bins * phi_bins);
  Point* save_points = (Point*)malloc(max_points_save * sizeof(Point));
  //float* depth_list = (float*)malloc(maxPoints * sizeof(float));

  //float* d_depth_list;
  //cudaMalloc((void**)&d_depth_list, maxPoints * sizeof(float));


  // Copy the points and gaze data to the GPU
  //cudaMemcpy(d_points, points, nPoints * sizeof(Point), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(gaze, &gaze, sizeof(Gaze));

  // Launch the kernel to classify the points
  int threadsPerBlock = 1024;
  int blocks = (nPoints + threadsPerBlock - 1) / threadsPerBlock;

  //calculate_depth<<<blocks, threadsPerBlock>>>(d_points, gaze, d_depth_list, nPoints);
  //cudaMemcpy(depth_list, d_depth_list, nPoints * sizeof(int), cudaMemcpyDeviceToHost);






  //printf("number %d:", nPoints);
  uint64_t t2 = NDKGetTime();
  std::cout << "time used:" << t2-t1 << std::endl;



  // float d_min = std::numeric_limits<float>::max();
  // float d_max = std::numeric_limits<float>::min();




  Vector3d rightVector;
  crossProduct(Vector3d{gaze.direction.x, gaze.direction.y, gaze.direction.z}, Vector3d{0,1,0}, rightVector);
  //printf("rightVector: %f, %f, %f\n", rightVector.x, rightVector.y, rightVector.z);
  //projectionDirection.cross(upVector);
  Vector3d upVector;
  crossProduct(rightVector, Vector3d{gaze.direction.x, gaze.direction.y, gaze.direction.z}, upVector);
  //printf("upVector: %f, %f, %f\n", upVector.x, upVector.y, upVector.z);
  cudaMemcpyToSymbol(rightVector, &rightVector, sizeof(Vector3d));
  cudaMemcpyToSymbol(upVector, &upVector, sizeof(Vector3d));

  // for (int i = 0; i < nPoints; i++) {

  //   float d = depth_list[i];
    
  //   d_min = std::min(d_min, d);
  //   d_max = std::max(d_max, d);

  // }
  float radius_min = 0.00029088821f;
  //float r_max = 50.0f * d_min;
  //std::cout << "radius_min:" << radius_min  << std::endl;
  //std::cout << "d_max:" << d_max << "d_min:" << d_min  << std::endl;
  uint64_t t3 = NDKGetTime();
  //std::cout << "Depth min_max time used:" << t3-t1 << std::endl;

    Point_rgb* logPolarBuffer;
    cudaMalloc((void**)&logPolarBuffer, sizeof(Point_rgb) * max_points_save);


    Point* d_save_points;
    cudaMalloc(&d_save_points, max_points_save * sizeof(Point));

    t3 = NDKGetTime();
    // Set the block and grid size for the kernel
    int blockSize = 1024;
    int numBlocks = (nPoints + blockSize - 1) / blockSize;

    setlogPolar<<<numBlocks, blockSize>>>(max_points_save, logPolarBuffer);
    //cudaMemcpy(temp_log_polar_buffer, logPolarBuffer, sizeof(Point_rgb) * r_bins * theta_bins * phi_bins, cudaMemcpyDeviceToHost);
    // Launch the kernel
    //logPolarTransformKernel<<<numBlocks, blockSize>>>(d_points, nPoints, radius_min, r_max, r_bins, theta_bins, phi_bins, logPolarBuffer, d_min, d_max, gaze, rightVector, upVector);
    logPolarTransformKernel<<<numBlocks, blockSize>>>(d_points, nPoints, radius_min, r_bins, theta_bins, phi_bins, logPolarBuffer, gaze, rightVector, upVector, rate_adapt);

    
    // Wait for the kernel to finish
    //cudaDeviceSynchronize();

    cudaMemcpy(temp_log_polar_buffer, logPolarBuffer, sizeof(Point_rgb) * r_bins * theta_bins * phi_bins, cudaMemcpyDeviceToHost);


    float d_min = std::numeric_limits<float>::max();
    float d_max = std::numeric_limits<float>::min();

    float depth_sum = 0;
    int depth_point_size = 0;
    for (int i = 0; i < max_points_save; i++) {
      // convert point to coordinates relative to gaze position and direction
      // calculate the component of point_relative vector along the gaze_direction
      //float d = point_relative.x * gaze.direction.x + point_relative.y * gaze.direction.y + point_relative.z * gaze.direction.z;
      float d = temp_log_polar_buffer[i].depth_val;
      if (d == 100.0f)
          continue;
      // update d_min and d_max
      d_min = std::min(d_min, d);
      d_max = std::max(d_max, d);
      depth_sum += d;
      depth_point_size++;
    }
    float depth_mean = depth_sum / float(depth_point_size);
    
    std::cout << "d_max:" << d_max << "d_min:" << d_min  << std::endl;

    // for (int i = 0; i < max_points_save; i++) {
    //     // convert point to coordinates relative to gaze position and direction
    //     // calculate the component of point_relative vector along the gaze_direction

    //     float d = temp_log_polar_buffer[i].depth_val;
    //     if (d == 100.0f)
    //         continue;
    //     // update d_min and d_max
    //     d_min = std::min(d_min, d);
    //     d_max = std::max(d_max, d);

    // }
    // std::cout << "d_max:" << d_max << "d_min:" << d_min  << std::endl;

    float unit_point_size = float(point_size) / d_min;
    //std::cout << "unit_point_size:" << unit_point_size << std::endl;
    float base = (theta_bins + M_PI) / (theta_bins - M_PI);
    //int inpainting_row_max = (int)log(unit_point_size / radius_min / (base-1)) / log(base);
    //std::cout << "inpainting_row_max:" << inpainting_row_max << std::endl;
    //std::cout << "corresponding d:" << radius_min * std::pow(base, (float)r_bins) << std::endl;


    // float depth_stdev_sum = 0;
    // for (int i = 0; i < max_points_save; i++) {
    //   // convert point to coordinates relative to gaze position and direction
    //   // calculate the component of point_relative vector along the gaze_direction
    //   //float d = point_relative.x * gaze.direction.x + point_relative.y * gaze.direction.y + point_relative.z * gaze.direction.z;
    //   float d = temp_log_polar_buffer[i].depth_val;

    //   if (d != 100.0f){
    //       depth_stdev_sum += (d - depth_mean) * (d - depth_mean);
    //       // temp_log_polar_buffer[i].r = 0;
    //       // temp_log_polar_buffer[i].g = 0;
    //       // temp_log_polar_buffer[i].b = 0;
    //       // temp_log_polar_buffer[i].count = 0;
    //       // temp_log_polar_buffer[i].depth_val = 100.0f;
    //   } 
    // }
    // float depth_stdev = sqrt(depth_stdev_sum / float(depth_point_size));


    // for (int i = 0; i < max_points_save; i++) {
    //   // convert point to coordinates relative to gaze position and direction
    //   // calculate the component of point_relative vector along the gaze_direction
    //   //float d = point_relative.x * gaze.direction.x + point_relative.y * gaze.direction.y + point_relative.z * gaze.direction.z;
    //   float d = temp_log_polar_buffer[i].depth_val;

    //   if (d != 100.0f){
    //       float zScore = (d  - depth_mean) / depth_stdev;
    //       if (zScore > 3.0f){
    //         temp_log_polar_buffer[i].r = 0;
    //         temp_log_polar_buffer[i].g = 0;
    //         temp_log_polar_buffer[i].b = 0;
    //         temp_log_polar_buffer[i].count = 0;
    //         temp_log_polar_buffer[i].depth_val = 100.0f;
    //       }
    //   } 
    // }


    // printf("unit_point_size %f", unit_point_size);
    // std::vector<int> remove_idx;
    // for (int theta_index = 1; theta_index < theta_bins-1; theta_index++){
    //   bool find_outter = false;
    //   for (int r_index = r_bins - 1; r_index > 0; r_index--){
    //     int tmp_position = r_index * theta_bins+ theta_index;
    //     float temp_point_size = radius_min * (std::pow(base, (float)r_index * rate_adapt) - std::pow(base, (float)(r_index-1) * rate_adapt));
    //     if (temp_point_size <= unit_point_size){
    //       break;
    //     }
    //     if (temp_log_polar_buffer[tmp_position].depth_val == 100.0f){
    //       continue;
    //     }
    //     if (!find_outter){
    //       if (temp_point_size > unit_point_size){
    //         remove_idx.push_back(tmp_position);
    //         //temp_log_polar_buffer[tmp_position].depth_val = 100.0f;
    //         find_outter = true;
    //       }
    //     }
    //     if (temp_log_polar_buffer[tmp_position-1].depth_val == 100.0f || temp_log_polar_buffer[tmp_position+1].depth_val == 100.0f){
    //       remove_idx.push_back(tmp_position);
    //     }
    //   }
    // }
    // for(int tmp_position : remove_idx) {
    //   temp_log_polar_buffer[tmp_position].depth_val = 100.0f;
    // }

    
    // printf("number of removed: %d", (int)remove_idx.size());
    
    // cudaMemcpy(logPolarBuffer, temp_log_polar_buffer, max_points_save * sizeof(Point_rgb), cudaMemcpyHostToDevice);



    
    Point_rgb *d_temp_log_polar_buffer;
    Point_rgb *d_temp_log_polar_buffer_filled;
    size_t buffer_size = r_bins * theta_bins * phi_bins * sizeof(Point_rgb);
    cudaMalloc((void**)&d_temp_log_polar_buffer, buffer_size);
    cudaMalloc((void**)&d_temp_log_polar_buffer_filled, buffer_size);
    cudaMemcpy(d_temp_log_polar_buffer, temp_log_polar_buffer, max_points_save * sizeof(Point_rgb), cudaMemcpyHostToDevice);

    if (aug == true){
      fill_empty_buffers_kernel<<<numBlocks, blockSize>>>(d_temp_log_polar_buffer, d_temp_log_polar_buffer_filled, r_bins, theta_bins, phi_bins, radius_min, base, unit_point_size);
      logPolarToCartesianKernel<<<(max_points_save + blockSize - 1) / blockSize, blockSize>>>(d_temp_log_polar_buffer_filled, r_bins, theta_bins, phi_bins, d_save_points, max_points_save, radius_min, d_min, d_max, gaze, rightVector, upVector, rate_adapt);

    }else{
      logPolarToCartesianKernel<<<(max_points_save + blockSize - 1) / blockSize, blockSize>>>(d_temp_log_polar_buffer, r_bins, theta_bins, phi_bins, d_save_points, max_points_save, radius_min, d_min, d_max, gaze, rightVector, upVector, rate_adapt);

    }

    //logPolarToCartesianKernel<<<(max_points_save + blockSize - 1) / blockSize, blockSize>>>(logPolarBuffer, r_bins, theta_bins, phi_bins, d_save_points, max_points_save, radius_min, r_max, d_min, d_max, gaze, rightVector, upVector);
    //logPolarToCartesianKernel<<<(max_points_save + blockSize - 1) / blockSize, blockSize>>>(logPolarBuffer, r_bins, theta_bins, phi_bins, d_save_points, max_points_save, radius_min, d_min, d_max, gaze, rightVector, upVector);
    // if (rate_adapt <=1.5f){
    //   logPolarToCartesianKernel<<<(max_points_save + blockSize - 1) / blockSize, blockSize>>>(d_temp_log_polar_buffer_filled, r_bins, theta_bins, phi_bins, d_save_points, max_points_save, radius_min, d_min, d_max, gaze, rightVector, upVector, rate_adapt);

    // }else{
    //   logPolarToCartesianKernel<<<(max_points_save + blockSize - 1) / blockSize, blockSize>>>(d_temp_log_polar_buffer, r_bins, theta_bins, phi_bins, d_save_points, max_points_save, radius_min, d_min, d_max, gaze, rightVector, upVector, rate_adapt);

    // }


    // Copy the result back to the host
    cudaMemcpy(save_points, d_save_points, max_points_save * sizeof(Point), cudaMemcpyDeviceToHost);

    uint64_t t5 = NDKGetTime();
    //std::cout << "time used:" << t5-t3 << std::endl;
    //std::vector<Point> ret_points_log_polar;
    //std::cout << "Total time used before push_back:" << t5-t0 << std::endl;

    oss << "GPU Total time used before push_back:" << t5-t0 << std::endl;

    logMessage = oss.str();

    oss.str("");
    oss.clear();

    writeToLogFile_gpu(logMessage);


    //float base = (theta_bins + M_PI) / (theta_bins - M_PI);
    //, float rate_adapt
		int inner_point_idx = (int)(log(tan(0.75 * M_PI / 180.0) / radius_min ) / log(base) / rate_adapt) * theta_bins;
		std::cout << "inner_point_idx:" << inner_point_idx << std::endl;
    for (int i = 0; i < max_points_save; i++) {
      Point point = save_points[i];
      if (point.count == 0){
          continue;
      }
      float x = point.x;
      float y = point.y;
      float z = point.z;

      int r = point.r;
      int g = point.g;
      int b = point.b;

      //float temp_point_size = 1.5f * std::max(point.point_size, point_size);std::max(point.point_size, 0.0005f);
      float temp_point_size = std::min(0.018f*rate_adapt, 1.0f * point.point_size+0.0007f);//std::max(point.point_size+0.0007f, 0.0003f));//std::min(0.01f,1.3f * std::max(point.point_size, 0.0005f));
      Point temp_point{x, y, z, r, g, b, temp_point_size};

      if (dynamic_skip){
        if (i < inner_point_idx){
          ret_points_log_polar_inner.push_back(temp_point);
        }
      }else{
          ret_points_log_polar_inner.push_back(temp_point);
      }
			// if (i < inner_point_idx)
      //ret_points_log_polar_inner.push_back(temp_point);
			// else
			// ret_points_log_polar_outer.push_back(temp_point);

    }
    uint64_t t6 = NDKGetTime();
    //std::cout << "remove empty points time used:" << t6-t5 << std::endl;
    //std::cout << "Total time used:" << t6-t0 << std::endl;


    oss << "GPU Total time used in GPU:" << t6-t0 << std::endl;

    logMessage = oss.str();

    oss.str("");
    oss.clear();

    writeToLogFile_gpu(logMessage);

  printf("number selected %lu:", ret_points_log_polar_inner.size());
  free(save_points);
  free(points);
  free(temp_log_polar_buffer);
  //free(depth_list);
  //cudaFree(d_depth_list);
  cudaFree(logPolarBuffer);
  cudaFree(d_save_points);
  cudaFree(d_pp_pc);
  cudaFree(d_temp_log_polar_buffer);
  cudaFree(d_temp_log_polar_buffer_filled);

  //return ret_points_log_polar;
}

/*
int main() {
  // Set up the point cloud and gaze data
  const int maxPoints = 1500000;
  Point* points = (Point*)malloc(maxPoints * sizeof(Point));

  // Allocate memory on the GPU for the points and classification arrays
  Point* d_points;
  cudaMalloc((void**)&d_points, maxPoints * sizeof(Point));
  int* d_classification;
  cudaMalloc((void**)&d_classification, maxPoints * sizeof(int));

  Gaze gaze;
  gaze.position.x = 0.0;
  gaze.position.y = 1.5;
  gaze.position.z = -0.5;
  gaze.direction.x = 0.0;
  gaze.direction.y = 0.0;
  gaze.direction.z = 1.0;
  int* classification = (int*)malloc(maxPoints * sizeof(int));

  for (int index = 536; index < 836; index++){
      // Read a point cloud from a PLY file

    char buffer[100];
    sprintf(buffer, "Ply_original/soldier_vox10_0%d.ply", index);



    // Read the points from a PLY file
    int numPoints = readPLY(buffer, points, maxPoints);
    if (numPoints < 0) {
      printf("Error reading PLY file\n");
      return 1;
    }
    printf("Point number: %d\n", numPoints);

  
    uint64_t t1 = NDKGetTime();
    // Copy the points and gaze data to the GPU
    cudaMemcpy(d_points, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(gaze, &gaze, sizeof(Gaze));

    // Launch the kernel to classify the points
    int threadsPerBlock = 1024;
    int blocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    classifyPoints<<<blocks, threadsPerBlock>>>(d_points, gaze, d_classification, numPoints);

  
    // Copy the classification data back to the host
    cudaMemcpy(classification, d_classification, numPoints * sizeof(int), cudaMemcpyDeviceToHost);




    printf("number %d:", numPoints);
    uint64_t t2 = NDKGetTime();
    std::cout << "time used:" << t2-t1 << std::endl;
    // Do something with the selected points...

  }



  // Free GPU memory
  cudaFree(d_points);
  cudaFree(d_classification);

  free(points);


  return 0;
}
*/


/*

*/
