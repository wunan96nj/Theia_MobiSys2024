For the hardware requirement on the server side, we tested on a machine equipped with an Intel i7-11700 CPU, 32GB memory, and an NVIDIA GeForce RTX 3060 GPU. We recommend using a machine with a similar or better GPU to reproduce the performance reported in the paper. The OS of our machine is Ubuntu 18.04. Other Linux distributions may also work. For software, we utilize the pthread library, Draco library (V1.5.5), and CUDA toolkit (V11.0).

For the server part: 

1) Install CUDA Toolkit, pthread library, and Draco library (https://github.com/google/draco/blob/main/BUILDING.md#cmake-basics);
2) Clone the code in (https://github.com/wunan96nj/Theia_MobiSys2024_Server);
3) Compile the code with the command:
   ```
   nvcc -o server settings.cpp theia_server.cpp tools.cpp videocomm.cpp videodata.cpp gpu_pc_select.cu -O1 -lpthread -ldraco
   ```
   
To run Theia(L), we can use the command: 

```./server 2 1 -1```

To run Theia(L+S), we can use the command: 

```./server 2 2 -1```

To run Theia(L+A), we can use the command: 

```./server 2 3 -1```

To run Theia, we can use the command: 

```./server 2 4 -1```
