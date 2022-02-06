# To-do

- [x] 2 **Prep** - Add transformation to PDAQ.transform_points to account for camera fish-eye lensing.

- [x] 0 **Prep** - Get lens specs for more accurate transformations.

- [x] 5 **Detect** - Improve object detection.

- [x] 5 **Detect** - Use LiDAR point cloud neural netowrk to get 3D bounding boxes.

- [ ] 2 **Track** - Write TrackableObject.is_collided function.  This is the centroid tracking function that determines if the current object is the same as the passed in parameters given some margin of error.  Object bases its own position on either its latest centroid OR its predicted position.

- [ ] 3 **Track** - Write TrackableObject.predict_position.  Predicts the object's position in the next frame based on current position, velocity, and acceleration.

- [ ] 3 **Track/Log** - Write TrackableObject.calculate_attributes function.  Calculates velocity and acceleration for an object and appends to the object's current velocity and acceleration logs.

- [x] 4 **General** - Add timing adjustment to alignment tool so it can be used to determine timing offsets.

- [ ] 4 **Optimize** - Multi-thread (multiprocess) video object detection.

- [x] 3 **Optimize** - Move transform_points function to Camera class.  Consolodate transform calculations to increase efficiency.

# Setting up Open3D-ML
We used Open3D-ML for our point cloud 3D object detection. To have an environment similar to mine, first install open3D with the following command:

```bash 
pip install open3d
```

Unfortunately, open3D is only available on Python 3.6, 3.7, and 3.8; I used Python 3.6. Once installed, you need to have cuda 11.0. You can install it from the toolkit website: https://developer.nvidia.com/cuda-toolkit-archive. For the last command, replace "cuda" with "cuda-11-0". Next, perform the post installation actions: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions. Once you logout of your computer (or source your .bashrc), you can verify by running:

```bash
nvcc --version
```

Which should give your cuda 11.0. Finally, you need to install pytorch with a specific compiled build from here: https://github.com/isl-org/open3d_downloads/releases/tag/torch1.7.1. I used torch-1.7.1-cp36-cp36m-linux_x86_64.whl. You can install with the following command:

```bash
pip install torch-1.7.1-cp36-cp36m-linux_x86_64.whl 
```
