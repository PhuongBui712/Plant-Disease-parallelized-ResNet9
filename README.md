# ResNet-9 Parallelization Project for Plant Disease Detection

## Team Members

| Name | ID | 
|---|---|
| Trần Quốc Trung | 20127655 | 
| Bùi Tấn Phương | 20127597 |
| Trần Minh Hiệp | 20127023 | 

## Overview of the Problem

Plant diseases pose a significant threat to agriculture, leading to substantial losses in crop yield and quality. Early detection of plant diseases is crucial for mitigating these effects and ensuring food security. Traditional methods of disease detection often rely on manual inspection, which is time-consuming and prone to errors.

To address this problem, we utilized ResNet-9, a convolutional neural network (CNN) architecture, to automatically detect plant diseases from images. However, training a ResNet-9 model on large datasets can be computationally intensive, limiting its practicality for real-time applications.

The primary challenge was to optimize the performance of the ResNet-9 model through parallelization, enabling faster training and inference without compromising accuracy.

## Proposed Solution and Work Done

### Proposed Solution

Our solution involved parallelizing the ResNet-9 model to leverage the computational power of modern multi-core processors or GPUs. This approach aimed to reduce training time and enhance the model's ability to process large volumes of data, making it more suitable for real-time plant disease detection.

### Implementation Details

1. **Data Parallelism**: We implemented data parallelism by splitting the plant disease image dataset across multiple GPUs. Each GPU processed a subset of the data, and the gradients were aggregated across all GPUs before updating the model parameters.

2. **Model Parallelism**: We distributed different layers of the ResNet-9 architecture across multiple GPUs, optimizing resource utilization and reducing the time taken for forward and backward propagation, particularly for layers with high computational costs.

3. **Hybrid Parallelism**: To further optimize performance, we combined data and model parallelism, effectively scaling the model to handle larger datasets.

4. **Optimization Techniques**: We employed techniques such as gradient averaging, synchronous updates, and load balancing across GPUs to enhance the efficiency of parallelization.

### Results

- **Training Time Reduction**: The parallelized ResNet-9 model demonstrated a significant reduction in training time compared to the sequential implementation, particularly when trained on large-scale plant disease datasets.
- **Scalability**: The model scaled efficiently across multiple GPUs, achieving near-linear speedup as more computational resources were added.
- **Accuracy in Disease Detection**: The accuracy of the parallelized ResNet-9 model in detecting plant diseases was comparable to that of the original implementation, with no significant degradation observed.
- **Web Application**
![](/frontend/form.png)
![](/frontend/result.png)
### Challenges

- **Synchronization Overhead**: Managing synchronization across multiple GPUs introduced overhead, which we mitigated by optimizing communication patterns and using efficient gradient averaging techniques.
- **Memory Constraints**: Distributing the model across multiple GPUs required careful management of memory resources, particularly when dealing with large models or datasets.

## References

- [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep Residual Learning for Image Recognition." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
- [2] Mohanty, S. P., Hughes, D. P., & Salathé, M. "Using Deep Learning for Image-Based Plant Disease Detection." Frontiers in Plant Science, 7, 1419 (2016).
- [3] Zhang, Y., Hu, Q., Liu, Q., & Luo, Y. "ResNet and DenseNet for the Diagnosis of Plant Diseases with Attention Mechanism." IEEE Access, 7, 57832-57840 (2019).
- [4] Torchvision Models Documentation. "ResNet." [Link to documentation or relevant source]
