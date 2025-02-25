# Using C++ Interface to Perform Inference

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/runtime_cpp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

> MindSpore has unified the inference API. If you want to continue to use the MindSpore Lite independent API for inference, you can refer to the [document](https://www.mindspore.cn/lite/docs/en/r1.3/use/runtime_cpp.html).

## Overview

After the model is converted into a `.ms` model by using the MindSpore Lite model conversion tool, the inference process can be performed in Runtime. For details, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html). This tutorial describes how to use the [C++ API](https://www.mindspore.cn/lite/api/en/master/index.html) to perform inference.

To use the MindSpore Lite inference framework, perform the following steps:

1. Read the model: Read the `.ms` model file converted by the [model conversion tool](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html) from the file system.
2. Create and configure context: Create and configure [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) to save some basic configuration parameters required to build and execute the model.
3. Create, load and build a model: Use [Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build) of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) to create and build the model, and configure the [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) obtained in the previous step. In the model loading phase, the file cache is parsed into a runtime model. In the model building phase, subgraph partition, operator selection and scheduling are performed, which will take a long time. Therefore, it is recommended that the model should be created once, built once, and performed for multiple times.
4. Input data: Before the model is executed, data needs to be filled in the `Input Tensor`.
5. Perform inference: Use [Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict) of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) to perform model inference.
6. Obtain the output: After the model execution is complete, you can obtain the inference result by `Output Tensor`.
7. Release the memory: If the MindSpore Lite inference framework is not required, release the created [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model).

![img](../images/lite_runtime.png)

> For details about the calling process of MindSpore Lite inference, see [Simplified MindSpore Lite C++ Demo](https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start_cpp.html).

## Model Reading

When MindSpore Lite is used for model inference, read the `.ms` model file converted by using the model conversion tool from the file system and store it in the memory buffer. For details, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html).

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L332) demonstrates how to load a MindSpore Lite model from the file system:

```cpp
// Read model file.
size_t size = 0;
char *model_buf = ReadFile(model_path, &size);
if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
}
```

## Creating and Configuring Context

The context saves some basic configuration parameters required to build and execute the model. If you use `new` to create a [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) and do not need it any more, use `delete` to release it. Generally, the [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) is released after the [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) is created and built.

The default backend of MindSpore Lite is CPU. After Context is created, call [MutableDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mutabledeviceinfo) to return list of backend device information. Add the default [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_CPUDeviceInfo.html#class-cpudeviceinfo) to the list.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L250) demonstrates how to create a context, configure the default CPU backend, and enable CPU float16 inference.

```cpp
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
}
auto &device_list = context->MutableDeviceInfo();
auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
if (cpu_device_info == nullptr) {
  std::cerr << "New CPUDeviceInfo failed." << std::endl;
}
// CPU use float16 operator as priority.
cpu_device_info->SetEnableFP16(true);
device_list.push_back(cpu_device_info);
```

> `MutableDeviceInfo` supports multiple DeviceInfos, including [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_CPUDeviceInfo.html#class-cpudeviceinfo), [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_GPUDeviceInfo.html#class-gpudeviceinfo), [KirinNPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_KirinNPUDeviceInfo.html#class-kirinnpudeviceinfo). The device number limit is 3. During the inference, the operator will choose device in order.
>
> Float16 takes effect only when the CPU is under the ARM v8.2 architecture. Other models and x86 platforms that do not supported Float16 will be automatically rolled back to Float32.
>
> For the iOS platform, only the CPU backend is supported, and Float16 is temporarily not supported.

The advanced interfaces contained in [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) are defined as follows:

### Configuring the Number of Threads

Use [SetThreadNum](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setthreadnum) of [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) to configure the number of threads:

```cpp
// Configure the number of worker threads in the thread pool to 2, including the main thread.
context->SetThreadNum(2);
```

### Configuring the Thread Affinity

Use [SetThreadAffinity](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setthreadaffinity) of [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) to configure the thread affinity. If the parameter is `int mode`, configure the binding strategy. The effective value is 0-2, 0 means no core binding by default, 1 means preferential binding to large cores, and 2 means preferential binding to small cores. If the parameter is `const std::vector<int> &core_list`, configure the binding core list. When configuring at the same time, the core_list is effective, but the mode is not effective.

```cpp
// Configure the thread to be bound to the big core first.
// Valid value: 0: no affinities, 1: big cores first, 2: little cores first
context->SetThreadAffinity(1);
```

### Configuring the Parallelization

Use [SetEnableParallel](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setenableparallel) of [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) to configure whether to support parallelism when executing inference:

```cpp
// Configure the inference supports parallel.
context->SetEnableParallel(true);
```

### Configuring the GPU Backend

If the backend to be executed is GPUs, you need to set [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_GPUDeviceInfo.html#class-gpudeviceinfo) as the first choice. It is suggested to set [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_CPUDeviceInfo.html#class-cpudeviceinfo) as the second choice, to ensure model inference. Use `SetEnableFP16` to enable GPU Float16 inference.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L114) demonstrates how to create the CPU and GPU heterogeneous inference backend and how to enable Float16 inference for the GPU.

```cpp
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
}
auto &device_list = context->MutableDeviceInfo();

// Set GPU device first, make GPU preferred backend.
auto gpu_device_info = std::make_shared<mindspore::GPUDeviceInfo>();
if (gpu_device_info == nullptr) {
  std::cerr << "New GPUDeviceInfo failed." << std::endl;
}
// GPU use float16 operator as priority.
gpu_device_info->SetEnableFP16(true);
// Set VNIDIA device id, only valid when GPU backend is TensorRT.
gpu_device_info->SetDeviceID(0);
// The GPU device context needs to be push_back into device_list to work.
device_list.push_back(gpu_device_info);

// Set CPU device after GPU as second choice.
auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
if (cpu_device_info == nullptr) {
  std::cerr << "New CPUDeviceInfo failed." << std::endl;
}
// CPU use float16 operator as priority.
cpu_device_info->SetEnableFP16(true);
device_list.push_back(cpu_device_info);
```

> The current GPU backend distinguishes `arm64`and `x86_64`platforms.
>
> - On `arm64`, the backend of GPU is based on OpenCL. GPUs of Mali and Adreno are supported. The OpenCL version is 2.0.
>
> The configuration is as follows:
>
> CL_TARGET_OPENCL_VERSION=200
>
> CL_HPP_TARGET_OPENCL_VERSION=120
>
> CL_HPP_MINIMUM_OPENCL_VERSION=120
>
> - On `x86_64`, the backend of GPU is based on TensorRT. The TensorRT version is 6.0.1.5.
>
> Whether the attribute `SetEnableFP16` can be set successfully depends on the [CUDA computer capability](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix) of the current device.
>
> The attribute `SetDeviceID` only valid for TensorRT, used to specify the NVIDIA device ID.

### Configuring the NPU Backend

If the backend to be executed is NPUs, you need to set [KirinNPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_KirinNPUDeviceInfo.html#class-kirinnpudeviceinfo) as the first choice. It is suggested to set [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_CPUDeviceInfo.html#class-cpudeviceinfo) as the second choice, to ensure model inference. Use `SetFrequency` to set npu frequency.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L127) shows how to create the CPU and NPU heterogeneous inference backend and set the NPU frequency to 3. It can be set to 1 (low power consumption), 2 (balanced), 3 (high performance), 4 (extreme performance).

```cpp
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
}
auto &device_list = context->MutableDeviceInfo();

// Set NPU device first, make NPU preferred backend.
auto npu_device_info = std::make_shared<mindspore::KirinNPUDeviceInfo>();
if (npu_device_info == nullptr) {
  std::cerr << "New KirinNPUDeviceInfo failed." << std::endl;
}
// NPU set frequency to be 3.
npu_device_info->SetFrequency(3);
// The NPU device context needs to be push_back into device_list to work.
device_list.push_back(npu_device_info);

// Set CPU device after NPU as second choice.
auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
if (cpu_device_info == nullptr) {
  std::cerr << "New CPUDeviceInfo failed." << std::endl;
}
// CPU use float16 operator as priority.
cpu_device_info->SetEnableFP16(true);
device_list.push_back(cpu_device_info);
```

### Configuring the NNIE Backend

When the backend that needs to be executed is the heterogeneous inference based on CPU and NNIE, you only need to create the Context according to the configuration method of [CPU Backend](#creating-and-configuring-context) without specifying a provider.

### Configuring the ASCEND Backend

If the backend to be executed is Ascend(only support ASCEND310), you need to set [Ascend310DeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_AscendDeviceInfo.html#class-documentation) as the first choice. It is suggested to set [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_CPUDeviceInfo.html#class-cpudeviceinfo) as the second choice, to ensure model inference. Use `SetDeviceID` to set ascend device id.

The following sample code shows how to create the CPU and ASCEND heterogeneous inference backend and set ascend device id to 0.

```cpp
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
}
auto &device_list = context->MutableDeviceInfo();

// Set Ascend310 device first, make Ascend310 preferred backend.
auto ascend_device_info = std::make_shared<mindspore::Ascend310DeviceInfo>();
if (ascend_device_info == nullptr) {
  std::cerr << "New Ascend310DeviceInfo failed." << std::endl;
}
// Ascend310 set device id to be 0.
ascend_device_info->SetDeviceId(0);
// The ascend310 device context needs to be push_back into device_list to work.
device_list.push_back(ascend_device_info);

// Set CPU device after Ascend310 as second choice.
auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
if (cpu_device_info == nullptr) {
  std::cerr << "New CPUDeviceInfo failed." << std::endl;
}
// CPU use float16 operator as priority.
cpu_device_info->SetEnableFP16(true);
device_list.push_back(cpu_device_info);
```

## Model Creating Loading and Building

When MindSpore Lite is used for inference, [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) is the main entry for inference. You can use [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) to load, build and execute model. Use the [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) created in the previous step to call the [Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build) of Model to load and build the runtime model.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L265) demonstrates how to create, load and build a model:

```cpp
// Create model
auto model = new (std::nothrow) mindspore::Model();
if (model == nullptr) {
  std::cerr << "New Model failed." << std::endl;
}
// Build model
auto build_ret = model->Build(model_buf, size, mindspore::kMindIR, context);
delete[](model_buf);
// After the model is built, the Context can be released.
...
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model failed." << std::endl;
}
```

> After the [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) is loaded and built, the [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) created in the previous step can be released.

## Inputting Data

Before executing a model, obtain the input [MSTensor](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_MSTensor.html) of the model and copy the input data to the input Tensor using `memcpy`. In addition, you can use the [DataSize](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#datasize) method to obtain the size of the data to be filled in to the tensor, use the [DataType](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#datatype) method to obtain the data type of the tensor, and use the [MutableData](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mutabledata) method to obtain the writable data pointer.

MindSpore Lite provides two methods to obtain the input tensor of a model.

1. Use the [GetInputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputbytensorname) method to obtain the input tensor based on the name. The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L154) demonstrates how to call `GetInputByTensorName` to obtain the input tensor and fill in data.

   ```cpp
   // Pre-processing of input data, convert input data format to NHWC.
   ...
   // Assume that the model has only one input tensor named graph_input-173.
   auto in_tensor = model->GetInputByTensorName("graph_input-173");
   if (in_tensor.impl() == nullptr) {
       std::cerr << "Input tensor is nullptr" << std::endl;
   }
   auto input_data = in_tensor.MutableData();
   if (input_data == nullptr) {
       std::cerr << "MallocData for inTensor failed." << std::endl;
   }
   memcpy(in_data, input_buf, data_size);
   // Users need to free input_buf.
   ```

2. Use the [GetInputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputs) method to directly obtain the vectors of all model input tensors. The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L137) demonstrates how to call `GetInputs` to obtain the input tensor and fill in data.

   ```cpp
   // Pre-processing of input data, convert input data format to NHWC.
   ...
   // Assume we have created a Model instance named model.
   auto inputs = model->GetInputs();
   // Assume that the model has only one input tensor.
   auto in_tensor = inputs.front();
   if (in_tensor == nullptr) {
       std::cerr << "Input tensor is nullptr" << std::endl;
   }
   auto *in_data = in_tensor.MutableData();
   if (in_data == nullptr) {
       std::cerr << "Data of in_tensor is nullptr" << std::endl;
   }
   memcpy(in_data, input_buf, data_size);
   // Users need to free input_buf.
   ```

> The data layout in the input tensor of the MindSpore Lite model must be `NHWC`. For more information about data pre-processing, see step 2 in [Writing On-Device Inference Code](https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start.html#writing-on-device-inference-code) in Android Application Development Based on JNI Interface to convert the input image into the Tensor format of the MindSpore model.
>
> [GetInputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputs) and [GetInputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputbytensorname) methods return data that do not need to be released by users.

## Executing Inference

Call the [Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict) function of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) for model inference.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L355) demonstrates how to call `Predict` to perform inference.

```cpp
auto inputs = model->GetInputs();
auto outputs = model->GetOutputs();
auto predict_ret = model->Predict(inputs, &outputs);
if (predict_ret != mindspore::kSuccess) {
  std::cerr << "Predict error " << predict_ret << std::endl;
}
```

## Obtaining Output

After performing inference, MindSpore Lite can obtain the inference result of the model. MindSpore Lite provides three methods to obtain the output [MSTensor](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_MSTensor.html) of a model.

1. Use the [GetOutputsByNodeName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputsbynodename) method to obtain the vector of the tensor connected to the model output tensor based on the name of the model output node. The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L170) demonstrates how to call `GetOutputsByNodeName` to obtain the output tensor.

   ```cpp
   // Assume we have created a Model instance named model before.
   // Assume that model has a output node named Softmax-65.
   auto output_vec = model->GetOutputsByNodeName("Softmax-65");
   // Assume that output node named Default/Sigmoid-op204 has only one output tensor.
   auto out_tensor = output_vec.front();
   if (out_tensor == nullptr) {
       std::cerr << "Output tensor is nullptr" << std::endl;
   }
   // Post-processing your result data.
   ```

2. Use the [GetOutputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputbytensorname) method to obtain the corresponding model output tensor based on the name of the model output tensor. The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L200) demonstrates how to call `GetOutputsByTensorName` to obtain the output tensor.

   ```cpp
   // Assume we have created a Model instance named model.
   // We can use GetOutputTensorNames method to get all name of output tensor of model which is in order.
   auto tensor_names = model->GetOutputTensorNames();
   // Assume we have created a Model instance named model before.
   for (auto tensor_name : tensor_names) {
       auto out_tensor = model->GetOutputByTensorName(tensor_name);
       if (out_tensor == nullptr) {
           std::cerr << "Output tensor is nullptr" << std::endl;
       }
       // Post-processing the result data.
   }
   ```

3. Use the [GetOutputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputs) method to directly obtain the names of all model output tensors vector. The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L226) demonstrates how to call `GetOutputs` to obtain the output tensor.

   ```cpp
   // Assume we have created a Model instance named model.
   auto out_tensors = model->GetOutputs();
   for (auto out_tensor : out_tensors) {
       // Post-processing the result data.
   }
   ```

> The data returned by the [GetOutputsByNodeName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputsbynodename), [GetOutputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputbytensorname), and [GetOutputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputs) methods does not need to be released by the user.

## Releasing Memory

If the MindSpore Lite inference framework is not required, you need to release the created Model. The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L370) demonstrates how to release the memory before the program ends.

```cpp
// Delete model.
// Assume that the variable of Model * is named model.
delete model;
```

## Advanced Usage

### Resizing the Input Dimension

When MindSpore Lite is used for inference, if the input shape needs to be resized, you can call the [Resize](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#resize) API of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) to resize the shape of the input tensor after a model is created and built.

> Some networks do not support variable dimensions. As a result, an error message is displayed and the model exits unexpectedly. For example, the model contains the MatMul operator, one input tensor of the MatMul operator is the weight, and the other input tensor is the input. If a variable dimension API is called, the input tensor does not match the shape of the weight tensor. As a result, the inference fails.
>
> When the GPU backend is TensorRT, Resize only valid at dims NHW for NHWC format inputs, resize shape value should not be larger than the model inputs.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L321) demonstrates how to perform Resize on the input tensor of MindSpore Lite:

```cpp
// Assume we have created a Model instance named model.
auto inputs = model->GetInputs();
std::vector<int64_t> resize_shape = {1, 128, 128, 3};
// Assume the model has only one input,resize input shape to [1, 128, 128, 3]
std::vector<std::vector<int64_t>> new_shapes;
new_shapes.push_back(resize_shape);
return model->Resize(inputs, new_shapes);
```

### Parallel Models

MindSpore Lite provides multi model concurrent reasoning interface Realize the concurrent reasoning of multiple models. The internal thread pools of multiple models of concurrent reasoning are independent of each other, and the constant weights are shared with each other, so as to reduce the memory use. Multi model concurrent reasoning now supports CPU and GPU backend.

>For a quick understanding of the complete calling process of MindSpore Lite executing concurrent reasoning, please refer to [Experience C++ Minimalist Concurrent Reasoning Demo](https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start_server_inference_cpp.html).

1. Create configuration

    The configuration item will save some basic configuration parameters required for concurrent reasoning, which are used to guide the number of concurrent models, model compilation and model execution.

    The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_server_inference_cpp/main.cc#L135) demonstrates how to create a RunnerConfig and configure the number of workers for concurrent reasoning:

    ```cpp
    // Init Context
    auto context = std::make_shared<mindspore::Context>();
    if (context == nullptr) {
        std::cerr << "New context failed." << std::endl;
    }
    auto &device_list = context->MutableDeviceInfo();
    auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
    if (cpu_device_info == nullptr) {
      std::cerr << "New CPUDeviceInfo failed." << std::endl;
    }
    // CPU use float16 operator as priority.
    cpu_device_info->SetEnableFP16(true);
    device_list.push_back(cpu_device_info);
    // Init RunnerConfig
    auto runner_config = std::make_shared<RunnerConfig>();
    runner_config->context = context;
    runner_config->workers_num = 2;
    ```

2. Initialization

    When using MindSpore Lite to execute concurrent reasoning, ModelParallelRunner is the main entry of concurrent reasoning. Through ModelParallelRunner, you can initialize and execute concurrent reasoning. Use the RunnerConfig created in the previous step and call the init interface of ModelParallelRunner to initialize ModelParallelRunner.

    The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_server_inference_cpp/main.cc#L135) demonstrates how to call Predict to execute reasoning:

    ```cpp
      auto predict_ret = model_runner->Predict(inputs, &outputs);
      if (predict_ret != mindspore::kSuccess) {
        delete model_runner;
        std::cerr << "Predict error " << predict_ret << std::endl;
        return -1;
      }
    ```

3. Execute concurrent reasoning

    MindSpore Lite calls the predict interface of ModelParallelRunner for model concurrent reasoning.

    The following [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_server_inference_cpp/main.cc#L187) Demonstrates how calls Predict to execute reasoning.

    ```cpp
      auto predict_ret = model_runner->Predict(inputs, &outputs);
      if (predict_ret != mindspore::kSuccess) {
        delete model_runner;
        std::cerr << "Predict error " << predict_ret << std::endl;
        return -1;
      }
    ```

4. Memory release

    When you do not need to use the MindSpore Lite reasoning framework, you need to release the created ModelParallelRunner. The following [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_server_inference_cpp/main.cc#L207) Demonstrates how to free memory before the end of the program.

    ```cpp
    // Delete ModelParallelRunner.
    delete model_runner;
    ```

### Mixed Precision Inference

MindSpore Lite supports mixed precision inference.
Users can set mixed precision information by calling the [LoadConfig](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html) API of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) after a model is created and before built.
The example of the config file is as follows:

```text
[execution_plan]
op_name1=data_type:float16
op_name2=data_type:float32
```

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L470) demonstrates how to infer model in the mixed precision:

```cpp
Status load_config_ret = model->LoadConfig(config_file_path);
if (load_config_ret != mindspore::kSuccess) {
  std::cerr << "Model load config error " << load_config_ret << std::endl;
  return -1;
}

Status build_ret = model->Build(graph_cell, context);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Model build error " << build_ret << std::endl;
  return -1;
}

auto inputs = model->GetInputs();
auto outputs = model->GetOutputs();
Status predict_ret = model->Predict(inputs, &outputs);
if (predict_ret != mindspore::kSuccess) {
  std::cerr << "Model predict error " << predict_ret << std::endl;
  return -1;
}
```

### Multiple Heterogeneous Devices Inference

MindSpore Lite supports multiple heterogeneous devices inference.
Users can set multiple heterogeneous devices inference information by set multiple [DeviceInfoContext](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_DeviceInfoContext.html) in [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html).

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L546) demonstrates how to infer the model in multiple heterogeneous devices:

```cpp
mindspore::Context context;
// enable NPU CPU GPU in inference. NPU is preferentially used, then the CPU, and GPU get the lowest priority.
context.MutableDeviceInfo().push_back(std::make_shared<mindspore::KirinNPUDeviceInfo>());
context.MutableDeviceInfo().push_back(std::make_shared<mindspore::CPUDeviceInfo>());
context.MutableDeviceInfo().push_back(std::make_shared<mindspore::GPUDeviceInfo>());

Status build_ret = model->Build(graph_cell, context);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Model build error " << build_ret << std::endl;
  return -1;
}

auto inputs = model->GetInputs();
auto outputs = model->GetOutputs();
Status predict_ret = model->Predict(inputs, &outputs);
if (predict_ret != mindspore::kSuccess) {
  std::cerr << "Model predict error " << predict_ret << std::endl;
  return -1;
}
```

### OpenGL Texture Data Input

MindSpore Lite supports OpenGL texture input, performs end-to-end GPU isomorphic inference, and the inference result is returned as OpenGL texture data. This function needs to be configured in the Context during use, and OpenGL texture data is bound to it during inference. These two processes.

1. Configured Context

    The user needs to set the SetEnableGLTexture property in dev.gpu_device_info_ in Context to true, and configure the user's current OpenGL EGLContext and EGLDisplay through the SetGLContext interface and SetGLDisplay interface respectively.

    ```cpp
    const std::shared_ptr<mindspore::Context> context;
    auto &device_list = context->MutableDeviceInfo();

    // 1. Set EnableGLTexture true
    gpu_device_info->SetEnableGLTexture(true);

    // 2. Set GLContext
    EGLContext *gl_context = new (std::nothrow) EGLContext();
    if (gl_context == nullptr) {
      MS_LOG(ERROR) << "new EGLContext failed";
      return RET_ERROR;
    } else {
      *gl_context = eglGetCurrentContext();
    }
    gpu_device_info->SetGLContext(gl_context);

    // 3. Set GLDisplay
    EGLDisplay *gl_display = new (std::nothrow) EGLDisplay();
    if (gl_display == nullptr) {
      MS_LOG(ERROR) << "new EGLDisplay failed";
      return RET_ERROR;
    } else {
      *gl_display = eglGetCurrentDisplay();
    }
    gpu_device_info->SetGLDisplay(gl_display);
    ```

 2. Bind OpenGL Texture Data

    After the model is compiled and before the model runs, the user needs to call BindGLTexture2DMemory(const std::map<std::string, GLuint> &inputGlTexture, std::map<std::string, GLuint> *outputGLTexture;) function to bind the input Output texture, instead of the original input data step. Because MindSpore Lite itself does not allocate OpenGL memory, the user is required to create the input and output texture memory in advance according to the tensor size of the model input and output, and the texture memory corresponding to the texture ID Bind to the input and output of the model, the sample code is as follows:

    ```cpp
    std::map<std::string, GLuint> input_gl_texture;
    std::map<std::string, GLuint> output_gl_texture;

    ... // Write OpenGL Texture data(GLuint) into input_gl_texture and output_gl_texture

    // Bind texture data with input and output tensors
    auto status = ms_model_.BindGLTexture2DMemory(input_gl_texture, &output_gl_texture);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "BindGLTexture2DMemory failed";
      return RET_ERROR;
    }
    return RET_OK;
    ```

    In std::map<std::string, GLuint> input_gl_texture, the key is the model input tensor name, and the value is the corresponding GLuint texture; std::map<std::string, GLuint> the key in the output_gl_texture variable is the model output tensor name, Value is the corresponding GLuint texture. The model input and output tensor name can be obtained through the tensor.Name() interface. The sample code is as follows:

    ```cpp
    std::vector<mindspore::MSTensor> inputs;
    vector<GLuint> inTextureIDs;
    for (auto i; i < inputs.size(); i++) {
      inputGlTexture.insert(std::pair<std::string, GLuint>(inputs.at(i).Name(), inTextureIDs.at(i));
    }

    std::vector<mindspore::MSTensor> outputs;
    vector<GLuint> outTextureIDs;
    for (auto i; i < inputs.size(); i++) {
      outputGlTexture.insert(std::pair<std::string, GLuint>(inputs.at(i).Name(), outTextureIDs.at(i));
    }
    ```

3. Predict

    After the binding is completed, you can directly call the Predict interface of ms_model_ for inference. The model output will be copied to the memory corresponding to the bound output texture ID, and the user can obtain the inference result from the outputs.

    ```cpp
    std::vector<MSTensor> outputs;
    auto ret = ms_model_.Predict(ms_inputs_for_api_, &outputs, ms_before_call_back_, ms_after_call_back_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Inference error ";
      std::cerr << "Inference error " << std::endl;
      return RET_ERROR;
    }
    ```

### Sharing a Memory Pool

If there are multiple [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model), you can configure the same [Allocator](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Allocator.html#class-allocator) in [DeviceInfoContext](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_DeviceInfoContext.html#class-deviceinfocontext) to share the memory pool and reduce the memory size during running. The maximum memory size of the memory pool is `3 GB`, and the maximum memory size allocated each time is `2 GB`.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L546) demonstrates how to share the memory pool between two models:

```cpp
auto context1 = std::make_shared<mindspore::Context>();
if (context1 == nullptr) {
  std::cerr << "New context failed." << std::endl;
}
auto &device_list1 = context1->MutableDeviceInfo();
auto device_info1 = CreateCPUDeviceInfo();
if (device_info1 == nullptr) {
  std::cerr << "Create CPUDeviceInfo failed." << std::endl;
}
device_list1.push_back(device_info1);

auto model1 = new (std::nothrow) mindspore::Model();
if (model1 == nullptr) {
  std::cerr << "New Model failed." << std::endl;
}
auto build_ret = model1->Build(model_buf, size, mindspore::kMindIR, context1);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model failed." << std::endl;
}

auto context2 = std::make_shared<mindspore::Context>();
if (context2 == nullptr) {
  std::cerr << "New context failed." << std::endl;
}
auto &device_list2 = context2->MutableDeviceInfo();
auto device_info2 = CreateCPUDeviceInfo();
if (device_info2 == nullptr) {
  std::cerr << "Create CPUDeviceInfo failed." << std::endl;
}
// Use the same allocator to share the memory pool.
device_info2->SetAllocator(device_info1->GetAllocator());
device_list2.push_back(device_info2);

auto model2 = new (std::nothrow) mindspore::Model();
if (model2 == nullptr) {
  std::cerr << "New Model failed." << std::endl;
}
build_ret = model2->Build(model_buf, size, mindspore::kMindIR, context2);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model failed." << std::endl;
}
```

### Calling Back a Model During the Running Process

MindSpore Lite can pass two [MSKernelCallBack](https://www.mindspore.cn/lite/api/en/master/generate/typedef_mindspore_MSKernelCallBack-1.html) function pointers to [Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict) to call back a model for inference. Compared with common graph execution, callback execution can obtain additional information during the running process to help developers analyze performance and debug bugs. Additional information includes:

- Name of the running node
- Input and output tensors before the current node is inferred
- Input and output tensors after the current node is inferred

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L672) demonstrates how to define two callback functions as the pre-callback pointer and post-callback pointer and pass them to the Predict API for callback inference.

```cpp
// Definition of callback function before forwarding operator.
auto before_call_back = [](const std::vector<mindspore::MSTensor> &before_inputs,
                           const std::vector<mindspore::MSTensor> &before_outputs,
                           const mindspore::MSCallBackParam &call_param) {
  std::cout << "Before forwarding " << call_param.node_name_ << " " << call_param.node_type_ << std::endl;
  return true;
};
// Definition of callback function after forwarding operator.
auto after_call_back = [](const std::vector<mindspore::MSTensor> &after_inputs,
                          const std::vector<mindspore::MSTensor> &after_outputs,
                          const mindspore::MSCallBackParam &call_param) {
  std::cout << "After forwarding " << call_param.node_name_ << " " << call_param.node_type_ << std::endl;
  return true;
};

auto inputs = model->GetInputs();
auto outputs = model->GetOutputs();
auto predict_ret = model->Predict(inputs, &outputs, before_call_back, after_call_back);
if (predict_ret != mindspore::kSuccess) {
  std::cerr << "Predict error " << predict_ret << std::endl;
}
```

### Separating Graph Loading and Model Build

Use [Load](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#load) of [Serialization](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Serialization.html#class-serialization) to load [Graph](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Graph.html#class-graph) and use [Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build) of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) to build the model.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L282) demonstrates how to load graph and build model separately.

```cpp
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
  std::cerr << "New context failed." << std::endl;
}
auto &device_list = context->MutableDeviceInfo();
auto cpu_device_info = CreateCPUDeviceInfo();
if (cpu_device_info == nullptr) {
  std::cerr << "Create CPUDeviceInfo failed." << std::endl;
}
device_list.push_back(cpu_device_info);

// Load graph
mindspore::Graph graph;
auto load_ret = mindspore::Serialization::Load(model_buf, size, mindspore::kMindIR, &graph);
if (load_ret != mindspore::kSuccess) {
  std::cerr << "Load graph failed." << std::endl;
}

// Create model
auto model = new (std::nothrow) mindspore::Model();
if (model == nullptr) {
  std::cerr << "New Model failed." << std::endl;
  return nullptr;
}
// Build model
mindspore::GraphCell graph_cell(graph);
auto build_ret = model->Build(graph_cell, context);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model failed." << std::endl;
}
```

### Viewing Logs

If an exception occurs during inference, you can view logs to locate the fault. For the Android platform, use the `Logcat` command line to view the MindSpore Lite inference log information and use `MS_LITE` to filter the log information.

```bash
logcat -s "MS_LITE"
```

> For the iOS platform, does not support viewing logs temporarily.

### Obtaining the Version Number

MindSpore Lite provides the [Version](https://www.mindspore.cn/lite/api/en/master/generate/function_mindspore_Version-1.html#function-documentation) method to obtain the version number, which is included in the `include/api/types.h` header file. You can call this method to obtain the version number of MindSpore Lite.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L717) demonstrates how to obtain the version number of MindSpore Lite:

```cpp
#include "include/api/types.h"
std::string version = mindspore::Version();
```

### Extension Usage

In this chapter, we will show the users an example of extending MindSpore Lite inference, covering the whole process of creation and registration of custom operator. The example will help the users understand the extension usage as soon as possible. The chapter takes a simple model that consists of a single operator `Add` as an example. The code related to the example can be obtained from the directory [mindspore/lite/examples/runtime_extend](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/runtime_extend).

The chapter only provides instruction in the Linux System.

#### Operator InferShape Extension

The users need to inherit the basic class [KernelInterface](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_kernel_KernelInterface.html), and override the interface function Infer.

```cpp
int CheckInputs(const std::vector<mindspore::MSTensor> &inputs) {         // check function when compiling, to judge the shape of input tensor is valid or not
  for (auto &input : inputs) {
    auto input_shape = input.Shape();
    if (std::find(input_shape.begin(), input_shape.end(), -1) != input_shape.end()) {
      return lite::RET_INFER_INVALID;
    }
  }
  return lite::RET_OK;
}

class CustomAddInfer : public kernel::KernelInterface {
 public:
  CustomAddInfer() = default;
  ~CustomAddInfer() = default;

  Status Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
               const schema::Primitive *primitive) override {        // override interface
    (*outputs)[0].SetFormat((*inputs)[0].format());
    (*outputs)[0].SetDataType((*inputs)[0].DataType());
    auto ret = CheckInputs(inputs);
    if (ret == lite::RET_INFER_INVALID) {
      (*outputs)[0].SetShape({-1});        // set the shape as {-1}，which represents the inferring process will be called again when running
      return kLiteInferInvalid;
    } else if (ret != lite::RET_OK) {
      return kLiteError;
    }
    (*outputs)[0].SetShape((*inputs)[0].Shape());
    return kSuccess;
  }
};
std::shared_ptr<kernel::KernelInterface> CustomAddInferCreator() { return std::make_shared<CustomAddInfer>(); }
REGISTER_CUSTOM_KERNEL_INTERFACE(CustomOpTutorial, Custom_Add, CustomAddInferCreator)       // call the registration interface
```

> The process of inferring shape is composed of two periods, one is static inference when compiling graph, and the other is dynamic inference when running.
>
> Static inference:
>
> 1. If the called function `CheckInputs` returns false or the current node needs to be inferred in the period of running, the shape of output tensor should be set as {-1}, which will be viewed as an identification to infer again when running. In such situation, the return code needs to be set to `RET_INFER_INVALID`.
> 2. In other situation, please return other code. If the code is not `kSuccess`, the program will be aborted and please check the program accordingly.
>
> Dynamic inference
>
> In this period, whether the dynamic inference is needed is up to the shape of output tensor of current node. Please refer to the `Operator Extension` as follows.

#### Operator Extension

1. The users need to inherit the basic class [Kernel](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore_kernel.html#kernel), and override the related interface.

    - Prepare: The interface will be called during graph compilation. Users can make preparations or necessary verifications for the current node before running.

    - Execute：The interface is running interface. Users can call **dynamic inference** [PreProcess](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_extend/src/custom_add_kernel.cc) in this interface.

      ```cpp
      int CheckOutputs(const std::vector<mindspore::MSTensor> &outputs) {           // Check function when running, to judge whether the shape inference is needed
        for (auto &output : outputs) {
          auto output_shape = output.Shape();
          if (std::find(output_shape.begin(), output_shape.end(), -1) != output_shape.end()) {
            return lite::RET_INFER_INVALID;
          }
        }
        return lite::RET_OK;
      }
      ```

    - ReSize: The interface is used to handle the changeable information of the current node due to the shape change of graph inputs.

    - Attribute Parsing: The users need to provide their own parsing of custom operator, which can refer to [ParseAttrData](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_extend/src/custom_add_kernel.cc).

2. Operator registration. The users can refer to the interface [REGISTER_CUSTOM_KERNEL](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore_registry.html#register-custom-kernel).

   ```cpp
   const auto kFloat32 = DataType::kNumberTypeFloat32;
   std::shared_ptr<Kernel> CustomAddCreator(const std::vector<mindspore::MSTensor> &inputs,
                                            const std::vector<mindspore::MSTensor> &outputs,
                                            const schema::Primitive *primitive, const mindspore::Context *ctx) {
     return std::make_shared<CustomAddKernel>(inputs, outputs, primitive, ctx);
   }
   REGISTER_CUSTOM_KERNEL(CPU, CustomOpTutorial, kFloat32, Custom_Add, CustomAddCreator)
   ```

#### Example

1. Compile

   - Environment Requirements

       - System environment: Linux x86_64; Recommend Ubuntu 18.04.02LTS
       - compilation dependencies:
           - [CMake](https://cmake.org/download/) >= 3.18.3
           - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

   - Compilation and Build

     Execute the script [build.sh](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_extend/build.sh) in the directory of `mindspore/lite/examples/runtime_extend`, And then, the released package of MindSpore Lite will be downloaded and the demo will be compiled automatically.

     ```bash
     bash build.sh
     ```

     > If the automatic download is failed, users can download the specified package manually. The hardware platform is CPU and the system is Ubuntu-x64 [mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html), After unzipping, please copy the dynamic library `libmindspore-lite.so` in the directory of `runtime/lib` to the directory of `mindspore/lite/examples/runtime_extend/lib` and copy the directory of `runtime/include` to the directory of `mindspore/lite/examples/runtime_extend`.
     >
     > If the model `add_extend.ms` is failed to download, please download [add_extend.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/add_extend.ms) manually, and copy to the directory of `mindspore/lite/examples/runtime_extend/model`.
     >
     > After manually downloading and storing the specified file, users need to execute the `build.sh` script to complete the compilation and build process.

   - Compilation Result

     The executable program `runtime_extend_tutorial` will be generated in the directory of `mindspore/lite/examples/runtime_extend/build`.

2. Execute Program

   After compiling and building, please enter the directory of `mindspore/lite/examples/runtime_extend/build`, and then execute the following command to experience the extension usaged.

   ```bash
   ./runtime_extend_tutorial ../model/add_extend.ms
   ```

   After the execution, the following information is displayed, including the tensor name, tensor size, number of output tensors, and the first 20 pieces of data.

   ```text
   tensor name is:add-0 tensor size is:400 tensor elements num is:100
   output data is:2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
   ```
