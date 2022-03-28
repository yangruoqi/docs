.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

深度开发
=====================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 数据处理

   data_engine/introduction
   data_engine/eager
   data_engine/auto_augmentation
   data_engine/cache
   data_engine/optimize_data_processing

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 算子执行

   operation/op_classification
   operation/op_overload
   operation/op_cpu
   operation/op_gpu
   operation/op_ascend
   operation/op_custom
   operation/incremental_compilation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 模型推理

   model_infer/inference
   model_infer/online_inference
   model_infer/offline_inference

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 调试调优

   debug/read_ir_files
   debug/debug_in_pynative_mode
   debug/dump_in_graph_mode
   debug/custom_debugging_info
   debug/ms_class

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 分布式并行

   parallel/parallel_architecture
   parallel/parallel_strategy
   parallel/apply_adaptive_summation
   parallel/distributed_inference
   parallel/communication_primitive
   parallel/distributed_training_transformer
   parallel/pangu_alpha

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 高级特性

   others/second_order_optimizer
   others/quantization_aware_training
   others/mixed_precision
   others/gradient_accumulation
   others/apply_host_device_training
   others/numpy
   others/custom_operator
   others/lossscale
   
   
