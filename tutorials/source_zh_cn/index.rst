.. MindSpore documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:00:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MindSpore教程
=====================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 初级
   :hidden:

   beginner/introduction
   beginner/quick_start
   beginner/tensor
   beginner/dataset
   beginner/model
   beginner/autograd
   beginner/train
   beginner/save_load
   beginner/infer

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 进阶
   :hidden:

   advance/linear_fitting
   advance/dataset
   advance/network
   advance/callback
   advance/pynative_mode_and_graph_mode

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 计算机视觉
   :hidden:

   cv/cv_resnet50
   cv/transfer_learning
   cv/fgsm
   cv/dcgan

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 自然语言处理
   :hidden:

   nlp/sentiment_analysis
   nlp/bert_poetry

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 数据引擎
   :hidden:

   data_engine/introduction
   data_engine/eager
   data_engine/auto_augmentation
   data_engine/cache
   data_engine/optimize_data_processing

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 算子执行
   :hidden:

  operation/op_classification
  operation/op_overload
  operation/op_cpu
  operation/op_gpu
  operation/op_ascend
  operation/op_custom

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 编译优化
   :hidden:

   compiler_optimization/enable_graph_kernel_fusion
   compiler_optimization/jit_fallback

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 模型推理
   :hidden:

   model_infer/inference
   model_infer/online_inference
   model_infer/offline_inference

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 分布式并行
   :hidden:

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
   :caption: 调试调优
   :hidden:

   debug/read_ir_files
   debug/debug_in_pynative_mode
   debug/dump_in_graph_mode
   debug/incremental_operator_build
   debug/custom_debugging_info

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 其他特性
   :hidden:

   others/second_order_optimizer
   others/apply_quantization_aware_training
   others/enable_mixed_precision
   others/apply_gradient_accumulation
   others/apply_host_device_training
   others/numpy
   others/custom_operator
   others/lossscale


.. raw:: html

   <div class="container">
         <h2>面向新手</h2>
			<div class="row">
				<div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./beginner/quick_start.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">快速入门</span>
                           </div>
                           <div class="doc-article-desc">
                              使用MindSpore实现手写数字识别。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./intermediate/dataset_load_process.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">数据处理与加载</span>
                           </div>
                           <div class="doc-article-desc">
                              详细描述待更新。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
			</div>
         <div class="row">
				<div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./intermediate/build_net.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">网络构建</span>
                           </div>
                           <div class="doc-article-desc">
                              详细描述待更新。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./intermediate/pynative_mode_and_graph_mode.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">动态图与静态图</span>
                           </div>
                           <div class="doc-article-desc">
                              详细描述待更新。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
			</div>
         <h2>面向专家</h2>
         <div class="row">
				<div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./data_engine/eager.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">数据引擎</span>
                           </div>
                           <div class="doc-article-desc">
                              详细描述待更新。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./compiler_optimization/enable_graph_kernel_fusion.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">编译优化</span>
                           </div>
                           <div class="doc-article-desc">
                              详细描述待更新。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
			</div>
         <div class="row">
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./model_infer/inference.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">模型推理</span>
                           </div>
                           <div class="doc-article-desc">
                              详细描述待更新。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
				<div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./parallel/parallel_architecture.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">分布式并行</span>
                           </div>
                           <div class="doc-article-desc">
                              详细描述待更新。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
			</div>
         <div class="row">
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./debug/read_ir_files.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">调试调优</span>
                           </div>
                           <div class="doc-article-desc">
                              详细描述待更新。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./other_feature/second_order_optimizer.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">其他特征</span>
                           </div>
                           <div class="doc-article-desc">
                              详细描述待更新。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
         </div>
         <h2>应用实践</h2>
         <div class="row">
				<div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./cv/cv_resnet50.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">计算机视觉</span>
                           </div>
                           <div class="doc-article-desc">
                              详细描述待更新。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
            <div class="col-md-6">
               <div class="doc-article-list">
                  <div class="doc-article-item">
                     <a href="./nlp/bert_poetry.html" class="article-link">
                        <div>
                           <div class="doc-article-head">
                              <span class="doc-head-content">自然语言处理</span>
                           </div>
                           <div class="doc-article-desc">
                              详细描述待更新。
                           </div>
                        </div>
                     </a>
                  </div>
					</div>
				</div>
			</div>
	</div>
		
