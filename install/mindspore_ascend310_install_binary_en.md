# Installing MindSpore in Ascend 310 by binary package

<!-- TOC -->

- [Installing MindSpore in Ascend 310 by binary package](#installing-mindspore-in-ascend-310-by-binary-package)
    - [Installing MindSpore and dependencies](#installing-mindspore-and-dependencies)
        - [Installing Python](#installing-python)
        - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
        - [Installing GCC](#installing-gcc)
        - [Installing gmp](#installing-gmp)
        - [Installing CMake](#installing-cmake)
        - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Verifying the Installation](#verifying-the-installation)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend310_install_pip_en.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The following describes how to quickly install MindSpore by pip on Linux in the Ascend 310 environment, MindSpore in Ascend 310 only supports inference.

## Installing MindSpore and dependencies

The following table lists the system environment and third-party dependencies required for installing MindSpore.

|software|version|description|
|-|-|-|
|Ubuntu 18.04/CentOS 7.6/EulerOS 2.8|-|OS for running and compiling MindSpore|
|[Python](#installing-python)|3.7-3.9|Python environment that MindSpore depends|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|-|Ascend platform AI computing library used by MindSpore|
|[GCC](#installing-gcc)|7.3.0|C++ compiler for compiling MindSpore|
|[gmp](#installing-gmp)|6.1.2|Multiple precision arithmetic library used by MindSpore|
|[CMake](#installing-cmake)|3.18.3 or later| Compilation tool that builds MindSpore                  |

The following describes how to install the third-party dependencies.

### Installing Python

[Python](https://www.python.org/) can be installed by Conda.

Install Miniconda:

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

After the installation is complete, you can set up Tsinghua source acceleration download for Conda, and see [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

Create a virtual environment, taking Python 3.7.5 as an example:

```bash
conda create -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

Run the following command to check the Python version.

```bash
python --version
```

If you are using an ARM architecture system, please ensure that pip installed for current Python has a version >= 19.3. If not, upgrade pip with the following command.

```bash
python -m pip install -U pip
```

### Installing Ascend AI processor software package

For detailed installation guide, please refer to [Ascend Data Center Solution 21.1.0 Installation Guide].

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path of Ascend AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

Install the .whl packages provided in Ascend AI processor software package. The .whl packages are released with the software package. If the .whl packages have been installed before, you need to uninstall the packages by the following command.

```bash
pip uninstall te topi hccl -y
```

Run the following command to install the .whl packages if the Ascend AI package has been installed in default path. If the installation path is not the default path, you need to replace the path in the command with the installation path.

```bash
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-*-py3-none-any.whl
```

### Installing GCC

- On Ubuntu 18.04, run the following commands to install.

    ```bash
    sudo apt-get install gcc-7 -y
    ```

- On CentOS 7, run the following commands to install.

    ```bash
    sudo yum install centos-release-scl
    sudo yum install devtoolset-7
    ```

    After installation, run the following commands to switch to GCC 7.

    ```bash
    scl enable devtoolset-7 bash
    ```

- On EulerOS, run the following commands to install.

    ```bash
    sudo yum install gcc -y
    ```

### Installing gmp

- On Ubuntu 18.04, run the following commands to install.

    ```bash
    sudo apt-get install libgmp-dev -y
    ```

- On CentOS 7 and EulerOS, run the following commands to install.

    ```bash
    sudo yum install gmp-devel -y
    ```

### Installing CMake

- On Ubuntu 18.04, run the following commands to install [CMake](https://cmake.org/).

    ```bash
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
    sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
    sudo apt-get install cmake -y
    ```

- Other Linux systems can be installed with the following commands.

    Choose different download links based on the system architecture.

    ```bash
    # x86 run
    curl -O https://cmake.org/files/v3.19/cmake-3.19.8-Linux-x86_64.sh
    # aarch64 run
    curl -O https://cmake.org/files/v3.19/cmake-3.19.8-Linux-aarch64.sh
    ```

    Run the script to install CMake, which is installed in the `/usr/local` by default.

    ```bash
    sudo mkdir /usr/local/cmake-3.19.8
    sudo bash cmake-3.19.8-Linux-*.sh --prefix=/usr/local/cmake-3.19.8 --exclude-subdir
    ```

    Finally, add CMake to the `PATH` environment variable. Run the following commands if it is installed in the default path, and other installation path need to be modified accordingly.

    ```bash
    echo -e "export PATH=/usr/local/cmake-3.19.8/bin:\$PATH" >> ~/.bashrc
    source ~/.bashrc
    ```

### Installing MindSpore

First, refer to [Version List](https://www.mindspore.cn/versions) to select the version of MindSpore you want to install, and perform SHA-256 integrity check. Taking version 1.7.0 as an example, execute the following commands.

```bash
export MS_VERSION=1.7.0
```

Then run the following commands to install MindSpore according to the system architecture and Python version.

```bash
# x86_64
wget https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/ascend/{arch}/mindspore_ascend-${MS_VERSION/-/}-linux_x86_64.tar.gz --no-check-certificate
tar -zxf mindspore_ascend-${MS_VERSION/-/}-linux_x86_64.tar.gz

# aarch64
wget https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/ascend/{arch}/mindspore_ascend-${MS_VERSION/-/}-linux_aarch64.tar.gz --no-check-certificate
tar -zxf mindspore_ascend-${MS_VERSION/-/}-linux_aarch64.tar.gz
```

## Configuring Environment Variables

After MindSpore is installed, export runtime environment variables. In the following command, `/usr/local/Ascend` in `LOCAL_ASCEND=/usr/local/Ascend` indicates the installation path of the software package. Change it to the actual installation path.

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
## TBE operator implementation tool path
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
## OPP path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp
## AICPU path
export ASCEND_AICPU_PATH=${ASCEND_OPP_PATH}/..
## TBE operator compilation tool path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}
## Python library that TBE implementation depends on
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
```

## Verifying the Installation

Create a directory to store the sample code project, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample`. You can obtain the code from the [official website](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/sample_resources/ascend310_single_op_sample.zip). A simple example of adding `[1, 2, 3, 4]` to `[2, 3, 4, 5]` is used and the code project directory structure is as follows:

```text
└─ascend310_single_op_sample
    ├── CMakeLists.txt                    // Build script
    ├── README.md                         // Usage description
    ├── main.cc                           // Main function
    └── tensor_add.mindir                 // MindIR model file
```

Go to the directory of the sample project and change the path based on the actual requirements.

```bash
cd /home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample
```

Build a project by referring to `README.md`, where `{mindspore_path}` represents the absolute path to the location where the MindSpore binary package is located, and is replaced according to the actual situation.

```bash
cmake . -DMINDSPORE_PATH={mindspore_path}
make
```

After the build is successful, execute the case.

```bash
./tensor_add_sample
```

The following information is displayed:

```text
3
5
7
9
```

The preceding information indicates that MindSpore is successfully installed.
