Bootstrap: docker
From: ubuntu:18.04

%post -c /bin/bash

    cd /

    # Prepare directories for installing applications
    mkdir -p APPS
    mkdir -p INSTALLERS

    # Update all libraries
    apt-get -y update

    # Install xvfb
    apt-get -y install xvfb

    # Install ghostscript for pdf management
    apt-get -y install ghostscript

    # Install MRTrix3
    apt-get -y install git g++ python python-numpy libeigen3-dev zlib1g-dev libqt4-opengl-dev libgl1-mesa-dev libfftw3-dev libtiff5-dev python3-distutils
    cd APPS
    git clone https://github.com/MRtrix3/mrtrix3.git
    cd mrtrix3
    git checkout 3.0.3
    ./configure
    ./build
    cd /

    # Install FSL
    apt-get -y install python wget ca-certificates libglu1-mesa libgl1-mesa-glx libsm6 libice6 libxt6 libpng16-16 libxrender1 libxcursor1 libxinerama1 libfreetype6 libxft2 libxrandr2 libgtk2.0-0 libpulse0 libasound2 libcaca0 libopenblas-base bzip2 dc bc 
    #wget -O /INSTALLERS/fslinstaller.py "https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py"
    wget -O /INSTALLERS/fslinstaller.py "https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation?action=AttachFile&do=get&target=fslinstaller.py"
    cd /INSTALLERS
    python fslinstaller.py -d /APPS/fsl -V 6.0.4
    cd /
    
    # Install Convert3D (stable build 1.0.0)
    apt-get -y install wget tar
    wget -O /INSTALLERS/c3d-1.0.0-Linux-x86_64.tar.gz "https://downloads.sourceforge.net/project/c3d/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fc3d%2Ffiles%2Fc3d%2F1.0.0%2Fc3d-1.0.0-Linux-x86_64.tar.gz%2Fdownload&ts=1571934949"
    tar -xf /INSTALLERS/c3d-1.0.0-Linux-x86_64.tar.gz -C /APPS/
    cd /

    # Install ANTs (and compatible CMake)
    apt-get -y install build-essential libssl-dev
    # CMake: The latest ANTs requires newer version of cmake than can be installed
    # through apt-get, so we need to build higher version of cmake from source
    cd /INSTALLERS
    mkdir cmake_install
    cd cmake_install
    wget https://github.com/Kitware/CMake/releases/download/v3.23.0-rc2/cmake-3.23.0-rc2.tar.gz
    tar -xf cmake-3.23.0-rc2.tar.gz
    cd cmake-3.23.0-rc2/
    ./bootstrap
    make
    make install
    cd /
    # ANTS
    cd /INSTALLERS
    mkdir ants_installer
    cd ants_installer
    git clone https://github.com/ANTsX/ANTs.git
    git checkout efa80e3f582d78733724c29847b18f3311a66b54
    mkdir ants_build
    cd ants_build
    cmake /INSTALLERS/ants_installer/ANTs -DCMAKE_INSTALL_PREFIX=/APPS/ants
    make 2>&1 | tee build.log
    cd ANTS-build
    make install 2>&1 | tee install.log
    cd /

    # Install FreeSurfer
    apt-get -y install bc binutils libgomp1 perl psmisc sudo tar tcsh unzip uuid-dev vim-common libjpeg62-dev
    wget -O /INSTALLERS/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.0/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz"
    tar -C /APPS -xzvf /INSTALLERS/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz
    echo "This is a dummy license file. Please bind your freesurfer license file to this file." > /APPS/freesurfer/license.txt
    cd /

    # Download the Matlab Compiled Runtime installer, install, clean up
    apt-get install -y wget unzip openjdk-8-jre libxt6
    mkdir MCR
    wget -nv -P /MCR http://ssd.mathworks.com/supportfiles/downloads/R2017a/deployment_files/R2017a/installers/glnxa64/MCR_R2017a_glnxa64_installer.zip
    unzip -q /MCR/MCR_R2017a_glnxa64_installer.zip -d /MCR/MCR_R2017a_glnxa64_installer
    /MCR/MCR_R2017a_glnxa64_installer/install -mode silent -agreeToLicense yes
    rm -r /MCR/MCR_R2017a_glnxa64_installer /MCR/MCR_R2017a_glnxa64_installer.zip
    rmdir /MCR
    
    # Make custom folders
    mkdir -p INPUTS
    mkdir -p SUPPLEMENTAL
    mkdir -p OUTPUTS
    mkdir -p CODE

    # Set Permissions
    chmod 755 /INPUTS
    chmod 755 /SUPPLEMENTAL
    chmod 755 /APPS
    chmod 775 /OUTPUTS
    chmod 755 /CODE
    
    # Install source code
    cd /
    apt-get -y install wget git gcc libpq-dev python-dev python-pip python3 python3.8 python3.8-venv python3.8-dev python3-dev python3-pip python3-venv python3-wheel libpng-dev libfreetype6-dev libblas3 liblapack3 libblas-dev liblapack-dev pkg-config
    cd /INSTALLERS
    git clone https://github.com/MASILab/PreQual.git
    cd PreQual
    git checkout v1.1.0
    mv src/APPS/* /APPS
    mv src/CODE/* /CODE
    mv src/SUPPLEMENTAL/* /SUPPLEMENTAL
    cd /APPS/synb0
    python3.6 -m venv pytorch
    source pytorch/bin/activate
    pip3 install wheel
    pip install -r /INSTALLERS/PreQual/venv/pip_install_synb0.txt
    deactivate
    cd /CODE/dtiQA_v7
    python3 -m venv venv
    source venv/bin/activate
    pip3 install wheel
    pip install -r /INSTALLERS/PreQual/venv/pip_install_dtiQA.txt
    deactivate
    cd /APPS/gradtensor
    python3.8 -m venv gradvenv
    source gradvenv/bin/activate
    wget https://bootstrap.pypa.io/get-pip.py
    python3.8 get-pip.py
    pip3 install --upgrade setuptools
    pip install fpdf imageio pypng freetype-py numpy==1.21.*
    git clone https://github.com/scilus/scilpy.git
    cd scilpy
    git checkout 1.4.0
    pip install -e .
    cd ..
    deactivate
    cd /

    # Clean up
    rm -r /INSTALLERS

%environment

    # MRTrix3
    export PATH="/APPS/mrtrix3/bin:$PATH"

    # FSL
    FSLDIR=/APPS/fsl
    . ${FSLDIR}/etc/fslconf/fsl.sh
    PATH=${FSLDIR}/bin:${PATH}
    export FSLDIR PATH

    # Convert3D
    export PATH="/APPS/c3d-1.0.0-Linux-x86_64/bin:$PATH"

    # ANTs
    export ANTSPATH=/APPS/ants/bin/
    export PATH=${ANTSPATH}:$PATH

    # FreeSurfer
    export FREESURFER_HOME=/APPS/freesurfer
    #source $FREESURFER_HOME/SetUpFreeSurfer.sh # For us, only synb0 needs it so will put in that script.

    # CUDA
    export CPATH="/usr/local/cuda/include:$CPATH"
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    export CUDA_HOME="/usr/local/cuda"

%runscript

    xvfb-run -a --server-num=$((65536+$$)) --server-args="-screen 0 1600x1280x24 -ac" bash /CODE/run_dtiQA.sh /INPUTS /OUTPUTS "$@"
