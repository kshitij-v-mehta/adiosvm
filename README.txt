adiosvm
=======

Packages and howtos for creating a linux system for ADIOS tutorials

Required steps to get plain ADIOS working: I.1-6,8 II.1,4 III.1

Steps:

I. Set up a Linux VM
====================

1. Install VirtualBox 

2. Get a linux ISO image
   We currently use Lubuntu 16.04 64bit and the descriptions below all refer to that system. Debian based systems use 'apt-get' to install packages. 


3. Create a new VM
   Type=Linux, System=Ubuntu (32bit)
   Memory: at least 2048MB, we use 3072MB just for the sake of it
   Virtual hard drive: VMDK type, dynamically allocated, about 16GB
      - the more the better, but an exported images should fit on an USB stick...
   Video memory: 64MB - we don't really know how much is needed
      - do not enable 2D video or 3D acceleration
   Processor: 2 CPUs 
      - 1 is enough, 2 builds codes faster
   Storage: load the linux iso image into the DVD drive. 


   - Start the VM and install linux, preferably with updates. 
    Your name: ADIOS
    Your computer's nane: adiosVM
    Username: adios
    password: adios 
    Log in automatically: yes, because this is a tutorial vm

    Note: if you use a computer name other than adiosVM, make sure you substitute for the name
    everywhere in this document (e.g. Flexpath install)

   - Instead of restart, shut down and remove the linux DVD, start again

   Whenever Lubuntu offers updates, do it...

4. LXTerminal: Bottom left corner of screen has the start menu, open System Tools/LXTerminal
   - LXTerminal setup (if you don't like the default one)
     - Start LXTerminal
     - Edit/Profile preferences
     - Display tab: increase the scrollback to a few thousand lines (e.g. 5120)
     - Style tab: set font and background/foreground to your style
       (Monospace 12 and black font on white background for the tutorial VM)

   - Add terminal icon to task bar: right-click on task bar, select Add/Remove Panel Items
     Select Application Launch Bar, click Preferences
     Add System Tools/LXTerminal to the Launchers list

5. Virtualbox Guest Additions
   $ sudo apt-get update
   $ sudo apt-get install dkms 
     - Dynamic Kernel Modules, to ensure rebuilding Guest Additions at future kernel updates
     - or just reinstall Guest Additions each time if you don't want dkms installed

   - VirtualBox VM menu: Devices/Insert Guest Additions CD Image
     - under Lubuntu it will not autorun, so in a terminal
     $ cd /media/adios/VBOXADDITIONS*/
     $ sudo ./VBoxLinuxAdditions.run

     - this allows for resizing the window and 
       for copy/paste between the VM and your host machine
       (and sharing folders between your host machine and this VM if you want)
     - set in VirtualBox main menu:  Devices/Shared Clipboard/Bidirectional)
     Note: This has to be repeated when updating or recompiling the kernel unless 
           the dkms package is installed
     - reboot the machine

6. Install some linux packages (sudo apt-get install or can use sudo synaptic)
   $ sudo apt-get install apt-file 
   $ sudo apt-file update
   $ sudo apt-get install build-essential git-core libtool libtool-bin autoconf subversion 
   $ sudo apt-get install gfortran 
   $ sudo apt-get install pkg-config 
   $ sudo apt-get autoremove 
   -- this one is to remove unused packages, probably 0 at this point

   - Set HISTORY to longer: 
     $ vi ~/.bashrc
     increase HISTSIZE (to 5000) and HISTFILESIZE (to 10000)

   - Turn off screen saver and lock
     Start menu / Preferences  / Power Manager
     Display tab:
       Turn off flag to Handle display power management
       Set Blank after to the max (60 minutes on Lubuntu)
     Security tab: 
       Automatically lock the session: Never
       Turn off flag to lock screen when system is going to sleep


7. Shared folder between your host machine and the VM (optional)
   This is not needed for tutorial, just if you want to share files between host and vm.

   - In VirtualBox, while the VM is shut down, set up a shared folder, with auto-mount.
   - Start VM. You can see a folder  /media/sf_<your folder name>
   - Need to set group rights for adios user to use the folder
     $ sudo vi /etc/group 
     add "adios" to the vboxfs entry so that it looks like this

     vboxsf:x:999:adios
   - log out and back, run 'groups' to check if you got the group rights.



8. Download this repository
   You can postpone step 6 and 7 after 8 if you have a github account
   and want to edit this repository content.

   $ cd
   $ git clone https://github.com/pnorbert/adiosvm.git

9. VIM setup
   $ sudo apt-get install vim
   copy from this repo: vimrc to ~/.vimrc
   $ cp ~/adiosvm/vimrc .vimrc

   On Lubuntu for some reason, the root creates ~/.viminfo which we need to remove
   This will allow vi/vim to create it again under adios user and use it to remember
   positions in files opened before
   $ sudo rm ~/.viminfo

10. Github access setup
   This step is only needed for ADIOS developers to get 
   write access to the ADIOS repository from github.

   We need an account to github and a config for ssh.
   A minimum .ssh/config is found in this repository:  
   $ cd 
   $ mkdir .ssh
   $ cp ~/adiosvm/ssh_config ~/.ssh/config

   If the shared folder has access to you .ssh:

   $ cp /media/sf_<yourfolder>/.ssh/config .ssh
   $ cp /media/sf_<yourfolder>/.ssh/id_dsa_github* .ssh

   If you need a proxy to get to GitHub, use corkscrew and edit
   ~/.ssh/config to add the proxy command to each entry
   $ sudo apt-get install corkscrew

   If you have a github account and the config already, and postponed
   step 6, get the adiosvm repo now:

   $ git clone github:pnorbert/adiosvm.git

   Git settings:
   $ git config --global user.name "<your name>"
   $ git config --global user.email "<your email>"
   $ git config --global core.editor vim

   Of course, set an editor what you like.


II. Preparations to install ADIOS
=================================

1. Linux Packages
   $ sudo apt-get install openmpi-common openmpi-bin libopenmpi-dev 
   $ sudo apt-get install gfortran 
   $ sudo apt-get install python-cheetah python-yaml


2. CMake
   We need a newer CMake version than what's available in linux distros. 
   Download from: https://cmake.org/download/
   Choose the self-extracting archive package for the Linux x86_64 platform and install it. 

3. Staging support
   a. libfabric is required by the SST staging engine
   ------------
   Download latest release from Github
          https://github.com/ofiwg/libfabric/releases
   and extract in ~/Software
   $ cd libfabric-1.6.0
   $ ./configure --disable-verbs --prefix=/opt/libfabric
   $ make
   $ sudo make install

   In ~/.bashrc, add to LD_LIBRARY_PATH "/opt/libfabric/lib"

   b. ZeroMQ is required by the DataMan staging engine
   ---------

   $ sudo apt-get install libzmq5 libzmq3-dev

4. Compression libraries
   Only if you want to demo the transform library.

   zlib and bzip2 are installed as linux packages:
   --------------
   $ sudo apt-get install bzip2 libbz2-dev zlib1g zlib1g-dev

   SZ is provided in adiospackages/
   --

   $ cd ~/Software
   $ tar zxf ~/adiosvm/adiospackages/sz-1.4.13.0.tar.gz
   $ cd sz-1.4.13.0
   $ ./configure --prefix=/opt/SZ --with-pic --disable-shared --disable-fortran --disable-maintainer-mode
   $ make
   $ sudo make install


   BLOSC is available on GitHub:
   ----- 

   $ cd ~/Software
   $ git clone https://github.com/Blosc/c-blosc.git
   $ cd c-blosc
   $ mkdir build
   $ cd build
   $ cmake -DCMAKE_INSTALL_PREFIX=/opt/blosc ..
   $ make
   $ sudo make install

   In ~/.bashrc, add to LD_LIBRARY_PATH "/opt/blosc/lib"


5. Parallel HDF5 support 
   Only if you want HDF5 read/write in ADIOS.

   $ cd ~/Software
   $ tar jxf ~/adiosvm/adiospackages/hdf5-1.8.17.tar.bz2
   $ mv hdf5-1.8.17 hdf5-1.8.17-parallel
   $ cd hdf5-1.8.17-parallel
   $ ./configure --with-zlib=/usr --without-szlib --prefix=/opt/hdf5-1.8.17-parallel --enable-parallel --enable-fortran --with-pic  CC=mpicc FC=mpif90

   Verify that in the Features list:
        Parallel HDF5: yes

   Note: the -fPIC option is required for building parallel NetCDF4 later

   $ make -j 4
   $ sudo make install


6. Python/Numpy support

   To build Adios python wrapper, install following packages by:
   $ sudo apt-get install python3 python3-dev

   Note: To use a parallel version, we need mpi4py. 

   $ sudo apt-get install python3-pip python3-tk
   $ sudo -H pip3 install numpy mpi4py matplotlib
   
   Alternatively, we can install from a source code too:

   $ wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-2.0.0.tar.gz
   $ tar xvf mpi4py-2.0.0.tar.gz
   $ cd mpi4py-2.0.0
   $ python3 setup.py build
   $ sudo python3 setup.py install


9. Fastbit indexing support (needed for queries) for ADIOS 1.x
   $ cd ~/Software
   $ svn co https://code.lbl.gov/svn/fastbit/trunk fastbit
     username and password: anonsvn
   $ cd fastbit/
   $ ./configure --with-pic -prefix=/opt/fastbit
   $ make
   -- this will be slooooow
   $ sudo make install
   $ make clean  
   -- src/ is about 1.4GB after build

   In ~/.bashrc, add to LD_LIBRARY_PATH "/opt/fastbit/lib"


10. Alacrity indexing and query support for ADIOS 1.x
   $ cd ~/Software
   either
   $ tar zxf ~/adiosvm/adiospackages/alacrity-1.0.0.tar.gz
   $ cd alacrity-1.0.0
     or
   $ git clone https://github.com/ornladios/ALACRITY-ADIOS.git
   $ cd ALACRITY-ADIOS

   $ . ./runconf
     or
   $ ./configure CFLAGS="-g -fPIC -fno-common -Wall" CXXFLAGS="-g -fPIC -fno-exceptions -fno-rtti" --prefix=/opt/alacrity
   $ make
   $ sudo make install


III. ADIOS Installation
=======================

1. Download ADIOS
   2. Download ADIOS master from repository
   $ cd ~/Software
   $ git clone github:ornladios/ADIOS2.git
     OR
   $ git clone https://github.com/ornladios/ADIOS2.git
   $ cd ADIOS2

2. Build ADIOS
   Then:
   $ mkdir build
   $ cd build
   $ cmake -DCMAKE_INSTALL_PREFIX=/opt/adios2 \
      -DCMAKE_BUILD_TYPE=Release  \
      -DCMAKE_PREFIX_PATH="/opt/blosc;/opt/zfp/0.5.5;/opt/SZ/2.0.2.1;/opt/MGARD;/opt/hdf5-parallel" \
      -DADIOS2_USE_MPI=ON \
      -DADIOS2_USE_Python=ON \
      -DADIOS2_USE_Profiling=ON \
      -DADIOS2_BUILD_TESTING=OFF \
      ..
   $ make -j 4

3. Test ADIOS a bit
   $ ctest

4. Install 
   $ sudo make install
   In ~/.bashrc, add to LD_LIBRARY_PATH "/opt/adios2/lib" and 
     add to PATH "/opt/adios2/bin"

5. Python wrapper



IV. ADIOS 1.x for compression and queries
=========================================

1. Download ADIOS
   1. ADIOS 1.13.1 is in this repo: 
   $ cd ~/Software
   $ tar zxf ~/adiosvm/adiospackages/adios-1.13.1.tar.gz
   $ cd adios-1.13.1

   2. Download ADIOS master from repository
   $ cd ~/Software
   $ git clone github:ornladios/ADIOS.git
     OR
   $ git clone https://github.com/ornladios/ADIOS.git
   $ cd ADIOS
   $ ./autogen.sh

2. Build ADIOS
   Then: Edit runconf and change install path for the adiosVM target to /opt/adios1
   $ mkdir build
   $ cd build
   $ ~/adiosvm/adiospackages/runconf.adios1 
    Configure ADIOS 1.x on VirtualBox.
    ...
   $ make -j 4

3. Test ADIOS a bit
   $ make check

   $ cd tests/suite
   $ ./test.sh 01
     and so on up to 16 tests
     Some test may fail because multiple processes write a text file which is 
     compared to a reference file, but some lines may get mixed up. Try 
     running the same test a couple of times. One 'OK' means the particular
     test is okay.

4. Install 
   $ sudo make install
   In ~/.bashrc, add to LD_LIBRARY_PATH "/opt/adios1/lib" and 
     add to PATH "/opt/adios1/bin"

5. Build and install python wrapper

   To build Adios python wrapper, install following packages by:
   $ sudo apt-get install python3 python3-dev python3-tk

   Note: To use a parallel version, we need mpi4py. 

   $ sudo apt-get install python3-pip
   $ sudo -H pip3 install numpy mpi4py matplotlib
   
   Alternatively, we can install from a source code too:

   $ wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-2.0.0.tar.gz
   $ tar xvf mpi4py-2.0.0.tar.gz
   $ cd mpi4py-2.0.0
   $ python3 setup.py build
   $ sudo python3 setup.py install

   Then, we are ready to install adios and adios_mpi python module. An
   easy way is to use "pip3".
   
   $ sudo -H "PATH=$PATH" pip3 install --upgrade \
     --global-option build_ext --global-option -lrt adios adios_mpi

   If there is any error, we can build from source. Go to the
   wrapper/numpy directory under the adios source directory:
   $ cd wrapper/numpy

   Type the following to build Adios python wrapper:
   $ python3 setup.py build_ext 
   If not working, add " -lrt"

   The following command is to install:
   $ sudo "PATH=$PATH" python3 setup.py install

   Same for adios_mpi module:
   $ python3 setup_mpi.py build_ext -lrt
   $ sudo "PATH=$PATH" python3 setup_mpi.py install

   Test:
   A quick test can be done:
   $ cd wrapper/numpy/tests
   $ python3 test_adios.py
   $ mpirun -n 4 python3 test_adios_mpi.py


V. ADIOS Tutorial code
=======================

   The tutorial is included in this repository
   ~/adiosvm/Tutorial

1. Linux Packages
   KSTAR demo requires gnuplot
   $ sudo apt-get install gnuplot

   Brusselator demo requires FFTW3
   $ sudo apt-get install libfftw3-dev
   

VI. Build VTK-M from source
===========================

  $ cd ~/Software
  $ git clone https://gitlab.kitware.com/vtk/vtk-m.git
  $ cd vtk-m
  $ mkdir build
  $ cd build
  $ cmake -DCMAKE_INSTALL_PREFIX=/opt/vtk-m ..
  $ make -j 4 
  $ sudo make install

   In ~/.bashrc, add to LD_LIBRARY_PATH "/opt/vtk-m/lib"

VI. Build Visit from release
===========================

Simple way: install binary release
  - download a release from https://wci.llnl.gov/simulation/computer-codes/visit/executables
  E.g.:
  $ cd /opt
  $ sudo tar zxf visit2_10_2.linux-x86_64-ubuntu14.tar.gz
  $ sudo mv visit2_10_2.linux-x86_64 visit


Complicated way:

- Need to have many linux packages installed.
- Build a Visit release using it's build script that
  downloads/builds a lot of dependencies.

1. Linux packages
  $ sudo apt-get install dialog gcc-multilib subversion libx11-dev tcl tk
  $ sudo apt-get install libglu1-mesa-dev 
  $ sudo libxt-dev
  $ sudo apt-get install xutils-dev

  For Silo reader to believe in good Qt installation
  $ sudo apt-get install libxmu-dev libxi-dev

  Qt development packages (required or visit will build its own)
  $ sudo apt-get libqt4-dev 
  This will install all dependent packages

2. Build latest Visit release (with many dependencies)
   https://wci.llnl.gov/codes/visit/source.html

   Visit 2.7.2 release uses adios 1.6.0 but it downloads and builds its own
   version without compression or staging. 

   $ cd ~/Software
   $ mkdir -p visit
   $ cd visit
   $ wget http://portal.nersc.gov/svn/visit/trunk/releases/2.7.2/build_visit2_7_2
     Or download the latest build script from the website
     https://wci.llnl.gov/codes/visit/source.html

   $ chmod +x build_visit2_7_2
   $ ./build_visit2_7_2 --parallel --mesa --mxml --adios --hdf5 --xdmf --zlib --silo
   $ ./build_visit --system-qt --parallel --mesa --mxml --adios --hdf5 --xdmf --zlib --silo --console

    This script should be started again and again after fixing build problems.
    All log is founf in build_visit2_7_2_log, appended at each try.

    In the dialogs, just accept everything


                     |     You many now try to run VisIt by cd'ing into the     │  
                     │ visit2.7.2/src/bin directory and invoking "visit".       │  
                     │                                                          │  
                     │     To create a binary distribution tarball from this    │  
                     │ build, cd to                                             │  
                     │ /home/adios/Software/visit/visit2.7.2/src                │  
                     │     then enter: "make package"                           │  
                     │                                                          │  
                     │     This will produce a tarball called                   │  
                     │ visitVERSION.ARCH.tar.gz, where     VERSION is the       │  
                     │ version number, and ARCH is the OS architecure.          │  
                     │                                                          │  
                     │     To install the above tarball in a directory called   │  
                     │ "INSTALL_DIR_PATH",    enter: svn_bin/visit-install      │  
                     │ VERSION ARCH INSTALL_DIR_PATH                            |



3. Build Visit from svn trunk
- This has to be built on a second scratch disk. It requires ~10GB space. 
  The installation is just 0.5GB.

   $ cd /work/adios
   $ mkdir visit.build
   $ cd visit.build/
   $ svn co http://visit.ilight.com/svn/visit/trunk/src/svn_bin svn_bin
   $ ln -s svn_bin/build_visit .
   $ ln -s svn_bin/bv_support .
   $ git clone github:ornladios/ADIOS2.git
   $ ./build_visit --system-cmake --system-python --adios2 --silo --szip --zlib --hdf5 --makeflags -j4 --prefix /opt/visit

   when it is failed or succeeded with building the Visit source (not just the third-parties),
   i.e. when trunk/ already exists

   $ cp adiosVM.cmake trunk/src/config-site
   -- edit trunk/src/config-site/adiosVM.cmake  and point to a SERIAL ADIOS2 installation
       VISIT_OPTION_DEFAULT(VISIT_ADIOS2_DIR /opt/adios2-serial)
       VISIT_OPTION_DEFAULT(VISIT_ADIOS2_PAR_DIR /opt/adios2)


   and build manually after this:

   $ cd trunk
   $ mkdir -p build
   $ cd build
   $ cmake ../src
   $ make -j 4


   $ sudo make install
   This should put the installation files into /opt/visit/<version> and the executable script into /opt/visit/bin



VII. Build Plotter
=================

If you still want to use the our own plotter instead of / besides visit.

  1. Install grace or build it
  ----------------------------
  $ sudo apt-get install grace
  -- use XMGRACE=/usr in Makefile.adiosVM of the plotter package below

  or

  $ sudo apt-get install libmotif-common libmotif-dev
  $ tar zxf ~/adiosvm/plotterpackages/grace-5.1.25.tar.gz
  $ cd grace-5.1.25
  $ ./configure --prefix=/opt/plotter
  $ make -j 4
  $ sudo make install
  -- use XMGRACE=/opt/plotter/grace in Makefile.adiosVM of the plotter package below




  If not installed Visit yet:
  ---------------------------
  $ sudo apt-get install libglu1-mesa-dev libxt-dev

  $ mkdir ~/Software/plotter
  $ cd ~/Software/plotter


  1. Build Mesa library (libosmesa6-dev package does not seem to let vtk build)
  $ cd ~/Software/plotter
  $ tar zxf ~/adiosvm/plotterpackages/Mesa-7.8.2.tar.gz
  $ cd Mesa-7.8.2
  $ ./configure CFLAGS="-I/usr/include/i386-linux-gnu -O2 -DUSE_MGL_NAMESPACE -fPIC -DGLX_USE_TLS" CXXFLAGS="-O2 -DUSE_MGL_NAMESPACE -fPIC -DGLX_USE_TLS" --prefix=/opt/plotter --with-driver=osmesa --disable-driglx-direct 
  $ make -j 4
  $ sudo make install

  
  2. Build an old VTK library 
  $ tar zxf  ~/adiosvm/plotterpackages/visit-vtk-5.8.tar.gz
  $ mkdir vtk-5.8-build
  $ cd vtk-5.8-build
  $ export LD_LIBRARY_PATH=/opt/plotter/lib:$LD_LIBRARY_PATH
  $ cmake -DCMAKE_INSTALL_PREFIX:PATH=/opt/plotter -DCMAKE_C_FLAGS=-DGLX_GLXEXT_LEGACY -DCMAKE_CXX_FLAGS=-DGLX_GLXEXT_LEGACY -Wno-dev  ../visit-vtk-5.8

  Check if CMakeCache.txt has VTK_OPENGL_HAS_OSMESA:BOOL=ON, if not, turn on and rerun cmake. 
  It has to find the Mesa options and have this in CMakeCache.txt

    //Path to a file.
    OSMESA_INCLUDE_DIR:PATH=/opt/plotter/include

    //Path to a library.
    OSMESA_LIBRARY:FILEPATH=/opt/plotter/lib/libOSMesa.so
    ...
    //Use mangled Mesa with OpenGL.
    VTK_USE_MANGLED_MESA:BOOL=OFF  <-- This is OFF!!


  $ make -j 4
  $ sudo make install


  3. Build plotter
  $ cd ~/Software
  $ svn co https://svn.ccs.ornl.gov/svn-ewok/wf.src/trunk/plotter 
    ...OR...
  $ tar zxf ~/adiosvm/plotterpackages/plotter.tar.gz
  $ cd plotter
  
  Edit Makefile.adiosVM to point to the correct ADIOS, HDF5 (sequential), NetCDF (sequential), Grace and VTK libraries. 

  $ make
  $ sudo INSTALL_DIR=/opt/plotter INSTALL_CMD=install make install
  
  In ~/.bashrc, add to PATH "/opt/plotter/bin"




VIII. Clean-up a bit
==================
Not much space left after building visit and plotter. You can remove this big offenders

1.9GB  ~/Software/plotter/vtk-5.8-build
1.3GB  ~/Software/visit/qt-everywhere-opensource-src-4.8.3
       ~/Software/visit/qt-everywhere-opensource-src-4.8.3.tar.gz
       ~/Software/visit/VTK*-build




IX. Installing R and pbdR 
===========================

This is for the pbdR tutorial. Not required for an ADIOS-only tutorial. 

Packages that will be needed: 
$ sudo apt-get install libcurl4-gnutls-dev

Install R
---------
See instructions on http://cran.r-project.org/bin/linux/ubuntu/

$ sudo vi /etc/apt/sources.list
add a line to the end:
deb http://mirrors.nics.utk.edu/cran/bin/linux/ubuntu xenial/

Add the key for this mirror
$ sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
$ sudo apt-get update
$ sudo apt-get install r-base r-base-dev
$ sudo apt-get install openssl libssl-dev libssh2-1-dev libcurl4-openssl-dev

If these above don't work, you need to resort building from source.
from http://cran.r-project.org/sources.html
e.g. http://cran.r-project.org/src/base/R-3/R-3.1.2.tar.gz


--- non-SUDO version below, SUDO below ---

    $ sudo mkdir -p /opt/R/library
    $ sudo chgrp -R adios /opt/R
    $ sudo chmod -R g+w /opt/R
    
    $ export R_LIBS_USER=/opt/R/library
    $ R
    install.packages("ggplot2", repos="https://cran.cnr.berkeley.edu/", lib="/opt/R/library")
    install.packages("rlecuyer", repos="http://mirrors.nics.utk.edu/cran/", lib="/opt/R/library")
    install.packages("raster", repos="http://mirrors.nics.utk.edu/cran/", lib="/opt/R/library")
    install.packages("devtools", repos="http://mirrors.nics.utk.edu/cran/")
    library(devtools)
    install_github("RBigData/pbdMPI")
    install_github("RBigData/pbdSLAP")
    install_github("wrathematics/RNACI")
    install_github("RBigData/pbdBASE")
    install_github("RBigData/pbdDMAT")
    install_github("RBigData/pbdDEMO")
    install_github("RBigData/pbdPAPI")
    install.packages("pbdPROF", repos="http://mirrors.nics.utk.edu/cran/")
    quit()
    
    
    $ cd ~/Software
    $ git clone https://github.com/sgn11/pbdADIOS.git
    $ R CMD INSTALL pbdADIOS --configure-args="--with-adios-home=/opt/adios1/" --no-test-load
    
    Add to ~/.bashrc
        export R_LIBS_USER=/opt/R/library


--- SUDO version below, non-SUDO above ---

    Install 'rlecuyer' 
    ------------------
    from http://cran.r-project.org/web/packages/rlecuyer/index.html
    
    $ sudo R
    > install.packages("rlecuyer", repos="http://mirrors.nics.utk.edu/cran/")
    should get messages ending with:
    * DONE (rlecuyer)
    > q()
    quit R (or ctrl+D)
    
    Test if it is installed correctly:
    $ R
    > library(help=rlecuyer)
    > q()
    
    > install.packages("raster", repos="http://mirrors.nics.utk.edu/cran/")
    
    Install pbdR
    -------------
    
    $ sudo R
    install.packages("devtools", repos="http://mirrors.nics.utk.edu/cran/")
    library(devtools)
    install_github("RBigData/pbdMPI")
    install_github("RBigData/pbdSLAP")
    install_github("wrathematics/RNACI")
    install_github("RBigData/pbdBASE") 
    install_github("RBigData/pbdDMAT") 
    install_github("RBigData/pbdDEMO") 
    install_github("RBigData/pbdPAPI") 
    install.packages("pbdPROF", repos="http://mirrors.nics.utk.edu/cran/")
    > q()
    
    
    Install pbdADIOS
    ----------------
    $ cd ~/Software
    $ git clone https://github.com/sgn11/pbdADIOS.git
    $ sudo R CMD INSTALL pbdADIOS --configure-args="--with-adios-home=/opt/adios1/" --no-test-load
    -- quick test
    $ R
    > library(pbdADIOS)
    Loading required package: pbdMPI
    Loading required package: rlecuyer
    > quit()
    
--- end of SUDO version above ---

Quick test of pbdR
------------------
$ cd ~/adiosvm
$ mpirun -np 2 Rscript test_pbdR.r 
COMM.RANK = 0
[1] 0
COMM.RANK = 1
[1] 1


Download pbdR tutorial examples
-------------------------------
$ cd 
$ wget https://github.com/RBigData/RBigData.github.io/blob/master/tutorial/scripts.zip?raw=true
$ unzip  scripts.zip?raw=true

or get it using a browser from http://r-pbd.org/tutorial

Quick test of scripts:
----------------------
$ scripts/pbdMPI/quick_examples
$ mpirun -np 4 Rscript 1_rank.r
COMM.RANK = 0
[1] 0
COMM.RANK = 1
[1] 1
COMM.RANK = 2
[1] 2
COMM.RANK = 3
[1] 3


Install R Studio
----------------
$ sudo apt-get install libjpeg62
$ cd adiosvm/Rpackages
 if not there, get it from the web
$ wget http://download1.rstudio.org/rstudio-0.98.1091-i386.deb

$ sudo dpkg -i rstudio-0.98.1091-i386.deb


VII. Others
===========
  
Enable GDB debugging
--------------------
See http://askubuntu.com/questions/146160/what-is-the-ptrace-scope-workaround-for-wine-programs-and-are-there-any-risks
Ubuntu prohibits ptrace to see other processes so gdb will fail with permissions. 
Enable GDB binary to see your processes:

  $ sudo apt-get install libcap2-bin 
  $ sudo setcap cap_sys_ptrace=eip /usr/bin/gdb


GLOBAL/GTAGS
-------------
   GTAGS is useful for quickly finding definitions of functions in source codes in VIM
   It needs to be installed from source to make it work in VIM
   $ cd ~/Software
   $ tar zxf ~/adiosvm/linuxpackages/global-6.3.4.tar.gz
   $ cd global-6.3.4
   $ ./configure
   $ make
   $ sudo make install

   $ mkdir -p ~/.vim/plugin
   $ cp ~/adiosvm/linuxpackages/gtags.vim ~/.vim/plugin/
   NOTE: .vimrc already contains the flag to turn it on

   Generate the tags for a source
   $ cd ~/Software/ADIOS/src
   $ gtags
   $ ls G*
   GPATH  GRTAGS  GSYMS  GTAGS

   $ vi write/adios_posix.c
   /adios_posix_read_version
   Hit Ctrl-\ Ctrl-\
   Vi should open new file core/adios_bp_v1.c and jump to the definition of adios_posix_read_version
   :b#    -- to go back to the previous file

   If there are multiple finds, you can move among them with Ctrl-\ Ctrl-[   and  Ctrl-\ Ctrl-] 
   The commands can be modified at the bottom of ~/.vim/plugin/gtags.vim


   Add a path of any software so that gtags will search the GTAGS file there to find external references. 
   In .bashrc, add
   export GTAGSLIBPATH=/home/adios/Software/ADIOS/src
 
   You also need to run gtags in your application (top) source directory too, to get global/gtag working.



