# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/teng/cppprogram/pcdregister

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/teng/cppprogram/pcdregister/build

# Include any dependencies generated for this target.
include CMakeFiles/pcdregister.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pcdregister.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pcdregister.dir/flags.make

CMakeFiles/pcdregister.dir/pcdregister.cpp.o: CMakeFiles/pcdregister.dir/flags.make
CMakeFiles/pcdregister.dir/pcdregister.cpp.o: ../pcdregister.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/teng/cppprogram/pcdregister/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pcdregister.dir/pcdregister.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pcdregister.dir/pcdregister.cpp.o -c /home/teng/cppprogram/pcdregister/pcdregister.cpp

CMakeFiles/pcdregister.dir/pcdregister.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcdregister.dir/pcdregister.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/teng/cppprogram/pcdregister/pcdregister.cpp > CMakeFiles/pcdregister.dir/pcdregister.cpp.i

CMakeFiles/pcdregister.dir/pcdregister.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcdregister.dir/pcdregister.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/teng/cppprogram/pcdregister/pcdregister.cpp -o CMakeFiles/pcdregister.dir/pcdregister.cpp.s

# Object files for target pcdregister
pcdregister_OBJECTS = \
"CMakeFiles/pcdregister.dir/pcdregister.cpp.o"

# External object files for target pcdregister
pcdregister_EXTERNAL_OBJECTS =

pcdregister: CMakeFiles/pcdregister.dir/pcdregister.cpp.o
pcdregister: CMakeFiles/pcdregister.dir/build.make
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_people.so
pcdregister: /usr/lib/x86_64-linux-gnu/libboost_system.so
pcdregister: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
pcdregister: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
pcdregister: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
pcdregister: /usr/lib/x86_64-linux-gnu/libboost_regex.so
pcdregister: /usr/lib/x86_64-linux-gnu/libqhull.so
pcdregister: /usr/lib/libOpenNI.so
pcdregister: /usr/lib/libOpenNI2.so
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libfreetype.so
pcdregister: /usr/lib/x86_64-linux-gnu/libz.so
pcdregister: /usr/lib/x86_64-linux-gnu/libjpeg.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpng.so
pcdregister: /usr/lib/x86_64-linux-gnu/libtiff.so
pcdregister: /usr/lib/x86_64-linux-gnu/libexpat.so
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_features.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_search.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_io.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
pcdregister: /usr/lib/x86_64-linux-gnu/libpcl_common.so
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libfreetype.so
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
pcdregister: /usr/lib/x86_64-linux-gnu/libz.so
pcdregister: /usr/lib/x86_64-linux-gnu/libGLEW.so
pcdregister: /usr/lib/x86_64-linux-gnu/libSM.so
pcdregister: /usr/lib/x86_64-linux-gnu/libICE.so
pcdregister: /usr/lib/x86_64-linux-gnu/libX11.so
pcdregister: /usr/lib/x86_64-linux-gnu/libXext.so
pcdregister: /usr/lib/x86_64-linux-gnu/libXt.so
pcdregister: CMakeFiles/pcdregister.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/teng/cppprogram/pcdregister/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pcdregister"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pcdregister.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pcdregister.dir/build: pcdregister

.PHONY : CMakeFiles/pcdregister.dir/build

CMakeFiles/pcdregister.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pcdregister.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pcdregister.dir/clean

CMakeFiles/pcdregister.dir/depend:
	cd /home/teng/cppprogram/pcdregister/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/teng/cppprogram/pcdregister /home/teng/cppprogram/pcdregister /home/teng/cppprogram/pcdregister/build /home/teng/cppprogram/pcdregister/build /home/teng/cppprogram/pcdregister/build/CMakeFiles/pcdregister.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pcdregister.dir/depend

