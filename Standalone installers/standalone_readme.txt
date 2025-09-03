The application requires the Matlab runtime, which will be downloaded when installing. You may also be prompted to install a Java runtime.

After installing, you need to manually set the path to the runtime.

Open a terminal window and first add the following entry to zshrc file.

Use the following command to add the entry:
nano ~/.zshrc

Then paste in the following:
export DYLD_LIBRARY_PATH=/Applications/MATLAB/MATLAB_Runtime/R2024b/runtime/maca64:/Applications/MATLAB/MATLAB_Runtime/R2024b/sys/os/maca64:/Applications/MATLAB/MATLAB_Runtime/R2024b/bin/maca64:

Then enter control-O to save the changes, and control-X to exit

Finally, type exit in the terminal window, then open a new window.


Usage:

While there is an application in the /Applications folder, it is not run by double clicking. Using the terminal navigate to the folder containing the image folders. input the name of the folder containing the images, as well as any arguments. In the following example, the actual corealign application is not added to the path, so the full path to it is specified in the command, as follows:

/Applications/Akita_University/corealign/application/corealign.app/Contents/MacOS/corealign 'PC-207-S3A' Crop 1

where 'PC-207-S3A' is the name of the folder containing the images
'Crop' is an argument with the value of '1', which turns on automatic cropping

The above test images can be downloaded from here:
https://drive.google.com/drive/folders/1p294k38NE67OnRJXnzsjre5JriWajFxj?usp=sharing