Minimal Visual Bag of Words Image Classifier
============================================

Implementation of a content based image classifier using the visual bag of words approach. 

It relies on:
 - SIFT features as local features
 - k-means for generation of the words via clustering
 - SVM as classifier using the LIBSVM library

The two main files are `learn.py` and `classify.py`.
  
You can train the classifier for a specific dataset with: 

    python learn.py -d path_to_folders_with_images

To classify images use:

    python classify.py -c path_to_folders_with_images/codebook.file -m path_to_folders_with_images/trainingdata.svm.model images_you_want_to_classify


Run this from the folder to install the necessary libraries:

    # installing libsvm
    wget -O libsvm.tar.gz http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+tar.gz
    tar -xzf libsvm.tar.gz
    mkdir libsvm
    cp -r libsvm-*/* libsvm/
    rm -r libsvm-*/
    cd libsvm
    make
    cp tools/grid.py ../grid.py
    cd ..
    
    # installing sift
    wget http://www.cs.ubc.ca/~lowe/keypoints/siftDemoV4.zip
    unzip siftDemoV4.zip
    cp sift*/sift sift