wget -O libsvm.tar.gz http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+tar.gz
tar -xzf libsvm.tar.gz
mkdir libsvm
cp -r libsvm-*/* libsvm/
rm -r libsvm-*/
cd libsvm
make
cp tools/grid.py ../grid.py

cd ..

wget http://www.cs.ubc.ca/~lowe/keypoints/siftDemoV4.zip
unzip siftDemoV4.zip
cp sift*/sift sift

python learn.py -d path_to_folders_with_images
