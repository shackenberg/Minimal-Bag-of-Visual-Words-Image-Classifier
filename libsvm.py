#!/usr/bin/env python

"""
Taken and modified from easy.py from the libsvm package,
which is under following copyright:

Copyright (c) 2000-2012 Chih-Chung Chang and Chih-Jen Lin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys
import os
from subprocess import *
import numpy


def cmd_output(args, **kwds):
  # get the output of the subprocess
  kwds.setdefault("stdout", subprocess.PIPE)
  kwds.setdefault("stderr", subprocess.STDOUT)
  p = subprocess.Popen(args, **kwds)
  return p.communicate()[0]

def test(test_pathname, model_file):
	is_win32 = (sys.platform == 'win32')
	if not is_win32:
		svmscale_exe = "libsvm/svm-scale"
		svmtrain_exe = "libsvm/svm-train"
		svmpredict_exe = "libsvm/svm-predict"
		grid_py = "./grid.py"
		gnuplot_exe = "/usr/bin/gnuplot"
	else:
	        # example for windows
		svmscale_exe = r"..\windows\svm-scale.exe"
		svmtrain_exe = r"..\windows\svm-train.exe"
		svmpredict_exe = r"..\windows\svm-predict.exe"
		gnuplot_exe = r"c:\tmp\gnuplot\binary\pgnuplot.exe"
		grid_py = r".\grid.py"

	assert os.path.exists(svmscale_exe),"svm-scale executable not found"
	assert os.path.exists(svmtrain_exe),"svm-train executable not found"
	assert os.path.exists(svmpredict_exe),"svm-predict executable not found"
	assert os.path.exists(grid_py),"grid.py not found"

	assert os.path.exists(test_pathname),"training file not found"
	trunc_filename = os.path.splitext(model_file)[0]
	scaled_test_file = trunc_filename + ".scale"
	range_file = trunc_filename + ".range"
	predict_test_file = trunc_filename + ".prediction"

	cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
	print('Scaling testing data...')
	Popen(cmd, shell = True, stdout = PIPE).communicate()

	cmd = '{0} "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
	print('Testing...')
	print('Output prediction: {0}'.format(predict_test_file))
	result = Popen(cmd, shell = True, stdout = PIPE).communicate()
	pred_class = []
	with open(predict_test_file,'r') as f:
		 for line in f:
			pred_class.append(int(line))

	#print result
	pred_class = numpy.asarray(pred_class)
	result = result[0]
	pivot = result.find(' = ')
	result = result[pivot+3:]
	pivot = result.find('%')
	result = result[:pivot]
	accuracy = float(result)
	return pred_class


def grid(train_pathname,test_pathname=None, png_filename=None):
	is_win32 = (sys.platform == 'win32')
	if not is_win32:
		svmscale_exe = "libsvm/svm-scale"
		svmtrain_exe = "libsvm/svm-train"
		svmpredict_exe = "libsvm/svm-predict"
		grid_py = "./grid.py"
		gnuplot_exe = "/usr/bin/gnuplot"
	else:
	        # example for windows
		svmscale_exe = r"..\windows\svm-scale.exe"
		svmtrain_exe = r"..\windows\svm-train.exe"
		svmpredict_exe = r"..\windows\svm-predict.exe"
		gnuplot_exe = r"c:\tmp\gnuplot\binary\pgnuplot.exe"
		grid_py = r".\grid.py"

	assert os.path.exists(svmscale_exe),"svm-scale executable not found"
	assert os.path.exists(svmtrain_exe),"svm-train executable not found"
	assert os.path.exists(svmpredict_exe),"svm-predict executable not found"
	assert os.path.exists(grid_py),"grid.py not found"

	assert os.path.exists(train_pathname),"training file not found"
	scaled_file = train_pathname + ".scale"
	model_file = train_pathname + ".model"
	range_file = train_pathname + ".range"

	if test_pathname != None:

		assert os.path.exists(test_pathname),"testing file not found"
		scaled_test_file = test_pathname + ".scale"
		predict_test_file = test_pathname + ".predict"
	if png_filename != None:
		png_filename = '-png {0}'.format(png_filename)
	else:
		png_filename = ''
	cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, train_pathname, scaled_file)
	print('Scaling training data...')
	Popen(cmd, shell = True, stdout = PIPE).communicate()

	cmd = 'python {0} -svmtrain "{1}" -gnuplot "{2}" {3} "{4}"'.format(grid_py, svmtrain_exe, gnuplot_exe, png_filename, scaled_file,)
	print "------------------------------"
	print cmd
	print('Cross validation...')
	f = Popen(cmd, shell = True, stdout = PIPE).stdout
	line = ''
	while True:
		last_line = line
		line = f.readline()
		if not line: break
	c,g,rate = map(float,last_line.split())

	print('Best c={0}, g={1} CV rate={2}'.format(c,g,rate))

	cmd = '{0} -c {1} -g {2} "{3}" "{4}"'.format(svmtrain_exe,c,g,scaled_file,model_file)
	print('Training...')
	Popen(cmd, shell = True, stdout = PIPE).communicate()

	print('Output model: {0}'.format(model_file))

	if test_pathname != None:
		cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
		print('Scaling testing data...')
		Popen(cmd, shell = True, stdout = PIPE).communicate()

		cmd = '{0} "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
		print('Testing...')
		print('Output prediction: {0}'.format(predict_test_file))
		result = Popen(cmd, shell = True, stdout = PIPE).communicate()
		pred_class = []
		with open(predict_test_file,'r') as f:
			 for line in f:
				pred_class.append(int(line))

		print result
		pred_class = numpy.asarray(pred_class)
		result = result[0]
		pivot = result.find(' = ')
		result = result[pivot+3:]
		pivot = result.find('%')
		result = result[:pivot]
		accuracy = float(result)
		return accuracy, pred_class, c, g, rate
	else:
		return c, g, rate, model_file
