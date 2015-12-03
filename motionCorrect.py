#!/usr/bin/env python

#
# Copyright (c) 2015 Wen Jiang <jiang12@purdue.edu>
# 
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
 
import os, sys, argparse

def main():
	args =  parse_command_line()
	
	tasks = []
	for i, m in enumerate(args.movieFiles):
		tasks += [ (args.movieFiles, i, args) ]

	for t in tasks:
		processOneMovie(t)

def processOneMovie(task):
	movies, index, args = task
	methodMapping = {"dosefgpu_driftcorr" : dosefgpu_driftcorr} # UCSF method

	methodMapping[args.method](task)

def dosefgpu_driftcorr(task):   # UCSF method
	movies, index, args = task
	m = movies[index]

	print "Processing %d/%d movies: %s" % (index+1, len(movies), m)

	m2 = os.path.splitext(m)[0]

	# find bin level from options
	import re
	f = re.findall("^.*-bin\s+(?P<bin>\d*).*", args.forwardedOptions)
	if f:
		bin = int(f[0])
	else:
		bin = 1

	if bin>1:
		logFile = "%s_%dx_Log.txt" % (m2, bin)
	else:
		logFile = "%s_Log.txt" % (m2)

	if os.path.exists(logFile) and args.force==0:
		print "\t%s is already aligned. skipped" % (m)
	else:
		cmd = "dosefgpu_driftcorr %s %s" % (m, args.forwardedOptions)
		print "\t%s" % (cmd)
		os.system(cmd)

	logFile2 = "stack_%04d_2x_Log.txt" % (index)   # for interactive screening using dosef_logviewer program
	if os.path.exists(logFile):
		try:
			os.symlink(logFile, logFile2)
		except OSError, e:
			import errno
			if e.errno == errno.EEXIST:
				os.remove(logFile2)
				os.symlink(logFile, logFile2)
			else:
				print "Error: linking %s to %s" % (logFile, logFile2)
				raise e

def parse_command_line():
	description = "Perform motion correction of cryo-EM movies"
	epilog  = "Author: Wen Jiang (jiang12@purdue.edu)\n";
	epilog += "Copyright (c) 2015 Purdue University\n";
	epilog += "$Revision$\n";
	epilog += "$LastChangedDate$\n\n";

	parser = argparse.ArgumentParser(description=description, epilog=epilog)

	parser.add_argument('movieFiles', metavar="<filename>", type=str, nargs="+", help='movie files to process')

	methods = ["dosefgpu_driftcorr"]
	parser.add_argument('--method', metavar='<%s>' % ('|'.join(methods)), type=str, choices=methods, help='which method to use. default to %(default)s', default="dosefgpu_driftcorr")

	parser.add_argument("--forwardedOptions", metavar="<options>", type=str, help="forward additional options to the motion correction program. default to \"%(default)s\"", default="")

	parser.add_argument("--force", metavar="<0|1>", type=int, help="force reprocess all movies. default to  %(default)s", default=0)

	parser.add_argument("--verbose", metavar="<n>", type=int, help="verbose level. default to %(default)s", default=0)

	args = parser.parse_args()

	return args


if __name__== "__main__":
	main()

