import argparse
import sys

argparser = argparse.ArgumentParser(sys.argv[0])

argparser.add_argument("--net_size",
		nargs = 4,
		type = int,
		default = [1,1,2,3]
		)

args = argparser.parse_args()

print args
