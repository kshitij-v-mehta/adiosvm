#!/usr/bin/env python3

import sys


#----------------------------------------------------------------------------#
def check_argv():
    if len(sys.argv) < 2:
        print("Need arguments: L num_steps (optional, default=1)")
        sys.exit(1)
    
    size = int(sys.argv[1])
    num_steps = 1
    if (len(sys.argv) == 3): num_steps = int(sys.argv[2])
    return (size, num_steps)


#----------------------------------------------------------------------------#
(size, num_steps) = check_argv()
print("Size: {}, num_steps: {}".format(size, num_steps))

outsize = (size**3) *2*8*num_steps

format_str = "MB"
divider = (2**20)

if outsize >= (2**20):
    format_str = "MB"
    divider = 2**20

if outsize >= (2**30):
    format_str = "GB"
    divider = 2**30

if outsize >= (2**40):
    format_str = "TB"
    divider = 2**40

outsize_formatted = round(outsize/divider, 2)
print("Size of output data: {} {}".format(outsize_formatted, format_str))

