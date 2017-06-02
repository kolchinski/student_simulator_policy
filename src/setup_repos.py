from __future__ import print_function

import subprocess
import os

if __name__ == "__main__":

    # get all of the subrepos
    if not os.path.exists("dkt"):
        subprocess.call("git clone https://github.com/mmkhajah/dkt".split())
        subprocess.call("2to3 -w dkt/dkt.py".split())
    if not os.path.exists("Student-HMM"):
        subprocess.call("git clone https://github.com/jrollinson/Student-HMM.git".split())
        subprocess.call("python Student-HMM/setup.py install")
    if not os.path.exists("edu"):
        subprocess.call("git clone https://github.com/jrollinson/edu".split())
        subprocess.call("python edu/setup.py install")

    print()
    print("Done importing!")
