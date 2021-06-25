import os
import sys
import time


def funcs_():
    for i in range(100):
        print(i)
        time.sleep(1)
    pass


def run():
    if (len(sys.argv) != 1):
        print("args error ...")
        sys.exit(0)

    funcs_()


if __name__ == "__main__":
    run()
