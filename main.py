# ------------ utf-8 encoding ----------------
from autodiff.diff import Tensor
import os
import sys
import json
import numpy as np

if __name__ == "__main__":
    ob1 = Tensor(data=[1, 2, 3], dtype=np.float16, requires_grad=True)
    ob2 = Tensor(data=[3, 10, 12], dtype=np.float16, requires_grad=True)
    out1 = ob1 + ob2
    c = Tensor(data=[3, 6, 9], dtype=np.float16, requires_grad=True)
    out2 = c.log()
    out = out1 + out2
    o = out.mean()
    o.backpropogate()
    print("obj1 is ", ob1)
    print("obk2 is ", ob2)
    print("c is ", c)
    print("out1 is ", out1)
    print("out2 is ", out2)
    print("out is ", out)
    print()
    print("=========================================")
    print()
    print("final output grad is ", o.grad)
    print("third output grad is ", out.grad)
    print("second output grad is ", out2.grad)
    print("first out grad is ", out1.grad)
    print("input1 grad is ", ob1.grad)
    print("input2 grad is ", ob2.grad)
    print("input3 grad is ", c.grad)
