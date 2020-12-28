import torch

def main():
    print("This is tested on CPU\n")
    a = torch.zeros(10, 10, dtype=torch.int32)
    print(f"This is a = \n{a}")
    print(f"id(a) = {id(a)}\n")

    a_conti = torch.Tensor.contiguous(a[5:])
    print(f"This is contiguous a = \n{a_conti}")
    print(f"id(contiguous a) = {id(a_conti)}\n")

    a_conti[:] = 999
    print(f"This is contiguous a = \n{a_conti}")
    print(f"id(contiguous a) = {id(a_conti)}\n")
    print(f"This is a = \n{a}")
    print(f"id(a) = {id(a)}\n")

    if torch.cuda.is_available():
        print("This is tested on GPU\n")

        b = torch.zeros(10, 10, dtype=torch.int32).cuda(0)
        print(f"This is b = \n{b}")
        print(f"id(b) = {id(b)}\n")

        b_conti = torch.Tensor.contiguous(b[5:])
        print(f"This is contiguous b = \n{b_conti}")
        print(f"id(contiguous b) = {id(b_conti)}\n")

        b_conti[:] = 999
        print(f"This is contiguous b = \n{b_conti}")
        print(f"id(contiguous b) = {id(b_conti)}\n")
        print(f"This is b = \n{b}")
        print(f"id(b) = {id(b)}\n")



if __name__ == "__main__":
    main()


'''
Result of the code

This is tested on CPU

This is a = 
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
id(a) = 2037422431992

This is contiguous a = 
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
id(contiguous a) = 2037422313944

This is contiguous a = 
tensor([[999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999]], dtype=torch.int32)
id(contiguous a) = 2037422313944

This is a = 
tensor([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999]], dtype=torch.int32)
id(a) = 2037422431992

This is tested on GPU

This is b = 
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0', dtype=torch.int32)
id(b) = 2037485341976

This is contiguous b = 
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0', dtype=torch.int32)
id(contiguous b) = 2037451163992

This is contiguous b = 
tensor([[999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999]], device='cuda:0',
       dtype=torch.int32)
id(contiguous b) = 2037451163992

This is b = 
tensor([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999]], device='cuda:0',
       dtype=torch.int32)
id(b) = 2037485341976
'''