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

        a = torch.zeros(10, 10, dtype=torch.int32).cuda(0)
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
id(a) = 2191134760696

This is contiguous a = 
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
id(contiguous a) = 2191134642648

This is contiguous a = 
tensor([[999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999]], dtype=torch.int32)
id(contiguous a) = 2191134642648

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
id(a) = 2191134760696

This is tested on GPU

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
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0', dtype=torch.int32)
id(a) = 2191197333688

This is contiguous a = 
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0', dtype=torch.int32)
id(contiguous a) = 2191134760696

This is contiguous a = 
tensor([[999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999],
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999]], device='cuda:0',
       dtype=torch.int32)
id(contiguous a) = 2191134760696

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
        [999, 999, 999, 999, 999, 999, 999, 999, 999, 999]], device='cuda:0',
       dtype=torch.int32)
id(a) = 2191197333688
'''