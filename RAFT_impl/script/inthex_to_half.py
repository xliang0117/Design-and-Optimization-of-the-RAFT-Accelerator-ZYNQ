import struct

while 1:
    int_string = input()
    num = int(int_string)
    halfout = struct.unpack("<e", num.to_bytes(2, 'little'))
    print(halfout)