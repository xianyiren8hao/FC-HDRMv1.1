from pickle import dumps as p_dumps, loads as p_loads


DHEAD_LEN = 10
DHEAD_CODS = 'ascii'


def myPack(obj_s):
    bs_data = p_dumps(obj_s)
    bs_len = len(bs_data)
    bs_str = str(bs_len).zfill(DHEAD_LEN)[0:DHEAD_LEN]
    bs_head = bs_str.encode(DHEAD_CODS)
    obj_d = bs_head + bs_data
    return obj_d


def myRecv(obj_sc):
    bs_head = obj_sc.recv(DHEAD_LEN)
    bs_len = int(bs_head.decode(DHEAD_CODS))
    bs_data = b''
    while bs_len:
        bs_more = obj_sc.recv(bs_len)
        bs_len -= len(bs_more)
        bs_data += bs_more
    obj_d = p_loads(bs_data)
    return obj_d
