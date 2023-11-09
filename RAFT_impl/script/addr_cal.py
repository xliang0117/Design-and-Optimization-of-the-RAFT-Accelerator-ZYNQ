


layer = { # type: 0:only3x3 1:only1x1 2:3x3+1x1 4.res

    "EA-3":{ "type":0,  "bn":False, "relu":False,   "inch":3,       "outch":3,      "inddr": "E-input",     "outddr": "E-x-3"},
    "EA-1":{ "type":1,  "bn":True,  "relu":True,    "inch":3,       "outch":32,     "inddr": "E-x-3",       "outddr": "E-x-1"},
    "EP1": { "type":1,  "bn":False, "relu":False,   "inch":32,      "outch":64,     "inddr": "E-x-1",       "outddr": "E-EP1"},
    "EE":  { "type":1,  "bn":True,  "relu":True,    "inch":32,      "outch":16,     "inddr": "E-x-1",       "outddr": "E-L2y1"},
    "EF-3":{ "type":0,  "bn":False, "relu":False,   "inch":16,      "outch":16,     "inddr": "E-L2y1",      "outddr": "E-L2y2-3"},
    "EF-1":{ "type":1,  "bn":True,  "relu":True,    "inch":16,      "outch":16,     "inddr": "E-L2y2-3",    "outddr": "E-L2y2-1"},
    "EG":  { "type":1,  "bn":True,  "relu":True,    "inch":16,      "outch":64,     "inddr": "E-L2y2-1",    "outddr": "E-L2y3"},
    "ER":  { "type":4,  "bn":False, "relu":False,   "inch":0,       "outch":0,      "inddr": "E-EP1",       "outddr": "E-ER1"},
    "EP2": { "type":1,  "bn":False, "relu":False,   "inch":64,      "outch":96,     "inddr": "E-ER1",       "outddr": "E-EP2"},
    "EH":  { "type":1,  "bn":True,  "relu":True,    "inch":64,      "outch":24,     "inddr": "E-ER1",       "outddr": "E-L3y1"},
    "EI-3":{ "type":0,  "bn":False, "relu":False,   "inch":24,      "outch":24,     "inddr": "E-L3y1",      "outddr": "E-L3y2-3"},
    "EI-1":{ "type":1,  "bn":True,  "relu":True,    "inch":24,      "outch":24,     "inddr": "E-L3y2-3",    "outddr": "E-L3y2-1"},
    "EJ":  { "type":1,  "bn":True,  "relu":True,    "inch":24,      "outch":96,     "inddr": "E-L3y2-1",    "outddr": "E-L3y3"},
    "ER2": { "type":4,  "bn":False, "relu":False,   "inch":0,       "outch":0,      "inddr": "E-EP2",       "outddr": "E-y3"},
    "EK":  { "type":1,  "bn":False, "relu":False,   "inch":96,      "outch":96,     "inddr": "E-y3",        "outddr": "E-fmap1"},

    "CA-3":{ "type":0,  "bn":False, "relu":False,   "inch":3,       "outch":3,      "inddr": "C-input",     "outddr": "E-x-3"},
    "CA-1":{ "type":1,  "bn":True,  "relu":True,    "inch":3,       "outch":32,     "inddr": "E-x-3",       "outddr": "E-x-1"},
    "CE":  { "type":1,  "bn":False, "relu":False,   "inch":32,      "outch":16,     "inddr": "E-x-1",       "outddr": "E-L2y1"},
    "CF-3":{ "type":0,  "bn":True,  "relu":True,    "inch":16,      "outch":16,     "inddr": "E-L2y1",      "outddr": "E-L2y2-3"},
    "CF-1":{ "type":1,  "bn":False, "relu":False,   "inch":16,      "outch":16,     "inddr": "E-L2y2-3",    "outddr": "E-L2y2-1"},
    "CG":  { "type":1,  "bn":True,  "relu":True,    "inch":16,      "outch":64,     "inddr": "E-L2y2-1",    "outddr": "E-L2y3"},
    "CP":  { "type":1,  "bn":True,  "relu":True,    "inch":32,      "outch":64,     "inddr": "E-x-1",       "outddr": "E-EP1"},
    "CR":  { "type":4,  "bn":False, "relu":False,   "inch":0,       "outch":0,      "inddr": "E-EP1",       "outddr": "E-ER1"},
    "CH":  { "type":1,  "bn":False, "relu":False,   "inch":64,      "outch":24,     "inddr": "E-ER1",       "outddr": "E-L3y1"},
    "CI-3":{ "type":0,  "bn":True,  "relu":True,    "inch":24,      "outch":24,     "inddr": "E-L3y1",      "outddr": "E-L3y2-3"},
    "CI-1":{ "type":1,  "bn":False, "relu":False,   "inch":24,      "outch":24,     "inddr": "E-L3y2-3",    "outddr": "E-L3y2-1"},
    "CJ":  { "type":1,  "bn":True,  "relu":True,    "inch":24,      "outch":96,     "inddr": "E-L3y2-1",    "outddr": "E-L3y3"},
    "CP2": { "type":1,  "bn":True,  "relu":True,    "inch":64,      "outch":96,     "inddr": "E-ER1",       "outddr": "E-EP2"},
    "CR2": { "type":4,  "bn":False, "relu":False,   "inch":0,       "outch":0,      "inddr": "E-EP2",       "outddr": "E-y3"},
    "Cinp":{ "type":1,  "bn":False, "relu":False,   "inch":96,      "outch":64,     "inddr": "E-y3",        "outddr": "inp"},
    "Cnet":{ "type":1,  "bn":False, "relu":False,   "inch":96,      "outch":96,     "inddr": "E-y3",        "outddr": "net"},


    "A":{ "type":1,     "bn":False, "relu":False,   "inch":196,     "outch":96,     "inddr": "corr",        "outddr": "cor"},
    "B":{ "type":2,     "bn":False, "relu":True,    "inch":2,       "outch":64,     "inddr": "flow",        "outddr": "flo1"},
    "C":{ "type":2,     "bn":False, "relu":True,    "inch":64,      "outch":32,     "inddr": "flo1",        "outddr": "flo2"},
    "D":{ "type":2,     "bn":False, "relu":True,    "inch":128,     "outch":80,     "inddr": "cor",         "outddr": "out"},
    "E":{ "type":2,     "bn":False, "relu":False,   "inch":242,     "outch":96,     "inddr": "net",         "outddr": "q_pre"},
    "F":{ "type":2,     "bn":False, "relu":False,   "inch":242,     "outch":96,     "inddr": "net",         "outddr": "z"},
    "G":{ "type":2,     "bn":False, "relu":False,   "inch":242,     "outch":96,     "inddr": "inp",         "outddr": "net"},
    "H":{ "type":2,     "bn":False, "relu":True,    "inch":96,      "outch":96,     "inddr": "net",         "outddr": "f_head"},
    "I":{ "type":2,     "bn":False, "relu":False,   "inch":128,     "outch":2,      "inddr": "f_head",      "outddr": "flow"},

    "Corr1":{ "type":4, "bn":False, "relu":False,   "inch":0,       "outch":0,      "inddr": "E-fmap2",     "outddr": "Corr1"},
    "Corr2":{ "type":4, "bn":False, "relu":False,   "inch":0,       "outch":0,      "inddr": "E-fmap2",     "outddr": "Corr2"},
    "Corr3":{ "type":4, "bn":False, "relu":False,   "inch":0,       "outch":0,      "inddr": "E-fmap2",     "outddr": "Corr3"},
    "Corr4":{ "type":4, "bn":False, "relu":False,   "inch":0,       "outch":0,      "inddr": "E-fmap2",     "outddr": "Corr4"}
}

ddr = {
    # static
    "PRESERVE": {"channel":1,   "height":1,     "width": 513,   "start":None},

    "E-input":  {"channel":3,   "height":384,   "width": 512,   "start":None},
    "C-input":  {"channel":3,   "height":384,   "width": 512,   "start":None},

    # dynamic
    "E-x-3":    {"channel":3,   "height":192,   "width": 256,   "start":None},
    "E-x-1":    {"channel":32,  "height":192,   "width": 256,   "start":None},
    "E-L2y1":   {"channel":16,  "height":192,   "width": 256,   "start":None},
    "E-L2y2-3": {"channel":16,  "height":96,    "width": 128,   "start":None},
    "E-L2y2-1": {"channel":16,  "height":96,    "width": 128,   "start":"E-L2y1"},
    "E-L2y3":   {"channel":64,  "height":96,    "width": 128,   "start":None},
    "E-EP1":    {"channel":64,  "height":96,    "width": 128,   "start":None},
    "E-ER1":    {"channel":64,  "height":96,    "width": 128,   "start":"E-x-3"},
    "E-L3y1":   {"channel":24,  "height":96,    "width": 128,   "start":None},
    "E-L3y2-3": {"channel":24,  "height":48,    "width": 64,    "start":None},
    "E-L3y2-1": {"channel":24,  "height":48,    "width": 64,    "start":None},
    "E-L3y3":   {"channel":96,  "height":48,    "width": 64,    "start":None},
    "E-EP2":    {"channel":96,  "height":48,    "width": 64,    "start":None},
    "E-y3":     {"channel":96,  "height":48,    "width": 64,    "start":None},

    "corr":     {"channel":196, "height":48,    "width": 64,    "start":"E-x-3"},
    "cor":      {"channel":96,  "height":48,    "width": 64,    "start":None},
    "flo2":     {"channel":32,  "height":48,    "width": 64,    "start":None},
    "flo1":     {"channel":64,  "height":48,    "width": 64,    "start":None},
    "net":      {"channel":96,  "height":48,    "width": 64,    "start":None},
    "inp":      {"channel":64,  "height":48,    "width": 64,    "start":None},
    "out":      {"channel":80,  "height":48,    "width": 64,    "start":None},
    "flow":     {"channel":2,   "height":48,    "width": 64,    "start":None},
    "q_pre":    {"channel":96,  "height":48,    "width": 64,    "start":None},
    "z":        {"channel":96,  "height":48,    "width": 64,    "start":None},
    "f_head":   {"channel":96,  "height":48,    "width": 64,    "start":None},

    "E-fmap1":  {"channel":96,  "height":48,    "width": 64,    "start":"static"},
    "E-fmap2":  {"channel":96,  "height":48,    "width": 64,    "start": None},
    "Corr1":    {"channel":3072,"height":48,    "width": 64,    "start": None},
    "Corr2":    {"channel":3072,"height":24,    "width": 32,    "start": None},
    "Corr3":    {"channel":3072,"height":12,    "width": 16,    "start": None},
    "Corr4":    {"channel":3072,"height":6,     "width": 8,     "start": None},
}


if __name__ == "__main__":
    addr = 0
    max_addr = 9
    ddr_addr = {}
    print("-----------------------ddr addr-----------------------")
    for key, value in ddr.items():
        channel_num = value["channel"]
        img_height = value["height"]
        img_width = value["width"]

        if value["start"] != None:
            if value["start"] == "static":
                addr = max_addr
            else:
                addr = ddr_addr[value["start"]]

        if channel_num % 8 != 0:
            channel_num = (channel_num // 8 + 1) * 8
        ddr_addr[key] = addr
        addr += channel_num * img_height * img_width // 8
        if addr > max_addr:
            max_addr = addr

    for key, value in ddr_addr.items():
        print("{:<10}->  {:<10d}".format(key, value))

    print("-----------------------feature map addr-----------------------")
    index = 0
    for key, value in layer.items():
        if value["type"] == 0 or value["type"] == 2:
            in_addr = ddr_addr[value["inddr"]] - ddr[value["inddr"]]["width"] - 1   # for padding
        else:
            in_addr = ddr_addr[value["inddr"]]

        if value["relu"]:
            print("{{\"{:<5}\",\t{:<7d},\t{:<7d},\ttrue }}, \t//{}".format(key, in_addr, ddr_addr[value["outddr"]], index))
        else:
            print("{{\"{:<5}\",\t{:<7d},\t{:<7d},\tfalse}}, \t//{}".format(key, in_addr, ddr_addr[value["outddr"]], index))
        if key == "EK" or key == "Cnet":
            index = 0
        else:
            index += 1



    addr = 0
    index = 0
    print("-----------------------weight addr-----------------------")
    for key, value in layer.items():
        inchannel = value["inch"]
        outchannel = value["outch"]

        if inchannel % 16 != 0:
            inchannel = (inchannel // 8 + 1) * 8
        if outchannel % 32 != 0:
            outchannel = (outchannel // 32 + 1) * 32

        if value["type"] == 0 or value["type"] == 2:
            w3_addr = addr
            addr += inchannel * 9 // 8
        else:
            w3_addr = 0

        if value["bn"]:
            bn_addr = addr
            addr += outchannel // 16
        else:
            bn_addr = 0

        if value["type"] == 0:
            w1_addr = 0
        else:
            w1_addr = addr


        print("{{\"{:<4}\",\t{:<4d},\t{:<4d},\t{:<4d}}}, //{}".format(key, w3_addr, w1_addr, bn_addr, index))
        
        if value["type"] == 1 or value["type"] == 2:
            addr += inchannel * outchannel // 16
        
        if key == "EK" or key == "Cnet":
            index = 0
        else:
            index += 1
