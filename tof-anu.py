import numpy as np
import json
import codecs
from gccestimating import GCC, corrlags
import matplotlib.pyplot as plt

def loadcsv(filename, delim=","):
    # Reads CSV from file
    data = np.loadtxt(filename, delimiter=delim, dtype= np.float64)
    return data

def loadjson(filename):
    # Reads JSON from file
    data = json.load(open(filename))
    # Returns as dict
    return data

def gccest(siga, sigb, samplerate=1, cctype="scot"):
    n = int((len(siga)+ len(sigb)) / 2)
    
    siga -= np.mean(siga, axis=0)
    sigb -= np.mean(sigb, axis=0)
    
    lags = corrlags(2*n-1, samplerate=samplerate)
    
    gcc = GCC(sig1=siga,sig2=sigb)
    
    match cctype.lower():
        case "cc":
            cc = gcc.cc()
        case "phat":
            cc = gcc.phat()
        case "scot":
            cc = gcc.scot()
        case"roth":
            cc = gcc.roth()
        case"ht":
            cc = gcc.ht()
        case _:
            cc = gcc.cc()
    
    cc /= np.max(np.abs(cc))    # normalize

    return cc, lags

# y = 98946x + timestamp[0]

def findtimestampavg(arr):
    return np.mean(np.diff(arr))

def timestampextrapfromavg(x,origin,arr):
    return findtimestampavg(arr)*x + origin

def tauest(cc, lags, samplerate = 1, timestamp = None):
    shift = np.argmax(np.abs(cc)) # the index of the maximum value in cc
    # no need to roll to center, it is already centered
    tau = lags[shift] / float(samplerate)  # in seconds
    if timestamp is not None:
        if len(timestamp) < 999:
            # append to timestamp
            for x in range(1,251,1): # x from 1 to 250
                timestamp = np.append(timestamp, timestampextrapfromavg(x=x,origin=timestamp[0], arr=timestamp))
            for x in range(-1,-250,-1): # x from -1 to -249
                timestamp = np.insert(timestamp, 0, timestampextrapfromavg(x=x,origin=timestamp[0], arr=timestamp))     
        peaktimestamp = timestamp[shift]
        origintimestamp = timestamp[np.argmin(np.abs(lags))]  # the timestamp corresponding to the zero lag
        tau = peaktimestamp - origintimestamp # in nanoseconds
        # timestamp needs to be extrapolated to 2x the length
        tau /= 1000000000 # convert to seconds
    return np.abs(tau), shift

def safe_divide(num, denom, default_value=0):
    if not isinstance(num, (int, float)) or not isinstance(denom, (int, float)):
        raise ValueError("Both num and denom must be numbers!")
    else:
        if denom == 0:
            return default_value
        else:
            return num / denom

def taufromsig(siga, sigb, samplerate= 1, timestamp= None):
    cc, lags = gccest(siga= siga, sigb= sigb, samplerate= samplerate)
    tau, shift = tauest(cc= cc, lags= lags, samplerate= samplerate, timestamp= timestamp)
    return tau, shift

def onetapest(sigdict: list, which: int, diameter = 0.3):
    
    # function to tap once. produces 7 ToF/tau from 7 CC, out of 8 sensors
    
    # diameters in meters
    
    sig1 = sigdict[0].get("value1")
    sig2 = sigdict[1].get("value2")
    sig3 = sigdict[2].get("value3")
    sig4 = sigdict[3].get("value4")
    sig5 = sigdict[4].get("value5")
    sig6 = sigdict[5].get("value6")
    sig7 = sigdict[6].get("value7")
    sig8 = sigdict[7].get("value8")

    timestamp = sigdict[8].get("timestamp")
    
    avgtimestamp = findtimestampavg(timestamp)
    # samplerate = 100000
    samplerate = int(1/(avgtimestamp*1e-9)) # per second
        
    radius = diameter/2
    ab = radius * 0.76536686473 # sqrt(sqrt(2)-2)
    ac = radius * 1.41421356237 # sqrt(2)
    ad = radius * 1.84775906502 # sqrt(sqrt(2)+2)
    ae = float(diameter)
    
    # ab = 12,23,34,45,56,67,78,81
    # ac = 13,24,35,46,57,68,71,82
    # ad = 14,25,36,47,58,61,72,83
    # ae = 15,26,37,48,51,62,73,84
    
    match which:
        case 1:
            tof12 = taufromsig(siga= sig1, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo12 = safe_divide(ab, tof12)
            
            tof13 = taufromsig(siga= sig1, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo13 = safe_divide(ac, tof13)
            
            tof14 = taufromsig(siga= sig1, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo14 = safe_divide(ad, tof14)
            
            tof15 = taufromsig(siga= sig1, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo15 = safe_divide(ae, tof15)
            
            tof16 = taufromsig(siga= sig1, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo16 = safe_divide(ad, tof16)
            
            tof17 = taufromsig(siga= sig1, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo17 = safe_divide(ac, tof17)
            
            tof18 = taufromsig(siga= sig1, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo18 = safe_divide(ab, tof18)
            
            return np.array((0, velo12, velo13, velo14, velo15, velo16, velo17, velo18), dtype=np.float32)
        case 2:
            
            tof21 = taufromsig(siga= sig2, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo21 = safe_divide(ab, tof21)
            
            tof23 = taufromsig(siga= sig2, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo23 = safe_divide(ab, tof23)
            
            tof24 = taufromsig(siga= sig2, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo24 = safe_divide(ac, tof24)
            
            tof25 = taufromsig(siga= sig2, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo25 = safe_divide(ad, tof25)
            
            tof26 = taufromsig(siga= sig2, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo26 = safe_divide(ae, tof26)
            
            tof27 = taufromsig(siga= sig2, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo27 = safe_divide(ad, tof27)
            
            tof28 = taufromsig(siga= sig2, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo28 = safe_divide(ac, tof28)
            
            return np.array((velo21, 0, velo23, velo24, velo25, velo26, velo27, velo28), dtype=np.float32)
        case 3:
            
            tof31 = taufromsig(siga= sig3, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo31 = safe_divide(ac, tof31)
            
            tof32 = taufromsig(siga= sig3, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo32 = safe_divide(ab, tof32)
            
            tof34 = taufromsig(siga= sig3, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo34 = safe_divide(ab, tof34)
            
            tof35 = taufromsig(siga= sig3, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo35 = safe_divide(ac, tof35)
            
            tof36 = taufromsig(siga= sig3, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo36 = safe_divide(ad, tof36)
            
            tof37 = taufromsig(siga= sig3, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo37 = safe_divide(ae, tof37)
            
            tof38 = taufromsig(siga= sig3, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo38 = safe_divide(ad, tof38)
            
            return np.array((velo31, velo32, 0, velo34, velo35, velo36, velo37, velo38), dtype=np.float32) 
        case 4:
            
            tof41 = taufromsig(siga= sig4, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo41 = safe_divide(ad, tof41)
            
            tof42 = taufromsig(siga= sig4, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo42 = safe_divide(ac, tof42)
            
            tof43 = taufromsig(siga= sig4, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo43 = safe_divide(ab, tof43)
            
            tof45 = taufromsig(siga= sig4, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo45 = safe_divide(ab, tof45)
            
            tof46 = taufromsig(siga= sig4, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo46 = safe_divide(ac, tof46)
            
            tof47 = taufromsig(siga= sig4, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo47 = safe_divide(ad, tof47)
            
            tof48 = taufromsig(siga= sig4, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo48 = safe_divide(ae, tof48)
            
            return np.array((velo41, velo42, velo43, 0, velo45, velo46, velo47, velo48), dtype=np.float32)
        case 5:
            
            tof51 = taufromsig(siga= sig5, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo51 = safe_divide(ae, tof51)
            
            tof52 = taufromsig(siga= sig5, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo52 = safe_divide(ad, tof52)
            
            tof53 = taufromsig(siga= sig5, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo53 = safe_divide(ac, tof53)
            
            tof54 = taufromsig(siga= sig5, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo54 = safe_divide(ab, tof54)
            
            tof56 = taufromsig(siga= sig5, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo56 = safe_divide(ab, tof56)
            
            tof57 = taufromsig(siga= sig5, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo57 = safe_divide(ac, tof57)
            
            tof58 = taufromsig(siga= sig5, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo58 = safe_divide(ad, tof58)
            
            return np.array((velo51, velo52, velo53, velo54, 0, velo56, velo57, velo58), dtype=np.float32)
        case 6:
            
            tof61 = taufromsig(siga= sig6, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo61 = safe_divide(ad, tof61)
            
            tof62 = taufromsig(siga= sig6, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo62 = safe_divide(ae, tof62)
            
            tof63 = taufromsig(siga= sig6, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo63 = safe_divide(ad, tof63)
            
            tof64 = taufromsig(siga= sig6, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo64 = safe_divide(ac, tof64)
            
            tof65 = taufromsig(siga= sig6, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo65 = safe_divide(ab, tof65)
            
            tof67 = taufromsig(siga= sig6, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo67 = safe_divide(ab, tof67)
            
            tof68 = taufromsig(siga= sig6, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo68 = safe_divide(ac, tof68)
            
            return np.array((velo61, velo62, velo63, velo64, velo65, 0, velo67, velo68), dtype=np.float32)
        case 7:
            
            tof71 = taufromsig(siga= sig7, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo71 = safe_divide(ac, tof71)
            
            tof72 = taufromsig(siga= sig7, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo72 = safe_divide(ad, tof72)
            
            tof73 = taufromsig(siga= sig7, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo73 = safe_divide(ae, tof73)
            
            tof74 = taufromsig(siga= sig7, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo74 = safe_divide(ad, tof74)
            
            tof75 = taufromsig(siga= sig7, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo75 = safe_divide(ac, tof75)
            
            tof76 = taufromsig(siga= sig7, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo76 = safe_divide(ab, tof76)
            
            tof78 = taufromsig(siga= sig7, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo78 = safe_divide(ab, tof78)
            
            return np.array((velo71, velo72, velo73, velo74, velo75, velo76, 0, velo78), dtype=np.float32)
        case 8:
            tof81 = taufromsig(siga= sig8, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo81 = safe_divide(ab, tof81)
            
            tof82 = taufromsig(siga= sig8, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo82 = safe_divide(ac, tof82)
            
            tof83 = taufromsig(siga= sig8, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo83 = safe_divide(ad, tof83)
            
            tof84 = taufromsig(siga= sig8, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo84 = safe_divide(ae, tof84)
            
            tof85 = taufromsig(siga= sig8, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo85 = safe_divide(ad, tof85)
            
            tof86 = taufromsig(siga= sig8, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo86 = safe_divide(ac, tof86)
            
            tof87 = taufromsig(siga= sig8, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo87 = safe_divide(ab, tof87)
            
            return np.array((velo81, velo82, velo83, velo84, velo85, velo86, velo87, 0), dtype=np.float32)
        case _:
            raise ValueError


def onetaptof(sigdict: list, which: int, diameter = 0.3):
    
    # function to tap once. produces 7 ToF/tau from 7 CC, out of 8 sensors
    
    # diameters in meters
    
    sig1 = sigdict[0].get("value1")
    sig2 = sigdict[1].get("value2")
    sig3 = sigdict[2].get("value3")
    sig4 = sigdict[3].get("value4")
    sig5 = sigdict[4].get("value5")
    sig6 = sigdict[5].get("value6")
    sig7 = sigdict[6].get("value7")
    sig8 = sigdict[7].get("value8")
    samplerate = 35961 # Samples per second
    # Why? it takes an average of 27808 nanoseconds to capture one sample
    # so 1 second / 27808 nanoseconds = 35960.8746 samples per second
    # Round up to 35961 samples per second
    timestamp = sigdict[8].get("timestamp")
        
    radius = diameter/2
    ab = radius * 0.76536686473 # sqrt(sqrt(2)-2)
    ac = radius * 1.41421356237 # sqrt(2)
    ad = radius * 1.84775906502 # sqrt(sqrt(2)+2)
    ae = float(diameter)
    
    # ab = 12,23,34,45,56,67,78,81
    # ac = 13,24,35,46,57,68,71,82
    # ad = 14,25,36,47,58,61,72,83
    # ae = 15,26,37,48,51,62,73,84
    
    match which:
        case 1:
            tof12 = taufromsig(siga= sig1, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            tof13 = taufromsig(siga= sig1, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            tof14 = taufromsig(siga= sig1, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            tof15 = taufromsig(siga= sig1, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            tof16 = taufromsig(siga= sig1, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            tof17 = taufromsig(siga= sig1, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            tof18 = taufromsig(siga= sig1, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            
            return np.array((0, tof12, tof13, tof14, tof15, tof16, tof17, tof18), dtype=np.float32)
        case 2:
            
            tof21 = taufromsig(siga= sig2, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            tof23 = taufromsig(siga= sig2, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            tof24 = taufromsig(siga= sig2, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            tof25 = taufromsig(siga= sig2, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            tof26 = taufromsig(siga= sig2, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            tof27 = taufromsig(siga= sig2, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            tof28 = taufromsig(siga= sig2, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            return np.array((tof21, 0, tof23, tof24, tof25, tof26, tof27, tof28), dtype=np.float32)
        case 3:
            
            tof31 = taufromsig(siga= sig3, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            tof32 = taufromsig(siga= sig3, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            tof34 = taufromsig(siga= sig3, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            tof35 = taufromsig(siga= sig3, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            tof36 = taufromsig(siga= sig3, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            tof37 = taufromsig(siga= sig3, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            tof38 = taufromsig(siga= sig3, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            return np.array((tof31, tof32, 0, tof34, tof35, tof36, tof37, tof38), dtype=np.float32) 
        case 4:
            
            tof41 = taufromsig(siga= sig4, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            tof42 = taufromsig(siga= sig4, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            tof43 = taufromsig(siga= sig4, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            tof45 = taufromsig(siga= sig4, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            tof46 = taufromsig(siga= sig4, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            tof47 = taufromsig(siga= sig4, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            tof48 = taufromsig(siga= sig4, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            return np.array((tof41, tof42, tof43, 0, tof45, tof46, tof47, tof48), dtype=np.float32)
        case 5:
            
            tof51 = taufromsig(siga= sig5, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            tof52 = taufromsig(siga= sig5, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            tof53 = taufromsig(siga= sig5, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            tof54 = taufromsig(siga= sig5, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            tof56 = taufromsig(siga= sig5, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            tof57 = taufromsig(siga= sig5, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            tof58 = taufromsig(siga= sig5, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            return np.array((tof51, tof52, tof53, tof54, 0, tof56, tof57, tof58), dtype=np.float32)
        case 6:
            
            tof61 = taufromsig(siga= sig6, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            tof62 = taufromsig(siga= sig6, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            tof63 = taufromsig(siga= sig6, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            tof64 = taufromsig(siga= sig6, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            tof65 = taufromsig(siga= sig6, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            tof67 = taufromsig(siga= sig6, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            tof68 = taufromsig(siga= sig6, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            return np.array((tof61, tof62, tof63, tof64, tof65, 0, tof67, tof68), dtype=np.float32)
        case 7:
            
            tof71 = taufromsig(siga= sig7, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            tof72 = taufromsig(siga= sig7, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            tof73 = taufromsig(siga= sig7, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            tof74 = taufromsig(siga= sig7, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            tof75 = taufromsig(siga= sig7, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            tof76 = taufromsig(siga= sig7, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            tof78 = taufromsig(siga= sig7, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            return np.array((tof71, tof72, tof73, tof74, tof75, tof76, 0, tof78), dtype=np.float32)
        case 8:
            tof81 = taufromsig(siga= sig8, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            tof82 = taufromsig(siga= sig8, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            tof83 = taufromsig(siga= sig8, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            tof84 = taufromsig(siga= sig8, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            tof85 = taufromsig(siga= sig8, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            tof86 = taufromsig(siga= sig8, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            tof87 = taufromsig(siga= sig8, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            return np.array((tof81, tof82, tof83, tof84, tof85, tof86, tof87, 0), dtype=np.float32)
        case _:
            raise ValueError




ketuk1 = loadjson("data/ketuk1.json")
ketuk2 = loadjson("data/ketuk2.json")
ketuk3 = loadjson("data/ketuk3.json")
ketuk4 = loadjson("data/ketuk4.json")
ketuk5 = loadjson("data/ketuk5.json")
ketuk6 = loadjson("data/ketuk6.json")
ketuk7 = loadjson("data/ketuk7.json")
ketuk8 = loadjson("data/ketuk8.json")
# ketuk9 = loadjson(".\\Exper2\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755249140_1.json")
# ketuk10 = loadjson(".\\Exper2\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755249279_2.json")
# ketuk11 = loadjson(".\\Exper2\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755249376_3.json")
# ketuk12 = loadjson(".\\Exper2\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755249466_4.json")
# ketuk13 = loadjson(".\\Exper2\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755249699_5.json")
# ketuk14 = loadjson(".\\Exper2\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755249790_6.json")
# ketuk15 = loadjson(".\\Exper2\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755249965_7.json")
# ketuk16 = loadjson(".\\Exper2\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755250052_8.json")
# ketuk17 = loadjson(".\\Exper3\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755250863_1.json")
# ketuk18 = loadjson(".\\Exper3\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755250960_2.json")
# ketuk19 = loadjson(".\\Exper3\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755251040_3.json")
# ketuk20 = loadjson(".\\Exper3\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755251127_4.json")
# ketuk21 = loadjson(".\\Exper3\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755251240_5.json")
# ketuk22 = loadjson(".\\Exper3\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755251349_6.json")
# ketuk23 = loadjson(".\\Exper3\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755251538_7.json")
# ketuk24 = loadjson(".\\Exper3\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755251659_8.json")
# ketuk25 = loadjson(".\\Exper4\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755252136_1.json")
# ketuk26 = loadjson(".\\Exper4\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755252219_2.json")
# ketuk27 = loadjson(".\\Exper4\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755252304_3.json")
# ketuk28 = loadjson(".\\Exper4\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755252386_4.json")
# ketuk29 = loadjson(".\\Exper4\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755252481_5.json")
# ketuk30 = loadjson(".\\Exper4\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755252592_6.json")
# ketuk31 = loadjson(".\\Exper4\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755252680_7.json")
# ketuk32 = loadjson(".\\Exper4\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755252762_8.json")
# ketuk33 = loadjson(".\\Exper5\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755253233_1.json")
# ketuk34 = loadjson(".\\Exper5\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755253316_2.json")
# ketuk35 = loadjson(".\\Exper5\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755253423_3.json")
# ketuk36 = loadjson(".\\Exper5\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755253539_4.json")
# ketuk37 = loadjson(".\\Exper5\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755253628_5.json")
# ketuk38 = loadjson(".\\Exper5\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755253974_6.json")
# ketuk39 = loadjson(".\\Exper5\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755254067_7.json")
# ketuk40 = loadjson(".\\Exper5\\08-3A-F2-8D-CA-F4_aa5ac152-3379-4f0b-8253-7591669e4e82_1755254391_8.json")


veloketuk1 = onetapest(ketuk1,1,0.3)
veloketuk2 = onetapest(ketuk2,2,0.3)
veloketuk3 = onetapest(ketuk3,3,0.3)
veloketuk4 = onetapest(ketuk4,4,0.3)
veloketuk5 = onetapest(ketuk5,5,0.3)
veloketuk6 = onetapest(ketuk6,6,0.3)
veloketuk7 = onetapest(ketuk7,7,0.3)
veloketuk8 = onetapest(ketuk8,8,0.3)
# veloketuk9 = onetapest(ketuk9,1,0.3)
# veloketuk10 = onetapest(ketuk10,2,0.3)
# veloketuk11 = onetapest(ketuk11,3,0.3)
# veloketuk12 = onetapest(ketuk12,4,0.3)
# veloketuk13 = onetapest(ketuk13,5,0.3)
# veloketuk14 = onetapest(ketuk14,6,0.3)
# veloketuk15 = onetapest(ketuk15,7,0.3)
# veloketuk16 = onetapest(ketuk16,8,0.3)
# veloketuk17 = onetapest(ketuk17,1,0.3)
# veloketuk18 = onetapest(ketuk18,2,0.3)
# veloketuk19 = onetapest(ketuk19,3,0.3)
# veloketuk20 = onetapest(ketuk20,4,0.3)
# veloketuk21 = onetapest(ketuk21,5,0.3)
# veloketuk22 = onetapest(ketuk22,6,0.3)
# veloketuk23 = onetapest(ketuk23,7,0.3)
# veloketuk24 = onetapest(ketuk24,8,0.3)
# veloketuk25 = onetapest(ketuk25,1,0.3)
# veloketuk26 = onetapest(ketuk26,2,0.3)
# veloketuk27 = onetapest(ketuk27,3,0.3)
# veloketuk28 = onetapest(ketuk28,4,0.3)
# veloketuk29 = onetapest(ketuk29,5,0.3)
# veloketuk30 = onetapest(ketuk30,6,0.3)
# veloketuk31 = onetapest(ketuk31,7,0.3)
# veloketuk32 = onetapest(ketuk32,8,0.3)
# veloketuk33 = onetapest(ketuk33,1,0.3)
# veloketuk34 = onetapest(ketuk34,2,0.3)
# veloketuk35 = onetapest(ketuk35,3,0.3)
# veloketuk36 = onetapest(ketuk36,4,0.3)
# veloketuk37 = onetapest(ketuk37,5,0.3)
# veloketuk38 = onetapest(ketuk38,6,0.3)
# veloketuk39 = onetapest(ketuk39,7,0.3)
# veloketuk40 = onetapest(ketuk40,8,0.3)


veloall1 = np.vstack((veloketuk1,veloketuk2,veloketuk3,veloketuk4,veloketuk5,veloketuk6,veloketuk7,veloketuk8), dtype=float)
beloall1 = veloall1.tolist()
file_path = ".//Exper1TOF.json"
json.dump(beloall1, codecs.open(file_path, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)



