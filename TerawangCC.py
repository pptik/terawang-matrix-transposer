import numpy as np
import pandas as pd
import scipy
import json
import codecs
# from gccestimating import GCC, corrlags

def loadcsv(filename, delim=","):
    # Reads CSV from file
    data = np.loadtxt(filename, delimiter=delim, dtype= np.float64)
    return data

def loadjson(filename):
    # Reads JSON from file
    data = json.load(open(filename))
    # Returns as dict
    return data

def gccnormal(sig, refsig, fs=1000000, interp=128, max_tau=None, CCType="PHAT", timestamp=None):
    
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCCnormal-PHAT)method.
    '''

    # Generalized Cross Correlation Phase Transform
    n = len(sig)
    
    # Remove DC component
    
    
    sig -= np.mean(sig, axis=0)
    refsig -= np.mean(refsig, axis=0)

    # Generalized Cross Correlation Phase Transform
    # RFFT because it's faster, it doesn't compute the negative side
    SIG = np.fft.rfft(sig, axis=0, n=n)
    REFSIG = np.fft.rfft(refsig, axis=0, n=n)
    
    CONJ = np.conj(SIG)
    
    R = np.multiply(REFSIG,CONJ)
    
    match CCType:
        case "CC" | "cc":
            WEIGHT = 1
        case "PHAT" | "Phat" | "phat":
            CCType = "PHAT"
            WEIGHT = 1/np.abs(R)
        case "SCOT" | "Scot" | "scot":
            CCType = "SCOT"
            WEIGHT = 1/np.sqrt(SIG*np.conj(SIG)*REFSIG*CONJ)
        case "ROTH" | "Roth" | "roth":
            CCType = "ROTH"
            WEIGHT = 1/(SIG*np.conj(SIG))
        case _:
            CCType = "CC"
            WEIGHT = 1
    
    Integ = np.multiply(R,WEIGHT)
    
    cc = np.fft.irfft(a=Integ, axis=0, n=n)
    lags = scipy.signal.correlation_lags(len(refsig), len(sig), mode= 'same')

    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)
        
    
    smallcc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    smallcc /= np.max(cc)

    # find max cross correlation index
    shift = np.argmax(smallcc) - max_shift

    # Sometimes,    there is a 180-degree phase difference between the two microphones.
    # shift = np.argmax(np.abs(cc)) - max_shift
    
    
    cc = scipy.ndimage.shift(cc, len(cc)/2, mode="grid-wrap", order = 5)
    cc /= np.max(cc)
    
    tau = shift / float(interp * fs)
    
    if timestamp is not None:
        
        peaktimestamp = timestamp[np.argmax(cc)] # peak of CC within timetamp
        
        timestamp = scipy.ndimage.shift(timestamp, len(timestamp)/2, mode="grid-wrap", order = 5) # shift the timestamp to 'wrap around'
        
        a = timestamp[0] # first possible timestamp on the dataframe
        b = timestamp[max_shift] # timestamp that corresponds fo the end of smalltimestamp 
        c = timestamp[-max_shift-1] # timestamp that corresponds to the start of the smalltimestamp
        d = timestamp[-1] # last possible timestamp on the dataframe
        # smalltimestamp = np.concatenate((timestamp[-max_shift:], timestamp[:max_shift+1]))
        # peaktimestamp = smalltimestamp[np.argmax(smallcc)]
        
        print(peaktimestamp)
        
        if a > peaktimestamp >=  b:
            tau = int(peaktimestamp - a) # in micros
        else:
            tau = int(-peaktimestamp + c) # in micros, negative
        tau /= 1000000 # convert to seconds
    
    return np.abs(tau), cc, lags

def gccest(sig1, sig2, samplerate=1 ,cctype="phat"):
    n = len(sig1)
    
    sig1 -= np.mean(sig1, axis=0)
    sig2 -= np.mean(sig2, axis=0)
    
    lags = corrlags(2*n-1, samplerate=samplerate)
    
    gcc = GCC(sig1=sig1,sig2=sig2)
    
    match cctype.lower():
        case "cc":
            cc = gcc.cc()
        case "phat":
            cc = gcc.phat()
        case "scot":
            cc = gcc.scot()
        case"roth":
            cc = gcc.roth()
        case _:
            cc = gcc.cc()
    
    cc /= np.max(np.abs(cc))    # normalize
    
    return cc, lags


def tauest(cc, lags, samplerate = 1, timestamp = None):
    n = len(cc)
        
        

def onetap(sigdict: list, which: int, diameter = 0.3):
    
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
    soundspeed = 4150 # Average speed of sound in wood
        
    radius = diameter/2
    ab = radius * 0.76536686473 # sqrt(sqrt(2)-2)
    ac = radius * 1.41421356237 # sqrt(2)
    ad = radius * 1.84775906502 # sqrt(sqrt(2)+2)
    ae = float(diameter)
    
    max_tau = 0.3 / soundspeed
    
    # ab = 12,23,34,45,56,67,78,81
    # ac = 13,24,35,46,57,68,71,82
    # ad = 14,25,36,47,58,61,72,83
    # ae = 15,26,37,48,51,62,73,84
    
    match which:
        case 1:
            tof12 = gccnormal(refsig=sig1, sig=sig2, timestamp=None, max_tau=max_tau)[0]
            tof13 = gccnormal(refsig=sig1, sig=sig3, timestamp=None, max_tau=max_tau)[0]
            tof14 = gccnormal(refsig=sig1, sig=sig4, timestamp=None, max_tau=max_tau)[0]
            tof15 = gccnormal(refsig=sig1, sig=sig5, timestamp=None, max_tau=max_tau)[0]
            tof16 = gccnormal(refsig=sig1, sig=sig6, timestamp=None, max_tau=max_tau)[0]
            tof17 = gccnormal(refsig=sig1, sig=sig7, timestamp=None, max_tau=max_tau)[0]
            tof18 = gccnormal(refsig=sig1, sig=sig8, timestamp=None, max_tau=max_tau)[0]
            velo12 = ab / tof12
            velo13 = ac / tof13
            velo14 = ad / tof14
            velo15 = ae / tof15
            velo16 = ad / tof16
            velo17 = ac / tof17
            velo18 = ab / tof18
            
            return np.array((0, velo12, velo13, velo14, velo15, velo16, velo17, velo18), dtype=np.float32)
        case 2:
            tof21 = gccnormal(refsig=sig2, sig=sig1, timestamp=None, max_tau=max_tau)[0]
            tof23 = gccnormal(refsig=sig2, sig=sig3, timestamp=None, max_tau=max_tau)[0]
            tof24 = gccnormal(refsig=sig2, sig=sig4, timestamp=None, max_tau=max_tau)[0]
            tof25 = gccnormal(refsig=sig2, sig=sig5, timestamp=None, max_tau=max_tau)[0]
            tof26 = gccnormal(refsig=sig2, sig=sig6, timestamp=None, max_tau=max_tau)[0]
            tof27 = gccnormal(refsig=sig2, sig=sig7, timestamp=None, max_tau=max_tau)[0]
            tof28 = gccnormal(refsig=sig2, sig=sig8, timestamp=None, max_tau=max_tau)[0]
            velo21 = ab / tof21
            velo23 = ab / tof23
            velo24 = ac / tof24
            velo25 = ad / tof25
            velo26 = ae / tof26
            velo27 = ad / tof27
            velo28 = ac / tof28
            
            return np.array((velo21, 0, velo23, velo24, velo25, velo26, velo27, velo28), dtype=np.float32)
        case 3:
            tof31 = gccnormal(refsig=sig3, sig=sig1, timestamp=None, max_tau=max_tau)[0]
            tof32 = gccnormal(refsig=sig3, sig=sig2, timestamp=None, max_tau=max_tau)[0]
            tof34 = gccnormal(refsig=sig3, sig=sig4, timestamp=None, max_tau=max_tau)[0]
            tof35 = gccnormal(refsig=sig3, sig=sig5, timestamp=None, max_tau=max_tau)[0]
            tof36 = gccnormal(refsig=sig3, sig=sig6, timestamp=None, max_tau=max_tau)[0]
            tof37 = gccnormal(refsig=sig3, sig=sig7, timestamp=None, max_tau=max_tau)[0]
            tof38 = gccnormal(refsig=sig3, sig=sig8, timestamp=None, max_tau=max_tau)[0]
            velo31 = ac / tof31
            velo32 = ab / tof32
            velo34 = ab / tof34
            velo35 = ac / tof35
            velo36 = ad / tof36
            velo37 = ae / tof37
            velo38 = ad / tof38
            
            return np.array((velo31, velo32, 0, velo34, velo35, velo36, velo37, velo38), dtype=np.float32) 
        case 4:
            tof41 = gccnormal(refsig=sig4, sig=sig1, timestamp=None, max_tau=max_tau)[0]
            tof42 = gccnormal(refsig=sig4, sig=sig2, timestamp=None, max_tau=max_tau)[0]
            tof43 = gccnormal(refsig=sig4, sig=sig3, timestamp=None, max_tau=max_tau)[0]
            tof45 = gccnormal(refsig=sig4, sig=sig5, timestamp=None, max_tau=max_tau)[0]
            tof46 = gccnormal(refsig=sig4, sig=sig6, timestamp=None, max_tau=max_tau)[0]
            tof47 = gccnormal(refsig=sig4, sig=sig7, timestamp=None, max_tau=max_tau)[0]
            tof48 = gccnormal(refsig=sig4, sig=sig8, timestamp=None, max_tau=max_tau)[0]
            velo41 = ad / tof41
            velo42 = ac / tof42
            velo43 = ab / tof43
            velo45 = ab / tof45
            velo46 = ac / tof46
            velo47 = ad / tof47
            velo48 = ae / tof48
            
            return np.array((velo41, velo42, velo43, 0, velo45, velo46, velo47, velo48), dtype=np.float32)
        case 5:
            tof51 = gccnormal(refsig=sig5, sig=sig1, timestamp=None, max_tau=max_tau)[0]
            tof52 = gccnormal(refsig=sig5, sig=sig2, timestamp=None, max_tau=max_tau)[0]
            tof53 = gccnormal(refsig=sig5, sig=sig3, timestamp=None, max_tau=max_tau)[0]
            tof54 = gccnormal(refsig=sig5, sig=sig4, timestamp=None, max_tau=max_tau)[0]
            tof56 = gccnormal(refsig=sig5, sig=sig6, timestamp=None, max_tau=max_tau)[0]
            tof57 = gccnormal(refsig=sig5, sig=sig7, timestamp=None, max_tau=max_tau)[0]
            tof58 = gccnormal(refsig=sig5, sig=sig8, timestamp=None, max_tau=max_tau)[0]
            velo51 = ae / tof51
            velo52 = ad / tof52
            velo53 = ac / tof53
            velo54 = ab / tof54
            velo56 = ab / tof56
            velo57 = ac / tof57
            velo58 = ad / tof58
            
            return np.array((velo51, velo52, velo53, velo54, 0, velo56, velo57, velo58), dtype=np.float32)
        case 6:
            tof61 = gccnormal(refsig=sig6, sig=sig1, timestamp=None, max_tau=max_tau)[0]
            tof62 = gccnormal(refsig=sig6, sig=sig2, timestamp=None, max_tau=max_tau)[0]
            tof63 = gccnormal(refsig=sig6, sig=sig3, timestamp=None, max_tau=max_tau)[0]
            tof64 = gccnormal(refsig=sig6, sig=sig4, timestamp=None, max_tau=max_tau)[0]
            tof65 = gccnormal(refsig=sig6, sig=sig5, timestamp=None, max_tau=max_tau)[0]
            tof67 = gccnormal(refsig=sig6, sig=sig7, timestamp=None, max_tau=max_tau)[0]
            tof68 = gccnormal(refsig=sig6, sig=sig8, timestamp=None, max_tau=max_tau)[0]
            velo61 = ad / tof61
            velo62 = ae / tof62
            velo63 = ad / tof63
            velo64 = ac / tof64
            velo65 = ab / tof65
            velo67 = ab / tof67
            velo68 = ac / tof68
            
            return np.array((velo61, velo62, velo63, velo64, velo65, 0, velo67, velo68), dtype=np.float32)
        case 7:
            tof71 = gccnormal(refsig=sig7, sig=sig1, timestamp=None, max_tau=max_tau)[0]
            tof72 = gccnormal(refsig=sig7, sig=sig2, timestamp=None, max_tau=max_tau)[0]
            tof73 = gccnormal(refsig=sig7, sig=sig3, timestamp=None, max_tau=max_tau)[0]
            tof74 = gccnormal(refsig=sig7, sig=sig4, timestamp=None, max_tau=max_tau)[0]
            tof75 = gccnormal(refsig=sig7, sig=sig5, timestamp=None, max_tau=max_tau)[0]
            tof76 = gccnormal(refsig=sig7, sig=sig6, timestamp=None, max_tau=max_tau)[0]
            tof78 = gccnormal(refsig=sig7, sig=sig8, timestamp=None, max_tau=max_tau)[0]
            velo71 = ac / tof71
            velo72 = ad / tof72
            velo73 = ae / tof73
            velo74 = ad / tof74
            velo75 = ac / tof75
            velo76 = ab / tof76
            velo78 = ab / tof78
            
            return np.array((velo71, velo72, velo73, velo74, velo75, velo76, 0, velo78), dtype=np.float32)
        case 8:
            tof81 = gccnormal(refsig=sig8, sig=sig1, timestamp=None, max_tau=max_tau)[0]
            tof82 = gccnormal(refsig=sig8, sig=sig2, timestamp=None, max_tau=max_tau)[0]
            tof83 = gccnormal(refsig=sig8, sig=sig3, timestamp=None, max_tau=max_tau)[0]
            tof84 = gccnormal(refsig=sig8, sig=sig4, timestamp=None, max_tau=max_tau)[0]
            tof85 = gccnormal(refsig=sig8, sig=sig5, timestamp=None, max_tau=max_tau)[0]
            tof86 = gccnormal(refsig=sig8, sig=sig6, timestamp=None, max_tau=max_tau)[0]
            tof87 = gccnormal(refsig=sig8, sig=sig7, timestamp=None, max_tau=max_tau)[0]
            velo81 = ab / tof81
            velo82 = ac / tof82
            velo83 = ad / tof83
            velo84 = ae / tof84
            velo85 = ad / tof85
            velo86 = ac / tof86
            velo87 = ab / tof87
            
            return np.array((velo81, velo82, velo83, velo84, velo85, velo86, velo87, 0), dtype=np.float32)
        case _:
            raise ValueError

def onebyeight(sensarray,which,diameter):
    
    # function to make the ToF of 1x8 matrix of a particular tap
    
    # sensarray = np.concatenate((sensarray,whichsensoristapped), axis=None)
    
    return onetap(sensarray,which=which,diameter=diameter)

# ketuk1 = loadjson(".\\1754553587_1.json")
# ketuk2 = loadjson(".\\1754553619_2.json")
# ketuk3 = loadjson(".\\1754553659_3.json")
# ketuk4 = loadjson(".\\1754553718_4.json")
# ketuk5 = loadjson(".\\1754553836_5.json")
# ketuk6 = loadjson(".\\1754553905_6.json")
# ketuk7 = loadjson(".\\1754553953_7.json")
# ketuk8 = loadjson(".\\1754553987_8.json")

ketuk1 = loadjson("test/ketuk1.json")
ketuk2 = loadjson("test/ketuk2.json")
ketuk3 = loadjson("test/ketuk3.json")
ketuk4 = loadjson("test/ketuk4.json")
ketuk5 = loadjson("test/ketuk5.json")
ketuk6 = loadjson("test/ketuk6.json")
ketuk7 = loadjson("test/ketuk7.json")
ketuk8 = loadjson("test/ketuk8.json")

veloketuk1 = onetap(ketuk1,1,0.3)
veloketuk2 = onetap(ketuk2,2,0.3)
veloketuk3 = onetap(ketuk3,3,0.3)
veloketuk4 = onetap(ketuk4,4,0.3)
veloketuk5 = onetap(ketuk5,5,0.3)
veloketuk6 = onetap(ketuk6,6,0.3)
veloketuk7 = onetap(ketuk7,7,0.3)
veloketuk8 = onetap(ketuk8,8,0.3)

veloall = np.vstack((veloketuk1,veloketuk2,veloketuk3,veloketuk4,veloketuk5,veloketuk6,veloketuk7,veloketuk8), dtype=float)

beloall = veloall.tolist()
file_path = ".//contoh1.json"
json.dump(beloall, codecs.open(file_path, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4) ### this saves the array in .json format


print(veloall)

