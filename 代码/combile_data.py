import scipy.io as scio
import numpy as np
from sklearn.utils import shuffle
import math
from pyemd import EEMD
# 0--Right
# 1--Inner error
# 2--Ball error
# 3--Outer Centered
# 4--Outer Orthogonal
# 5--Outer Opposite

def insertFactor(File,number):
    Oristaticnumber = len(File[0])
    File_RPM = []
    for i in range(Oristaticnumber):
        File_RPM.append(number)
    File = np.vstack((File, File_RPM))
    return File

#计算波形熵
def MFE(oriFile):
    Oristaticnumber = len(oriFile)
    # RMS 均方根 ARV 整流平均值
    tempSUM = 0
    tempARV = 0
    for i in range(0, Oristaticnumber):
        tempSUM = tempSUM + np.power(oriFile[i], 2)
        tempARV = tempARV + abs(oriFile[i])
    RMS = np.sqrt(tempSUM / Oristaticnumber)
    ARV = np.sqrt(tempARV / Oristaticnumber)
    waveFactor = RMS / ARV
    WFE = waveFactor * math.log(waveFactor)
    return WFE

def input_97_file():
    Data_97='../Data/initial_Data/normal/97.mat'
    Data_97 = scio.loadmat(Data_97)
    OriInputN97_1 = Data_97['X097_DE_time']
    OriInputN97_2 = Data_97['X097_FE_time']
    OriA97_1 = np.array(OriInputN97_1)
    OriA97_2 = np.array(OriInputN97_2)
    OriA97_1 = OriA97_1.reshape(1, -1)
    OriA97_2 = OriA97_2.reshape(1, -1)
    """A97整合了A97_1和A97_2"""
    OriA97 = np.vstack((OriA97_1, OriA97_2))
    """加上转速参量"""
    #OriA97 = insertFactor(OriA97, 0.1797*0.5)
    """加上DE波形熵参量"""
    #MFE_Ori_1 = MFE(OriA97_1.transpose())
    #OriA97 = insertFactor(OriA97, MFE_Ori_1[0]*0.1)
    """加上FE波形熵参量"""
    #MFE_Ori_2 = MFE(OriA97_2.transpose())
    #OriA97 = insertFactor(OriA97, MFE_Ori_2[0]*0.1)



    OriA97 = OriA97.transpose()
    OriA97 = OriA97.astype(np.float64)
    return OriA97

def input_105_file():
    Data_105 = '../Data/initial_Data/12KDFData/7_0/0.007-0/105.mat'
    Data_105 = scio.loadmat(Data_105)
    OriInputN105_1 = Data_105['X105_DE_time']
    OriInputN105_2 = Data_105['X105_FE_time']
    OriInputN105_3 = Data_105['X105_BA_time']
    OriA105_1 = np.array(OriInputN105_1)
    OriA105_2 = np.array(OriInputN105_2)
    OriA105_3 = np.array(OriInputN105_3)
    OriA105_1 = OriA105_1.reshape(1, -1)
    OriA105_2 = OriA105_2.reshape(1, -1)
    OriA105_3 = OriA105_3.reshape(1, -1)
    """A105整合了A105_1,A105_2和A105_3"""
    #OriA105 = np.vstack((OriA105_1, OriA105_2,OriA105_3))


    temp1=OriA105_1[0][:7500]
    eemd = EEMD()
    eIMFs_1 = eemd(temp1,max_imf=10)
    #print("OriA105_eIMFs_1:", eIMFs_1.shape)
    temp2=OriA105_2[0][:7500]
    eemd = EEMD()
    eIMFs_2 = eemd(temp2,max_imf=10)
    #print("OriA105_eIMFs_2:", eIMFs_2.shape)
    temp3=OriA105_3[0][:7500]
    eemd = EEMD()
    eIMFs_3 = eemd(temp3,max_imf=10)
    #print("OriA105_eIMFs_3:", eIMFs_3.shape)
    OriA105 = np.vstack((eIMFs_1, eIMFs_2,eIMFs_3))



    """加上转速参量"""
    #OriA105 = insertFactor(OriA105, 0.1797)

    """加上DE波形熵参量"""
    MFE_Ori_1 = MFE(OriA105_1.transpose())
    OriA105 = insertFactor(OriA105, MFE_Ori_1[0])
    """加上FE波形熵参量"""
    MFE_Ori_2 = MFE(OriA105_2.transpose())
    OriA105 = insertFactor(OriA105, MFE_Ori_2[0])
    """加上BA波形熵参量"""
    MFE_Ori_3 = MFE(OriA105_3.transpose())
    OriA105 = insertFactor(OriA105, MFE_Ori_3[0])

    """加上错误标签"""
    OriA105 = insertFactor(OriA105, np.int(1))
    OriA105 = OriA105.transpose()
    OriA105 = OriA105.astype(np.float64)
    print("OriA105:", OriA105.shape)
    print("OriA105[0]:", OriA105[0])
    return OriA105

def input_118_file():
    Data_118 = '../Data/initial_Data/12KDFData/7_0/0.007-0/118.mat'
    Data_118 = scio.loadmat(Data_118)
    OriInputN118_1 = Data_118['X118_DE_time']
    OriInputN118_2 = Data_118['X118_FE_time']
    OriInputN118_3 = Data_118['X118_BA_time']
    OriA118_1 = np.array(OriInputN118_1)
    OriA118_2 = np.array(OriInputN118_2)
    OriA118_3 = np.array(OriInputN118_3)
    OriA118_1 = OriA118_1.reshape(1, -1)
    OriA118_2 = OriA118_2.reshape(1, -1)
    OriA118_3 = OriA118_3.reshape(1, -1)
    """A105整合了A105_1,A105_2和A105_3"""
    #OriA118 = np.vstack((OriA118_1, OriA118_2,OriA118_3))

    temp1 = OriA118_1[0][:7500]
    eemd = EEMD()
    eIMFs_1 = eemd(temp1,max_imf=10)

    temp2 = OriA118_2[0][:7500]
    eemd = EEMD()
    eIMFs_2 = eemd(temp2,max_imf=10)

    temp3 = OriA118_3[0][:7500]
    eemd = EEMD()
    eIMFs_3 = eemd(temp3,max_imf=10)

    OriA118 = np.vstack((eIMFs_1, eIMFs_2, eIMFs_3))


    """加上转速参量"""
    #OriA118 = insertFactor(OriA118, 0.1797)

    """加上DE波形熵参量"""
    MFE_Ori_1 = MFE(OriA118_1.transpose())
    OriA118 = insertFactor(OriA118, MFE_Ori_1[0])
    """加上FE波形熵参量"""
    MFE_Ori_2 = MFE(OriA118_2.transpose())
    OriA118 = insertFactor(OriA118, MFE_Ori_2[0])
    """加上BA波形熵参量"""
    MFE_Ori_3 = MFE(OriA118_3.transpose())
    OriA118 = insertFactor(OriA118, MFE_Ori_3[0])

    """加上错误标签"""
    OriA118 = insertFactor(OriA118, np.int(2))
    OriA118 = OriA118.transpose()
    OriA118 = OriA118.astype(np.float64)
    print("OriA118:", OriA118.shape)
    print("OriA118[0]:", OriA118[0])
    return OriA118

def input_130_file():
    Data_130 = '../Data/initial_Data/12KDFData/7_0/0.007-0/130.mat'
    Data_130 = scio.loadmat(Data_130)
    OriInputN130_1 = Data_130['X130_DE_time']
    OriInputN130_2 = Data_130['X130_FE_time']
    OriInputN130_3 = Data_130['X130_BA_time']
    OriA130_1 = np.array(OriInputN130_1)
    OriA130_2 = np.array(OriInputN130_2)
    OriA130_3 = np.array(OriInputN130_3)
    OriA130_1 = OriA130_1.reshape(1, -1)
    OriA130_2 = OriA130_2.reshape(1, -1)
    OriA130_3 = OriA130_3.reshape(1, -1)
    """A105整合了A105_1,A105_2和A105_3"""
    #OriA130 = np.vstack((OriA130_1, OriA130_2,OriA130_3))

    temp1 = OriA130_1[0][:7500]
    eemd = EEMD()
    eIMFs_1 = eemd(temp1,max_imf=10)

    temp2 = OriA130_2[0][:7500]
    eemd = EEMD()
    eIMFs_2 = eemd(temp2,max_imf=10)

    temp3 = OriA130_3[0][:7500]
    eemd = EEMD()
    eIMFs_3 = eemd(temp3,max_imf=10)

    OriA130 = np.vstack((eIMFs_1, eIMFs_2, eIMFs_3))

    """加上转速参量"""
    #OriA130 = insertFactor(OriA130, 0.1797)

    """加上DE波形熵参量"""
    MFE_Ori_1 = MFE(OriA130_1.transpose())
    OriA130 = insertFactor(OriA130, MFE_Ori_1[0])
    """加上FE波形熵参量"""
    MFE_Ori_2 = MFE(OriA130_2.transpose())
    OriA130 = insertFactor(OriA130, MFE_Ori_2[0])
    """加上BA波形熵参量"""
    MFE_Ori_3 = MFE(OriA130_3.transpose())
    OriA130 = insertFactor(OriA130, MFE_Ori_3[0])

    """加上错误标签"""
    OriA130 = insertFactor(OriA130, np.int(3))
    OriA130 = OriA130.transpose()
    OriA130 = OriA130.astype(np.float64)
    print("OriA130:", OriA130.shape)
    print("OriA130[0]:", OriA130[0])
    return OriA130

def input_144_file():
    Data_144 = '../Data/initial_Data/12KDFData/7_0/0.007-0/144.mat'
    Data_144 = scio.loadmat(Data_144)
    OriInputN144_1 = Data_144['X144_DE_time']
    OriInputN144_2 = Data_144['X144_FE_time']
    OriInputN144_3 = Data_144['X144_BA_time']
    OriA144_1 = np.array(OriInputN144_1)
    OriA144_2 = np.array(OriInputN144_2)
    OriA144_3 = np.array(OriInputN144_3)
    OriA144_1 = OriA144_1.reshape(1, -1)
    OriA144_2 = OriA144_2.reshape(1, -1)
    OriA144_3 = OriA144_3.reshape(1, -1)
    """A105整合了A105_1,A105_2和A105_3"""
    #OriA144 = np.vstack((OriA144_1, OriA144_2,OriA144_3))

    temp1 = OriA144_1[0][:7500]
    eemd = EEMD()
    eIMFs_1 = eemd(temp1,max_imf=10)

    temp2 = OriA144_2[0][:7500]
    eemd = EEMD()
    eIMFs_2 = eemd(temp2,max_imf=10)

    temp3 = OriA144_3[0][:7500]
    eemd = EEMD()
    eIMFs_3 = eemd(temp3,max_imf=10)

    OriA144 = np.vstack((eIMFs_1, eIMFs_2, eIMFs_3))


    """加上转速参量"""
    #OriA144 = insertFactor(OriA144, 0.1797)

    """加上DE波形熵参量"""
    MFE_Ori_1 = MFE(OriA144_1.transpose())
    OriA144 = insertFactor(OriA144, MFE_Ori_1[0])
    """加上FE波形熵参量"""
    MFE_Ori_2 = MFE(OriA144_2.transpose())
    OriA144 = insertFactor(OriA144, MFE_Ori_2[0])
    """加上BA波形熵参量"""
    MFE_Ori_3 = MFE(OriA144_3.transpose())
    OriA144 = insertFactor(OriA144, MFE_Ori_3[0])

    """加上错误标签"""
    OriA144 = insertFactor(OriA144, np.int(4))
    OriA144 = OriA144.transpose()
    OriA144 = OriA144.astype(np.float64)
    print("OriA144:", OriA144.shape)
    print("OriA144[0]:", OriA144[0])
    return OriA144

def input_156_file():
    Data_156 = '../Data/initial_Data/12KDFData/7_0/0.007-0/156.mat'
    Data_156 = scio.loadmat(Data_156)
    OriInputN156_1 = Data_156['X156_DE_time']
    OriInputN156_2 = Data_156['X156_FE_time']
    OriInputN156_3 = Data_156['X156_BA_time']
    OriA156_1 = np.array(OriInputN156_1)
    OriA156_2 = np.array(OriInputN156_2)
    OriA156_3 = np.array(OriInputN156_3)
    OriA156_1 = OriA156_1.reshape(1, -1)
    OriA156_2 = OriA156_2.reshape(1, -1)
    OriA156_3 = OriA156_3.reshape(1, -1)
    """A105整合了A105_1,A105_2和A105_3"""
    #OriA156 = np.vstack((OriA156_1, OriA156_2,OriA156_3))

    temp1 = OriA156_1[0][:7500]
    eemd = EEMD()
    eIMFs_1 = eemd(temp1,max_imf=10)

    temp2 = OriA156_2[0][:7500]
    eemd = EEMD()
    eIMFs_2 = eemd(temp2,max_imf=10)

    temp3 = OriA156_3[0][:7500]
    eemd = EEMD()
    eIMFs_3 = eemd(temp3,max_imf=10)

    OriA156 = np.vstack((eIMFs_1, eIMFs_2, eIMFs_3))

    """加上转速参量"""
    #OriA156 = insertFactor(OriA156, 0.1797)

    """加上DE波形熵参量"""
    MFE_Ori_1 = MFE(OriA156_1.transpose())
    OriA156 = insertFactor(OriA156, MFE_Ori_1[0])
    """加上FE波形熵参量"""
    MFE_Ori_2 = MFE(OriA156_2.transpose())
    OriA156 = insertFactor(OriA156, MFE_Ori_2[0])
    """加上BA波形熵参量"""
    MFE_Ori_3 = MFE(OriA156_3.transpose())
    OriA156 = insertFactor(OriA156, MFE_Ori_3[0])

    """加上错误标签"""
    OriA156 = insertFactor(OriA156, np.int(5))
    OriA156 = OriA156.transpose()
    OriA156 = OriA156.astype(np.float64)
    print("OriA156:", OriA156.shape)
    print("OriA156[0]:", OriA156[0])
    return OriA156

def combile_5_file(file_105,file_118,file_130,file_144,file_156,all_number):
    each_number=all_number/5
# 处理105文件
    rand_arr = np.arange(file_105.shape[0])
    np.random.shuffle(rand_arr)
    simple_105=file_105[rand_arr[0:int(each_number)]]
# 处理118文件
    rand_arr = np.arange(file_118.shape[0])
    np.random.shuffle(rand_arr)
    simple_118 = file_118[rand_arr[0:int(each_number)]]
# 处理130文件
    rand_arr = np.arange(file_130.shape[0])
    np.random.shuffle(rand_arr)
    simple_130 = file_130[rand_arr[0:int(each_number)]]
# 处理144文件
    rand_arr = np.arange(file_144.shape[0])
    np.random.shuffle(rand_arr)
    simple_144 = file_144[rand_arr[0:int(each_number)]]
# 处理156文件
    rand_arr = np.arange(file_156.shape[0])
    np.random.shuffle(rand_arr)
    simple_156 = file_156[rand_arr[0:int(each_number)]]
# combile the file
    simple_all=np.vstack((simple_105, simple_118,simple_130,simple_144,simple_156))
    simple_all = shuffle(simple_all)
    return simple_all

def combile_3_file(file_105,file_118,file_130,all_number):
    each_number=all_number/3
# 处理105文件
    rand_arr = np.arange(file_105.shape[0])
    np.random.shuffle(rand_arr)
    simple_105=file_105[rand_arr[0:int(each_number)]]
# 处理118文件
    rand_arr = np.arange(file_118.shape[0])
    np.random.shuffle(rand_arr)
    simple_118 = file_118[rand_arr[0:int(each_number)]]
# 处理130文件
    rand_arr = np.arange(file_130.shape[0])
    np.random.shuffle(rand_arr)
    simple_130 = file_130[rand_arr[0:int(each_number)]]

# combile the file
    simple_all=np.vstack((simple_105, simple_118,simple_130))
    simple_all = shuffle(simple_all)
    return simple_all



if __name__ == '__main__':
    file_97=input_97_file()
    file_105=input_105_file()
    file_118=input_118_file()
    file_130=input_130_file()
    file_144=input_144_file()
    file_156=input_156_file()

    combile_5_fault_10000_number=combile_5_file(file_105,file_118,file_130,file_144,file_156,1000)
    #combile_3_fault_10000_number=combile_3_file(file_105,file_118,file_130,10000)
    print("5_Data.shape:",combile_5_fault_10000_number.shape)
    print("5_Data:", combile_5_fault_10000_number)
    #print("3_Data.shape:",combile_3_fault_10000_number.shape)
    #print("3_Data:", combile_3_fault_10000_number)
    #scio.savemat('../Data/combileData/have_rate/5_error/combile_5_fault_10000_number.mat', {'output': combile_5_fault_10000_number})
    #scio.savemat('../Data/combileData/have_rate/3_error/combile_3_fault_10000_number.mat', {'output': combile_3_fault_10000_number})

    #scio.savemat('../Data/combileData/non_rate/5_error/combile_5_fault_1000_number.mat',{'output': combile_5_fault_10000_number})

    scio.savemat('../Data/combileData/non_rate/5_error/combile_5_fault_1000_number_EEMD_DE-FE-BA-MFE.mat', {'output': combile_5_fault_10000_number})
    #scio.savemat('../Data/combileData/non_rate/3_error/combile_3_fault_10000_number_EEMD.mat', {'output': combile_3_fault_10000_number})





