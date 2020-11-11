# coding=utf8
import os
import re
import wave
import pandas as pd

import pyaudio
from pydub import AudioSegment


import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
 
mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号

from sklearn.metrics.pairwise import cosine_similarity
import pdb

class voice():
    def loaddata(self, filepath):
        '''
        :param filepath: 文件路径，为wav文件
        :return: 如果无异常则返回True，如果有异常退出并返回False
        self.wave_data内储存着多通道的音频数据，其中self.wave_data[0]代表第一通道
        具体有几通道，看self.nchannels
        '''
        if type(filepath) != str:
            raise (TypeError, 'the type of filepath must be string')
        p1 = re.compile('\.wav')
        if p1.findall(filepath) is None:
            raise (IOError, 'the suffix of file must be .wav')
        try:
            f = wave.open(filepath, 'rb')
            params = f.getparams()
            #print (params,type(params))
            self.nchannels, self.sampwidth, self.framerate, self.nframes = params[:4]
            #print (self.nchannels,self.sampwidth, self.framerate,self.nframes)
            str_data = f.readframes(self.nframes)
            self.wave_data = np.fromstring(str_data, dtype=np.short)
            #print (self.wave_data,self.wave_data.shape)
            self.wave_data.shape = -1, self.sampwidth
            #print (self.wave_data.shape)
            self.wave_data = self.wave_data.T
            #print (self.wave_data)
            f.close()
            self.name = os.path.basename(filepath)  # 记录下文件名
            return True
        except:
            raise (IOError, 'File Error')
 
    def fft(self, frames=40):
        '''
        整体指纹提取的核心方法，将整个音频分块后分别对每块进行傅里叶变换，之后分子带抽取高能量点的下标
        :param frames: frames是指定每秒钟分块数
        :return:
        '''
        block = []
        fft_blocks = []
        self.high_point = []
        blocks_size = self.framerate // frames  # block_size为每一块的frame数量
        #print (blocks_size)
        blocks_num = self.nframes // blocks_size  # 将音频分块的数量
        #print (blocks_num)
        for i in range(0, len(self.wave_data[0]) - blocks_size, blocks_size):
            block.append(self.wave_data[0][i:i + blocks_size])
            #print (np.fft.fft(self.wave_data[0][i:i + blocks_size]))
            fft_blocks.append(np.abs(np.fft.fft(self.wave_data[0][i:i + blocks_size])))
            #print (fft_blocks,len(fft_blocks[0]))
            self.high_point.append((np.argmax(fft_blocks[-1][:40]),
                                    np.argmax(fft_blocks[-1][40:80]) + 40,
                                    np.argmax(fft_blocks[-1][80:120]) + 80,
                                    np.argmax(fft_blocks[-1][120:180]) + 120,
                                    np.argmax(fft_blocks[-1][180:300]) + 180,
                                    np.argmax(fft_blocks[-1][300:600]) + 300,
                                    ))
            break
        #print (len(self.high_point),self.high_point)
        return self.high_point
 
    def play(self, filepath):
        '''
        音频播放方法
        :param filepath:文件路径
        :return:
        '''
        chunk = 1024
        wf = wave.open(filepath, 'rb')
        p = pyaudio.PyAudio()
        # 打开声音输出流
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        print ("stream done")
        # 写声音输出流进行播放
        while True:
            data = wf.readframes(chunk)
            if data == "": break
            stream.write(data)
        stream.close()
        print ("while done")
        p.terminate()
 


  
  
    def trans_mp3_to_wav(self,filepath='./zhiduanqingchang.mp3',wpath='./zdqc.wav'):
     song = AudioSegment.from_mp3(filepath)
     song.export(wpath, format="wav")


#模拟混合信号，fft解析混合信号。https://blog.csdn.net/qq_27825451/article/details/88553441
def demo1():
    #采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
    x=np.linspace(0,1,1400)      
     
    #设置需要采样的信号，频率分量有200，400和600
    #y=7*np.sin(2*np.pi*200*x) + 5*np.sin(2*np.pi*400*x)+3*np.sin(2*np.pi*600*x)

    y=7*np.sin(2*np.pi*600*x) + 5*np.sin(2*np.pi*100*x)+3*np.sin(2*np.pi*300*x)+2*np.sin(2*np.pi*400*x)
     
    fft_y=fft(y)                          #快速傅里叶变换
     
    N=1400
    x = np.arange(N)             # 频率个数
    half_x = x[range(int(N/2))]  #取一半区间
     
    abs_y=np.abs(fft_y)                # 取复数的绝对值，即复数的模(双边频谱)
    angle_y=np.angle(fft_y)            #取复数的角度
    normalization_y=abs_y/N            #归一化处理（双边频谱）
    print (normalization_y)

    normalization_half_y = normalization_y[range(int(N/2))]      #由于对称性，只取一半区间（单边频谱）
    print (normalization_half_y)
     
    plt.subplot(231)
    plt.plot(x,y)   
    plt.title('原始波形')
     
    plt.subplot(232)
    plt.plot(x,fft_y,'black')
    plt.title('双边振幅谱(未求振幅绝对值)',fontsize=9,color='black') 
     
    plt.subplot(233)
    plt.plot(x,abs_y,'r')
    plt.title('双边振幅谱(未归一化)',fontsize=9,color='red') 
     
    plt.subplot(234)
    plt.plot(x,angle_y,'violet')
    plt.title('双边相位谱(未归一化)',fontsize=9,color='violet')
     
    plt.subplot(235)
    plt.plot(x,normalization_y,'g')
    plt.title('双边振幅谱(归一化)',fontsize=9,color='green')
     
    plt.subplot(236)
    plt.plot(half_x,normalization_half_y,'blue')
    plt.title('单边振幅谱(归一化)',fontsize=9,color='blue')
     
    plt.show()



'''
demo1
1、采样个数小于频率两倍，会丢失数据
2、采样点个数是横坐标，也表示频率。f=1/T频率=周期的倒数。角速度w=2pi/T=2pi*f。
3、y坐标表示对应频率的振幅。



'''







def generate_feature():
    p = voice()
    feature = {}
    data = []
    audioList = os.listdir('./sample')
    #print (audioList)

    for tmp in audioList:
        audioName = os.path.join('./sample', tmp)
        #print (audioName)
        if audioName.endswith('.wav'):
            p.loaddata(audioName)
            temp = p.fft()
            #print (temp[0],type(temp[0]))
            feature[tmp] = temp[0]
    #print (feature)
    keys = list(feature.keys())
    print (keys)
    for i in range(len(keys)-1):
        for j in range(i+1,len(keys)):
            print (keys[i],keys[j],np.dot(feature[keys[i]],feature[keys[j]]))
            data.append([keys[i],keys[j],cosine_similarity([feature[keys[i]],feature[keys[j]]])[0][1]])
    pd_data = pd.DataFrame(data,columns=["itemA","itemB","sim"])
    print (pd_data)
    pd_data.to_csv('./sim.csv',index=False)









            

class voice1():
    def loaddata(self, filepath):
        '''
        :param filepath: 文件路径，为wav文件
        :return: 如果无异常则返回True，如果有异常退出并返回False
        self.wave_data内储存着多通道的音频数据，其中self.wave_data[0]代表第一通道
        具体有几通道，看self.nchannels
        '''
        if type(filepath) != str:
            raise (TypeError, 'the type of filepath must be string')
        p1 = re.compile('\.wav')
        if p1.findall(filepath) is None:
            raise (IOError, 'the suffix of file must be .wav')
        try:
            f = wave.open(filepath, 'rb')
            params = f.getparams()
            print (params,type(params))
            self.nchannels, self.sampwidth, self.framerate, self.nframes = params[:4]
            print (self.nchannels,self.sampwidth, self.framerate,self.nframes)
            str_data = f.readframes(self.nframes)
            #测试nframes
            #str_data = f.readframes(16)
            self.wave_data = np.fromstring(str_data, dtype=np.short)
            print (self.wave_data,self.wave_data.shape)
            self.wave_data.shape = -1, self.sampwidth
            #self.wave_data.shape = -1 此处是一个整体（-1.2）的形式
            print (self.wave_data)
            print (self.wave_data.shape)
            
            self.wave_data = self.wave_data.T
            print (self.wave_data)
            f.close()
            self.name = os.path.basename(filepath)  # 记录下文件名
            return True
        except:
            raise (IOError, 'File Error')
 
    def fft(self, frames=40):
        '''
        整体指纹提取的核心方法，将整个音频分块后分别对每块进行傅里叶变换，之后分子带抽取高能量点的下标
        :param frames: frames是指定每秒钟分块数
        :return:
        '''
        block = []
        fft_blocks = []
        self.high_point = []
        blocks_size = self.framerate // frames  # block_size为每一块的frame数量
        print (blocks_size)
        blocks_num = self.nframes // blocks_size  # 将音频分块的数量
        print (blocks_num)
        for i in range(0, len(self.wave_data[0]) - blocks_size, blocks_size):
            block.append(self.wave_data[0][i:i + blocks_size])
            print (np.fft.fft(self.wave_data[0][i:i + blocks_size]))
            fft_blocks.append(np.abs(np.fft.fft(self.wave_data[0][i:i + blocks_size])))
            print (fft_blocks,len(fft_blocks[0]),len(fft_blocks))
            self.high_point.append((np.argmax(fft_blocks[-1][:40]),
                                    np.argmax(fft_blocks[-1][40:80]) + 40,
                                    np.argmax(fft_blocks[-1][80:120]) + 80,
                                    np.argmax(fft_blocks[-1][120:180]) + 120,
                                    np.argmax(fft_blocks[-1][180:300]) + 180,
                                    np.argmax(fft_blocks[-1][300:600]) + 300,
                                    ))
            break
        print (len(self.high_point),self.high_point)




'''
#采样点大于频率的2倍。blocks_size=采样率/frames的结果进行分割。每秒被分割为frames个窗口
#fft按blocks_size大小进行变换。（注意2倍频率可能超出blocks_size范围）
#是否考虑平移和滚动窗口？
#第几秒作为关键数据。或者几秒钟的窗口数据作为关键数据
#fft函数给了一个时间窗口min_c

'''
class voice2():
    def loaddata(self, filepath):
        '''
        :param filepath: 文件路径，为wav文件
        :return: 如果无异常则返回True，如果有异常退出并返回False
        self.wave_data内储存着多通道的音频数据，其中self.wave_data[0]代表第一通道
        具体有几通道，看self.nchannels
        '''
        if type(filepath) != str:
            raise (TypeError, 'the type of filepath must be string')
        p1 = re.compile('\.wav')
        if p1.findall(filepath) is None:
            raise (IOError, 'the suffix of file must be .wav')
        try:
            f = wave.open(filepath, 'rb')
            params = f.getparams()
            #print (params,type(params))
            self.nchannels, self.sampwidth, self.framerate, self.nframes = params[:4]
            #print (self.nchannels,self.sampwidth, self.framerate,self.nframes)
            str_data = f.readframes(self.nframes)
            #测试nframes
            #str_data = f.readframes(16)
            self.wave_data = np.fromstring(str_data, dtype=np.short)
            #print (self.wave_data,self.wave_data.shape)
            self.wave_data.shape = -1, self.sampwidth
            #self.wave_data.shape = -1 此处是一个整体（-1.2）的形式
            #print (self.wave_data)
            #print (self.wave_data.shape)
            
            self.wave_data = self.wave_data.T
            #print (self.wave_data)
            f.close()
            self.name = os.path.basename(filepath)  # 记录下文件名
            return True
        except:
            raise (IOError, 'File Error')
 
    def fft(self, frames=20,min_c=1):
        '''
        整体指纹提取的核心方法，将整个音频分块后分别对每块进行傅里叶变换，之后分子带抽取高能量点的下标
        :param frames: frames是指定每秒钟分块数
        :return:
        '''
        block = []
        fft_blocks = []
        self.high_point = []
        blocks_size = self.framerate // frames  # block_size为每一块的frame数量
        #print (blocks_size)
        blocks_num = self.nframes // blocks_size  # 将音频分块的数量

        #self.play_figure(blocks_size)


        
        #print (blocks_num)
        #for i in range(0, len(self.wave_data[0]) - blocks_size, blocks_size):

        min_num = frames*min_c
        for i in range(0, min_num*blocks_size, blocks_size): 
            block.append(self.wave_data[0][i:i + blocks_size])
            #print (np.fft.fft(self.wave_data[0][i:i + blocks_size]))
            fft_blocks.append(np.abs(np.fft.fft(self.wave_data[0][i:i + blocks_size])))
            
            #print (fft_blocks,len(fft_blocks[0]),len(fft_blocks))
            '''
            self.high_point.append((np.argmax(fft_blocks[-1][:40]),
                                    np.argmax(fft_blocks[-1][40:80]) + 40,
                                    np.argmax(fft_blocks[-1][80:120]) + 80,
                                    np.argmax(fft_blocks[-1][120:180]) + 120,
                                    np.argmax(fft_blocks[-1][180:300]) + 180,
                                    np.argmax(fft_blocks[-1][300:600]) + 300,
                                    ))
            '''
        import functools
        import operator
        import itertools
        #print (len(fft_blocks),fft_blocks)
        #fft_blocks = functools.reduce(operator.concat,fft_blocks)
        #平铺方法https://www.cnblogs.com/wushaogui/p/9241931.html
        fft_blocks = list(itertools.chain.from_iterable(fft_blocks))
        #print (len(fft_blocks),fft_blocks[-10:])

        #print ([np.argmax(fft_blocks[1:i+10]) for i in range(3)])

        #print ([np.argmax(fft_blocks[i*blocks_size:(i+1)*blocks_size])+ i*blocks_size for i in range(40)])
        self.high_point = [np.argmax(fft_blocks[i*blocks_size:(i+1)*blocks_size])+ i*blocks_size for i in range(min_num)]
        #print (len(self.high_point),self.high_point)
        return self.high_point

        



    def play_figure(self,blocks_size):  
        y = np.fft.fft(self.wave_data[0][30*blocks_size:31*blocks_size])
        x = np.arange(blocks_size)
        abs_y = np.abs(y)
        normalization_y=abs_y/ blocks_size

        plt.subplot(121)
        plt.plot(x,abs_y,'black')

        plt.subplot(122)
        plt.plot(x,normalization_y,'black')


        #设置关闭时间
        plt.pause(3)
        plt.close()
        #plt.show()


def generate_feature1():
    p = voice2()

    audioList = os.listdir('../sample')
    #print (audioList)


    for k in range(1,3):
        feature = {}
        data = []
        for tmp in audioList:
            audioName = os.path.join('../sample', tmp)
            #print (audioName)
            if audioName.endswith('.wav'):
                p.loaddata(audioName)
                temp = p.fft(min_c=k)
                #print (temp[0],type(temp[0]))
                feature[tmp] = temp
                #print (feature)
        keys = list(feature.keys())
        #print (keys)
        #print (feature)
        
        for i in range(len(keys)-1):
            for j in range(i+1,len(keys)):
                #print (keys[i],keys[j],np.dot(feature[keys[i]],feature[keys[j]]))
                #print (cosine_similarity([feature[keys[i]],feature[keys[j]]]))
                
                data.append([keys[i],keys[j],feature[keys[i]],feature[keys[j]],cosine_similarity([feature[keys[i]],feature[keys[j]]])[0][1]])
        pd_data = pd.DataFrame(data,columns=["itemA","itemB","itemA_feature","itemB_feature","sim"])
        pd_data.to_csv('./sim_%d.csv'%k,index=False)
        
        





def fft_test():
    a = np.array([1,2,3])
    pdb.set_trace()
    b = np.fft.fft(a)
    print (b)





def test_1():
    p = voice1()
    #bofang
    #p.play('./zdqc.wav')
    p.loaddata('../sample/2A.wav')
    p.fft()


def test_2():
    p = voice2()
    #bofang
    #p.play('./zdqc.wav')
    p.loaddata('../sample/2A.wav')
    p.fft(min_c=2)



 
if __name__ == '__main__':
    '''
    p = voice()
    #bofang
    #p.play('./zdqc.wav')
    p.loaddata('./sample/2A.wav')
    p.fft()
    print (p.name)
    #p.trans_mp3_to_wav(filepath='./zhiduanqingchang.mp3',wpath='./zdqc.wav')
    
    demo1()

    #generate_feature()
    '''    



    
    #fft_test()
    #test_2()
    generate_feature1()








'''
1、采样率。1秒采样的数量44100。
2、声道=2。每个采样点由二维数组组成。
3、nframe反应了时长。每个frame由二维数组组成，除以采样率44100。反应音频时长

[ 0  0  0  0  0  0  0  0  1  1  2  1  3  2  4  3  5  4  6  5  7  6  9  8
 11  9 12 10 12 11 13 11] (32,)
[[ 0  0]
 [ 0  0]
 [ 0  0]
 [ 0  0]
 [ 1  1]
 [ 2  1]
 [ 3  2]
 [ 4  3]
 [ 5  4]
 [ 6  5]
 [ 7  6]
 [ 9  8]
 [11  9]
 [12 10]
 [12 11]
 [13 11]]
(16, 2)
[[ 0  0  0  0  1  2  3  4  5  6  7  9 11 12 12 13]
 [ 0  0  0  0  1  1  2  3  4  5  6  8  9 10 11 11]]





'''
