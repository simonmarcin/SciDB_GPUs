import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

class plotClusterStackBar:
    font_size = 14
    plt.style.use(['seaborn-paper','seaborn-white'])
    plt.rc("font", family="serif")
    plt.rc('font', size=font_size)          # controls default text sizes
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)    # legend fontsize
    plt.rc('figure', titlesize=font_size)  # fontsize of the figure title
    
    def __init__(self,bars,clusters,layers):
        #nbrBars is a list where each element describes the nbr of stacked bars per element
        self.bars = bars
        self.clusters = clusters
        self.layers = layers
        self.bar_locations = np.arange((1+bars)*clusters -1)
        self.data = list()
        self.ticks_locations = list()
        self.titel=""
        self.hatches = []
        self.colors = []
        for i in range(0,layers):
            self.data.append(np.zeros( (1+bars)*clusters-1 ) )
            self.hatches.append('')
            self.colors.append(i)
        
    def setBarContent(self,nbrBar,nbrLayer,content):
        for i in range(0,len(content)):
            self.data[nbrLayer][i*(self.bars+1)+nbrBar]=content[i]
    
    def setXticks(self,ticks):
        self.xticks = ticks
        for i in range(0,self.clusters):
            self.ticks_locations.append(i*(self.bars+1)  +((self.bars+1)/2) -1)
    
    def setlegend(self,legend):
        self.legend=legend
        
    def setXYLabels(self,xlabel,ylabel):
        self.xlabel=xlabel
        self.ylabel=ylabel
    
    def setTitel(self,titel):
        self.titel = titel
        
    def setHatchToLayer(self, layer, hatch="//"):
        self.hatches[layer] = hatch
        
    def setColorToLayer(self, layer, color):
        self.colors[layer] = color
        
    def plot(self,log=False):
        plt.figure(figsize=(9,5))
        plt.xticks(self.ticks_locations, self.xticks)
        plt.yscale('linear')
        if log: plt.yscale('log')
        tmp=plt.bar(self.bar_locations, self.data[0],label=self.legend[0], 
                   hatch=self.hatches[0], color=cm.Dark2(self.colors[0]))
        labels=list()
        labels.append(tmp)
        offset=self.data[0]
        for i in range(1,self.layers):
            tmp=plt.bar(self.bar_locations, self.data[i], bottom=offset,label=self.legend[i], 
                        hatch=self.hatches[i], color=cm.Dark2(self.colors[i]))
            labels.append(tmp)
            offset=[i+j for i,j in zip(offset,self.data[i])]
            
        plt.legend(handles=labels,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.titel)
        plt.show()
        