# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:12:48 2017

@author: Erdig
"""
#PLOTTING
import pandas as pd
import numpy as np
mydata=pd.read_csv('PIKETTY-TS6_3.csv', index_col=0)
mydata.index = pd.to_datetime(mydata.index)
type(mydata)

mydata.head()

mydata.plot(figsize=(12,8))

import seaborn as sns
mydata.plot(figsize=(12,8))

##BOKEH
from bokeh.plotting import show
from bokeh.io import output_notebook
from bokeh.charts import Line

#doesn't work in spyder!!
output_notebook()
p= Line(mydata,legend='bottom_right')
p.width=600
p.height=400
show(p)

#HTML
from bokeh.plotting import figure
from bokeh.palettes import RdYlBu11
from bokeh.plotting import show, output_file, reset_output

output_file("bokehplot.html")

mytools = ['pan', 'box_zoom','resize', 'wheel_zoom', 'reset'] # Does not allow save and question options
p = figure(width=800,height=600,tools=mytools, x_axis_type='datetime')
cols = mydata.columns.values[:-1]
print cols
mypalette=RdYlBu11[0:len(cols)]
print mypalette
dates = mydata.index.values
print dates
for indx, col in enumerate(cols):
    p.line(dates,mydata[col],color=mypalette[indx],line_width=3)

show(p)
reset_output()

#mydata = quandl.get("PIKETTY/TS9_3")
#mydata1 = quandl.get("PIKETTY/TS9_5")
mydata = pd.read_csv('PIKETTY-TS9_3.csv', index_col=0)
#Transform to datetime
mydata.index = pd.to_datetime(mydata.index)

mydata1 = pd.read_csv('PIKETTY-TS9_5.csv', index_col=0)
mydata1.index = pd.to_datetime(mydata1.index)

print (mydata.head())
print (mydata1.head())

#remove 0.1
for col in mydata.columns:
    #Drop if 0.1 is in column name
    if '0.1' in col:
        mydata.drop(col,axis=1,inplace=True)
        
    
for col in mydata1.columns:
    #Drop if 0.1 is in column name
    if '0.1' in col:
        mydata1.drop(col,axis=1,inplace=True)
    
print mydata.columns
print mydata1.columns
print mydata.index
print mydata1.index

#Full outer join
result = mydata.join(mydata1, how='outer')
print np.shape(result)

result = mydata.join(mydata1, how='outer')
print np.shape(result)
cols = []
for col in result.columns:
    #Delete ': Top 1%' and ': 1%' and append
    col=col.replace(': Top 1%','')
    col=col.replace(': 1%','')
    cols.append(col)
    
result.columns = cols

print result.head()
for i in result.columns:
    print i

