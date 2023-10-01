import pandas as pd
import argparse
import csv

ssd = pd.read_csv('C:\\Users\\Sasha\\OneDrive\\Desktop\\SoilHealthCard\\farmer_info\\farmer_details.csv',header=None, encoding='ISO-8859-1')

ssd.columns=['Name','Email Id','Address','Geo Position (GPS)','Date of Sample Collection','Survey No., Khasra No,/ Dag No,','Farm Size']

irr=['Irrigated (Bore well)','Non-Irrigated','Irrigated (Pipeline)','Irrigated (Drip Irrigation)','Non-Irrigated']

ssd['Irrigation Status']=irr

#Basic details
ssd1=ssd[['Name', 'Email Id', 'Address']]
#Farm details
ssd2=ssd[['Date of Sample Collection', 'Survey No., Khasra No,/ Dag No,',
       'Farm Size','Geo Position (GPS)']]

SSD1=ssd1.T
SSD2=ssd2.T

Lst=[]
f='Farmer'
for i in range(len(ssd)):
    I=str(i+1)
    F=f+I
    Lst.append(F)

SSD1.columns=Lst
SSD2.columns=Lst

Personal_Info=[]
Farm_Info=[]

for i in range(len(ssd)):
    x=SSD1[[Lst[i]]]
    x = x.style.set_caption('Farmer Details')
    Personal_Info.append(x)

    y=SSD2[[Lst[i]]]
    y=y.style.set_caption('Farm Details')
    Farm_Info.append(y)
