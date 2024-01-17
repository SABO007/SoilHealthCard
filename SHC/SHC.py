import pandas as pd
def shc_generation(details):

    details['Nitrogen']= 1.35*10*details['Nitrogen']
    details['Phosphorous'] = 1.35*10*details['Phosphorous']
    details['Potassium'] = 1.35*10*details['Potassium']
    details['Electric Conductivity']= 0.1*details['Electric Conductivity']

    details['Salinity']=details['Electric Conductivity']

    details=details[['Nitrogen', 'Phosphorous', 'Potassium',
        'pH', 'Electric Conductivity','Salinity','Temperature', 'Moisture']]

    details=details.round(2)
    print('details: ', details)
    detail=details.iloc[0:1,:]

    SHC=detail.melt( 
            var_name="Parameter", 
            value_name="Test value")

    # # Ideal Range
    # # Tempertaure=> 20-30
    # # Moisture=> (Paddy, Sugarcane=> 80-85), (Cotton, Maize=> 50-60)=> 50-75%
    # # N=> 280-560
    # # P=> 11-26
    # # K=> 120-280
    # # pH=> >7, =7, <7
    # # Electrical Conductivity=> 0-2 dS/m

    std=['280-560','11-26','120-280','7, Neutral','0-2','2-4','20-30','50-75']

    unit=['Kg/Ha','Kg/Ha','Kg/Ha','H Potenz','dS/m','dS/m','°C','%']

    SHC['Unit']=unit

    SHC['Rating']=''

    # # N
    if (SHC['Test value'][0]<280):
        SHC['Rating'][0]='Low'
    elif(SHC['Test value'][0]>=280 and SHC['Test value'][0]<=560):
        SHC['Rating'][0]='Medium'
    elif(SHC['Test value'][0]>560):
        SHC['Rating'][0]='High'

    # P
    if (SHC['Test value'][1]<11):
        SHC['Rating'][1]='Low'
    elif(SHC['Test value'][1]>=11 and SHC['Test value'][1]<=26):
        SHC['Rating'][1]='Medium'
    elif(SHC['Test value'][1]>26):
        SHC['Rating'][1]='High'

    # K
    if (SHC['Test value'][2]<120):
        SHC['Rating'][2]='Low'
    elif(SHC['Test value'][2]>=120 and SHC['Test value'][2]<=280):
        SHC['Rating'][2]='Medium'
    elif(SHC['Test value'][2]>280):
        SHC['Rating'][2]='High'

    # pH
    if (SHC['Test value'][3]<6):
        SHC['Rating'][3]='Acidic'
    elif(SHC['Test value'][3]>=6 and SHC['Test value'][3]<7):
        SHC['Rating'][3]='Slightly Acidic'
    elif(SHC['Test value'][3]==7):
        SHC['Rating'][3]='Neutral'
    elif(SHC['Test value'][3]>7 and SHC['Test value'][3]<=8):
        SHC['Rating'][3]='Slightly Basic'
    elif(SHC['Test value'][3]>8):
        SHC['Rating'][3]='Basic'

    # EC
    if (SHC['Test value'][4]<0):
        SHC['Rating'][4]='Low'
    elif(SHC['Test value'][4]>=0 and SHC['Test value'][4]<=2):
        SHC['Rating'][4]='Normal'
    elif(SHC['Test value'][4]>2):
        SHC['Rating'][4]='High'

    # Salinity
    if (SHC['Test value'][5]<2):
        SHC['Rating'][5]='Low'
    elif(SHC['Test value'][5]>=2 and SHC['Test value'][5]<=4):
        SHC['Rating'][5]='Normal'
    elif(SHC['Test value'][5]>4):
        SHC['Rating'][5]='High'

    # Temperature
    if (SHC['Test value'][6]<20):
        SHC['Rating'][6]='Cool'
    elif(SHC['Test value'][6]>=20 and SHC['Test value'][6]<=30):
        SHC['Rating'][6]='Ideal'
    elif(SHC['Test value'][6]>30):
        SHC['Rating'][6]='Hot'

    # Moisture
    if (SHC['Test value'][7]<50):
        SHC['Rating'][7]='Low'
    elif(SHC['Test value'][7]>=50 and SHC['Test value'][7]<=75):
        SHC['Rating'][7]='Medium'
    elif(SHC['Test value'][7]>75):
        SHC['Rating'][7]='High'

    SHC['Normal Level']=std

    # print("Soil Health Card")
    # print(SHC)

    def color_rating(val):
        if val == 'Low':
            color = 'yellow'
        elif val == 'High':
            color = 'red'
        elif val == 'Neutral':
            color = 'green'
        elif val == 'Medium':
            color = 'grey'
        elif val == 'Ideal':
            color = 'green'
        elif val == 'Normal':
            color = 'orange'
        elif 'Acidic' in val:
            color = 'red'
        elif 'Slightly Acidic' in val:
            color = 'yellow'
        elif 'Slightly Basic' in val:
            color = 'blue'
        elif 'Basic' in val:
            color = 'blue'
        elif 'Cool' in val:
            color = 'lightblue'
        elif 'Hot' in val:
            color = 'red'
        else:
            color = 'white'
        return 'background-color: %s' % color

    # apply the function to the 'Rating' column and export the styled DataFrame as an HTML file
    SHC_c = SHC.style.map(color_rating, subset=['Rating'])
    return SHC_c