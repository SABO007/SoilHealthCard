
print(predict_cf)

n=len(ssd)-1
print("The Farmer Survey number taken is (Last Survey No.)",n+1)

from io import StringIO, BytesIO
from xhtml2pdf import pisa
from string import Template as HTMLTemplate

# Convert dataframes to HTML
html1 = Personal_Info[n].to_html()
html2 = Farm_Info[n].to_html()
html3=predict_cf.to_html()
html4 = SHC_c.to_html()


# Create HTML template
html_template = HTMLTemplate('''
   <html>
     <head>
       <style>
         table, th, td {
           border: 1px solid black;
           border-collapse: collapse;
           padding: 5px;
         }
       </style>
     </head>
     <body>
       <h1>Farmer Information</h1>
       $html1
       $html2
       <h1>Soil Health Card</h1>
       $html3
       $html4
     </body>
   </html>
''')

# Merge HTML content
html = html_template.substitute(html1=html1, html2=html2, html3=html3, html4=html4)

# Convert HTML to PDF
pdf = BytesIO()
pisa.CreatePDF(BytesIO(html.encode('utf-8')), pdf)

destination_folder = "C:\\Users\\Sasha\\OneDrive\\Desktop\\Plan-B (AI in Farming)\\SHC_external input\\"
file_name = "SoilHealthCard.pdf"
file_path = destination_folder + file_name

# Save PDF to file
with open(file_path, 'wb') as f:
    f.write(pdf.getvalue())