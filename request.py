import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'preg':4, 'plas':110, 'pres':60, 'skin':30, 'test':60, 'mass':30.2, 'pedi':0.369, 'age':40})

print(r.json())