# Follow the instructions below to run this code.
1. Make sure you give the correct path to the "best" model here https://github.com/shahaamirbader/GDN/blob/main/Visulaization/graph_visualization.py#L88  generated from the GDN code. 
2. The output will be saved as png into the same folder, where this file located. 

# Run commands:

To select the high anomaly as central node:
```
python anomaly.py auto out.png
```

To select a custom central node:
```
python anomaly.py [node idx] out.png
```
NOTE: The [node idx] can be found from the 'list.txt' file. For the feature listed on 2nd row, [node idx]= 1, so on and so forth.

Example (Use feature '0' as central node):
```
python anomaly.py 0 out.png
```
