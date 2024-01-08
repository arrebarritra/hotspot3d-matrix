# Hostpot 3D using CUDA matrix libraries
Group project in DD2360 Applied GPU Programming: Modification of the Rodinia Benchmark Suite application "Hotspot 3D" using cuSPARSE and cuBLAS matrix libraries

## Compilation Instructions
To compile the code, execute the following commands:

```
make clean
make 
```

## Data Source
The data being used is from Rodinia 3.1 and can be downloaded here:
[http://lava.cs.virginia.edu/Rodinia/download_links.htm]

The relevant data is located in the directory **'/data/hotspot3D/'** and includes the following files:
- Power files
    - power_64x8
    - power_512x2
    - power_512x4
    - power_512x8

- Temp files
    - temp_64x8
    - temp_512x2
    - temp_512x4
    - temp_512x8


## Running the Application
If the data files are placed in the data folder, the application can be run with:
`./run`

For the application to be run with different data or specified with different data paths:

`./3D ROW/COLS LAYERS ITERATIONS POWER_PATH TEMP_PATH OUTPUT_FILE ORIGINAL_OUTPUT_FILE`

_Note:_ The dimensions specified in **'POWER_PATH'** and **'TEMP_PATH'** should correspond to **'ROWS/COLS'** and **'LAYERS'**.

For example:

`./3D 512 8 100 ./data/power_512x8 ./data/temp_512x8 ./res/output_512x8 ./res/output_512x8_original`


## Output
The expected generated output will be:

- Time for setup: x (s)
- Time: x (s)
- Time Original: x (s)
- Accuracy: x
- Accuracy Original: x

_Note:_ Outputs without the label "Original" are the results from our modified implementation.