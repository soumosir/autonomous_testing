

## Build dependencies
* Numpy
* Opencv
* Argparse
* Matplotlib [OPTIONAL]

## Run Instructions
* Cd to the Code directory
```
cd Code/
```
* Run the following command to see the options
```
python main.py -h
```
* You will see the following options
```
usage: main.py [-h] [--problem PROBLEM] [--inDir INDIR] [--outDir OUTDIR] [--video VIDEO] [--display DISPLAY] [--save SAVE]

optional arguments:
  -h, --help         show this help message and exit
  --problem PROBLEM  problem number you want to solve, options- 1/2/3, Default: 1
  --inDir INDIR      path to input data folder, Default: ../Data/adaptive_hist_data
  --outDir OUTDIR    path to output folder where results will be stored, Default: ../Data/output
  --video VIDEO      Video File Path, Default: ../Data/whiteline.mp4
  --display DISPLAY  display Results, Default: 1
  --save SAVE        save Results, Default: 0

```
```
* To Solve the problem 3 run the following command, to see the results set display to 1, to save the results set save to 1
```
python Code/main.py --problem 1 --video "challenge.mp4" --outDir "output/" --display 1 --save 0
```

python3 Code/main.py --problem 1 --video "challenge_snow.mp4" --outDir "output/" --display 1 --save 1

test error
python3 testTurnPredictionError.py