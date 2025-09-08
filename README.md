# COL333_2025_A2 Stones and Rivers Game

This repository contains the code supporting the code for Agent and Engine for the River and Stones.

## Details
This is a course assignment for the graduate-level Artificial Intelligence course taught by [**Prof. Mausam**](https://www.cse.iitd.ac.in/~mausam/)

## Rules
You can find the documentation to get all the rules.

## Run Instructions
Here are the instructions used to match ai or human players against each other.


### Human vs Human
```sh
python gameEngine.py --mode hvh
```
### Human vs AI

```sh
python gameEngine.py --mode hvai --circle best
```
### AI vs AI

```sh
python gameEngine.py --mode aivai --circle best --square best
```

### No GUI
```sh
python gameEngine.py --mode aivai --circle best --square best --nogui
```
