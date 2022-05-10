# Safe Reinforcement Learning

***

## Overview


### Implementations 



### [Project Report]()

### [Presentation]()

## Usage

Following are the list of commands to compile \& run the codes for the various implementations mentioned above:

To install Anaconda follow the instructions in the following webpage:  
https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-20-04-quickstart

Create a conda environment for the Safe RL:  
```
$ conda create --name safe_rl  
```
Switch to the newly create environment:  
```
$ conda activate safe_rl  
```

Once in the desired environment install the following packages:

## Running the Experiments
`Unicycle` involves a unicycle robot tasked with reaching a desired location while avoiding obstacles

### Training: 

* Training the proposed approach: 
`python main.py --env Unicycle --gamma_b 20 --max_episodes 400 --cuda --updates_per_step 2 --batch_size 512  --seed 12345 --model_based`

* Training the baseline:
`python main.py --env Unicycle --gamma_b 20 --max_episodes 400 --cuda --updates_per_step 1 --batch_size 256  --seed 12345 --no_diff_qp`

* Training the modified approach from "End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks": 
`python main.py --env Unicycle --gamma_b 20 --max_episodes 400 --cuda --updates_per_step 1 --batch_size 256   --seed 12345 --no_diff_qp --use_comp True`

### Testing

* To test, add `--mode test` and `--resume /path/to/output/{1}-run-{2}`, where `{1}` is the env name and `{2}` is the experiment number, to any of the commands above.

## LICENSE

The code is licenced under the MIT license and free to use by anyone without any restrictions.

***

<p align='center'>Created with :heart: by <a href="https://github.com/tayalmanan28">Manan Tayal</a> </p>
