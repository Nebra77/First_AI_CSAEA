# First_AI_CSAEA
A project that involves creating my fist full GPT for my project for project week in October.
This AI trains on the data of some of Shakespeare's work, and then goes character by character to create its own stories. 

## Functionality
1. Take all data, turn each character into an integer, and split into 
    1. Training data, 90% of data, used for primary training
    2. Valuation data, 10% of data, checked against to check for overfitting
2. Take random 256 character chunks to train on (find patterns to find next character), then take the next character from each as the answer key
3. Turn each character into a vector + position value
4. Pass through transformer blocks
    1. Self-Attention: character "looks back" at past characters for significance
    2. Feed-Forward Network: each character processes what it gathered
5. Repeated thousands of times: predict next character, calculate how far off it was (loss), then adjust all the weights so the values move closer
6. *Making text*: predict next character, add it, then repeat with added character

## How to Run
1. Install PyTorch: `pip install torch`
2. Run the script (a GPU is best):
```bash
   python main.py
```
-- Note that the trained model is not saved, so it has to be retrained each run.


## My struggles
For this project, my main issue was simply getting all of the information into my head. It was incredibly overwheling at first, and a lot of it wasn't really intuitive for me. On top of this, it's been a while since I've worked in python, so I needed some time to just get used to the language again. However, both of those problems faded with time, and I can confidetnly say this is one of my personally benifical projects so far. 