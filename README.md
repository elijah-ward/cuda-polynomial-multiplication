
# CUDA Polynomial Multiplication

### Setup

Compile Binaries:

`make`

Run Question 1 Tests:

`make test_q1`

Run Question 2 Tests:

`make test_q2`

Run All Tests:

`make test_all`

# Usage

## Arguments

There are 3 modes the program can be run in:

- `run`: This is the default mode that will randomly generate coefficients for the input polynomials
- `dev`: The same as run but also displays the values of the workspace used to reduce the terms of each X^n value
- `test`: Uses coefficients of 1 (for easy verification of the result) for all terms an displays whether the computed result is correct or not

The program runs with the following arguments:

- Argument 1: `Mode` - Execution Mode. One of (`run`, `dev`, `test`).
- Argument 2: `Question Id` - Integer value of the question from the assignment to determine the block, thread configuration. One of (`1`, `2`).
- Argument 3: `Number of Terms` - Integer value for desired number of terms for the input polynomials. Must be a power of 2 and no greater than 512.
- Argument 4: `Modulo P value` - Integer value by which all calculations will be modulo'd by. Should be a small prime like 103.
- Argument 5: `Number of Threads` - Integer value for the desired number of threads. One of (`64`, `128`, `256`, `512`). Only applies for `Question Id == 2`.

## Example Executions

**Run in default mode with 64 terms, modulo p as 103 for question 1:**

`./poly_mult run 1 64 103`

**Run in dev mode with 128 terms, modulo p as 97 for question 1:**

`./poly_mult dev 1 128 97`

**Run in test mode with 512 terms, modulo p as 103 for question 1:**

`./poly_mult test 1 512 103`

**Run in test mode with 512 terms, modulo p as 103 for question 2:**

`./poly_mult test 2 512 103`

