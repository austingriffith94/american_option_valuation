# american_option_Sf
The following code provided valuation of options, and the numerical approximation of the exercise boundary of American Put options.

## Provided
There are a few different ways this code can be viewed. There is the original python code, which was run in Spyder, but can also be run in the command line. A jupyter notebook for each code is also provided, should that be easier. Finally, there is also a set of LaTex files (of the jupyter notebook) and their respective pdf outputs should that be the most convenient way of viewing the output.

## Overview
The value of an option can determined using a binomial pricing model. For this set of code, a binomial tree is determined for various underlying initial values. From here, the intrinsic value, European option value, American option value, and Black Scholes price can all be determined across the time values given. This allows for comparison between the values and how they evolve with varying underlying prices.

For the second part of the codes, the Sf(t) value is determined for the American option. This varies for the call and put. For the American put, the Sf(t) represents the HIGHEST value at time t for which the value along the binomial tree is equal to the intrinsic value. The call option's Sf(t) is the LOWEST value at time t where the value is equal to the intrinsic value. Both of these represent the value boundary at which the option is held or exercised (determined by the valuation at a given state of the world, and the given probability, up and down states). Therefore, this code serves as a numerical approximation of the early exercise boundary of American options.

## Methods
In this particular case, the up, down and probability of motion and valuation for the underlying of the PUT options were determined by the following equations:

    u = 1 + σ (dt^0.5)
    d = 1 - σ (dt^0.5)
    p = 1 + r (dt^0.5) / 2σ

    Su = S0*u
    Sd = S0*d
    E[S1] = p*Su + (1-p)Sd

For the underlying of the CALL options, a continuous dividend was added. The valuation of the motion were set as:

    Su = S0*u*(1-q*dt)
    Sd = S0*d*(1-q*dt)

The parameters that can be adjusted to the user's liking include:

    T = time of expiration (assumed to be years in this case)
    r = risk free rate
    mu = the implied drift term of the underlying
    sigma = implied volatility of the underlying
    K = strike price for the options
    m = number of steps to break T into (determines time step, dt)
    q = continous dividend payout (annual %)

The valuation of the intrinsic was determined by performing simple addition/subtraction on the binomial tree, as shown below.

    max(S - K, 0) for call options
    max(K - S, 0) for put options

The valuation of the European option at each node is determined from the expected probability of the motion. The American option is calculated using the same formula, only an additional check is used to determine whether the intrinsic value at the future states are more valuable than the expected value (thus informing whether it is more valuable to exercise in the current moment). More information on the parameters and methods of valuation used can be found in the code.

## Results
![American Put Comparison](https://github.com/austingriffith94/american_option_valuation/blob/master/amerPut/output_6_0.png "Value of Put Option")

![American Call Comparison](https://github.com/austingriffith94/american_option_valuation/blob/master/amerCall/output_6_0.png "Value of Call Option")

For both the call and put, it can be seen that while the European option can dip below intrinsic value, the American never does. These graphs serve as a check on the accuracy of the approximation; the American option should always be at or above intrinsic value due to its early exercise choice.

![American Put Boundary](https://github.com/austingriffith94/american_option_valuation/blob/master/amerPut/output_8_0.png "Early Exercise Boundary for Increasing Underlying of American Put")

For the American put, the starting initial underlying value increases from right ot left (this is better labeled in the discretized plots also provided by the code). Zero values represent no early exercise occured at this time. Intuitively, the early exercise boundary value increases over time, since approaching maturity leads to the value approximation that tells the holder there is less opportunity to recover losses.

![American Put Max](https://github.com/austingriffith94/american_option_valuation/blob/master/amerPut/output_9_0.png "Maximum Value of the Early Exercise Boundary of American Put")

This relationship is most apparent when the varying S0 values are broken down, so that only the max exercise value is shown above. It is apparent that as time approaches 100 (expiration), the derivative of the curve will go to infinity, termination approaching the exercise price of 5. A day out, the option holder will know that if they are even a few points below the exercise price (and the probability of the underlying's motion known), it will be more cost effective to cash in at a loss than risk a downward motion.

![American Call Boundary](https://github.com/austingriffith94/american_option_valuation/blob/master/amerCall/output_8_0.png "Early Exercise Boundary for Increasing Underlying of American Call, with Dividend")

The same principles hold true for the early exercise boundary of the American call. For this graph, the initial underlying value increases from right to left as well.

![American Call Min](https://github.com/austingriffith94/american_option_valuation/blob/master/amerCall/output_9_1.png "Minimum Value of the Early Exercise Boundary of American Call")

The minimum exercise boundary for the call shows that even for the worst case scenarios (starting at S0 = 2.9), the user will not exercise. There are no early exercises till t=25, as the more open ended nature of the call option allows for more opportunity of a profit.


