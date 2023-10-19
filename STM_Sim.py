#### This code simulates the application of the strong transition model (STM) for predicting seat outcomes in UK general elections
### The code first starts by simulatiing previous election results and polling data, then applies STM as if it were real data.
### There is some basic visualisation and outputting of results via text from the console.
### Comments have been put around print statements to simplfy the output, but these can be removed if you wish to get
### a better look at excatly what the program is doing.

import numpy as np
import matplotlib.pyplot as plt

# Seeding can be turned on here to investigate code with consistent random numbers
#np.random.seed(0)

alpha = 0.2 # Strong voter threshold
N_parties = 3
N_seats = 13

Seat_Counter = np.zeros([N_parties]) #Initialise a counter to see which party initially has the most seats
# We will use simulated polling data, generated from a random distribution, in this case a uniform distribution
S = np.random.uniform(0,1,3)
VoteShares_National_Predicted = np.round(S/sum(S),2)
print("Polling data: ", VoteShares_National_Predicted)




### Firstly, since this is a simulation, we will generate a random data set of voter shares in a chosen number of seats, then apply the STM model to the simulated data ###
### this could be easily adapted to a real data set from an excel workbook ###


# Initialise seat-party matrix (vote shares in each seat for previous election) V(n,m) is the vote share of party m in seat n.
V_SP = np.empty([N_seats, N_parties])
V_SP_Strong = np.empty([N_seats, N_parties])
V_SP_Weak = np.empty([N_seats, N_parties])
# Randomly generate vote shares in each seat, in this case we choose them from a dirichlet distribution, but this can be changed at will.
for i in range(0,N_seats):
    V_SP[i] = np.round(np.random.dirichlet(np.ones(N_parties), size=1), 2)

print("Seat-Party Matrix")
print(V_SP)

# Count seats of each party
for i in range(0,N_seats):
    index = np.argmax(V_SP[i])
    Seat_Counter[index] +=1
for i in range(0,N_parties):
    print("Party: ", i+1, " Seats: ", int(Seat_Counter[i]))

print("Current Leader is Party: ", np.argmax(Seat_Counter) + 1)
# Initialise seat turnouts, which will also be randomly generated between a chosen min and max
TO_seats = np.empty([N_seats])
for i in range(0,N_seats):
    TO_seats[i] = np.random.randint(10000,100000,1)

print("Seat Turnouts: ", TO_seats)

# National turnout
TO = TO_seats.sum()
print("National Turnout: ",  TO)

# Calculate national vote shares
## Define national vote share vector and calculate national vote shares from seat-party matrix
VoteShares_National = np.empty([N_parties])
for i in range(0, N_parties):
    s = 0
    for j in range(0,N_seats):
        s += TO_seats[j]*V_SP[j,i]
    VoteShares_National[i] = s/TO
# Round national vote shares off to 2 decimal places for simplicity
VoteShares_National = VoteShares_National
print("National vote shares", np.round(VoteShares_National,2))




### We will now caluclate the share of "strong" voters for each party, i.e. those above the 20% threshold
Vs = np.empty([N_parties])
for i in range(0,N_parties):
    s = 0
    for j in range(0,N_seats):
        s += TO_seats[j]*max(V_SP[j,i]-alpha, 0)
        V_SP_Strong[j,i] = max(V_SP[j,i]-alpha, 0)
        V_SP_Weak[j,i] = V_SP[j,i] - max(V_SP[j,i]-alpha, 0)
    Vs[i] = s/TO

print("Counting over all seats, this breaks down into the following strong and weak voter shares")
print("National strong voter share: ", np.round(Vs,2))
# Calculate national weak vote shares
Vw = VoteShares_National-Vs
print("National weak voter share: ", np.round(Vw,2))

# We will now calculate the predicted national strong and weak voter shares from the simulated polling data
print("Polling data: ", VoteShares_National_Predicted)
print("In the STM model, strong voters are retained until there are no more weak voters left")

## Predicted national strong vote shares
Ps = np.minimum(VoteShares_National_Predicted, Vs)
print("Predicted national strong voter shares: ", np.round(Ps,2))
## Predicted national weak vote shares
Pw = np.maximum(VoteShares_National_Predicted - Vs, 0)
print('Predicted national weak voter shares: ', np.round(Pw,2))


# The last step is to calculate the new vote shares in each party, given that all weak voters defect first.
# The proportion of weak voters that did not defect is calculated from the ratio of
# the predicted national weak voter share (Pw) to the national weak voter share calculated from the previous election (Vw)


# Fraction of weak voters that did not defect
WeakVoterRatio = np.round(Pw/Vw,4)
print("Proprtion of weak voters that did not defect: ",  WeakVoterRatio)
print("If >1, then the party gained voters")
# To calculate the new party vote shares in each seat, we now apply the WeakVoterRatio to the weak voter shares for seat, to calculate the new share of weak voters.
# All strong voters are assumed to not defect, thus the final vote shares in each seat are the previous strong vote shares + the new weak vote shares
"""print("Strong voter seat-party matrix:")
print( V_SP_Strong)
print("Weak voter seat-party matrix:")
print(V_SP_Weak)"""
V_SP_New_Strong = np.empty([N_seats, N_parties])
V_SP_New_Weak = np.empty([N_seats, N_parties])

# Check to see if weak voters for each party increased or decreased
for i in range(0,N_parties):
    if Pw[i]-Vw[i] <= 0:
        # Share of weak voters for party i decreased
        for j in range(0,N_seats):
            V_SP_New_Weak[j,i] = V_SP_Weak[j,i]*WeakVoterRatio[i]
    else:
        # Weak voter share increased
        Share_w = np.empty([N_parties])
        Swing = np.empty([N_seats])
        N = 0
        for k in range(0,N_parties):
            N += max(Pw[k]-Vw[k],0)

        for k in range(0,N_seats):
            for j in range(0,N_parties):
                Swing[k] = V_SP_Weak[k,j]*max(1-(Pw[j]/Vw[j]),0)
                Share_w[j] = max(Pw[j] - Vw[j], 0) / N
        for i in range(0,N_parties):
            for j in range(0,N_seats):
                V_SP_New_Weak[j,i] = V_SP_Weak[j,i] + Share_w[i]*Swing[j]


# In the STM model, strong voters cannot increase between elections, they can only stay the same or get smaller, all new voters are assumed to be 'weak voters'
for i in range(0,N_parties):
    for j in range(0,N_seats):
        V_SP_New_Strong[j,i] = V_SP_Strong[j,i]*(Ps[i]/Vs[i])

"""print("New weak voter shares")
print(np.round(V_SP_New_Weak,2))
print("New strong voter shares")
print(np.round(V_SP_New_Strong,2))"""

V_SP_New = np.round(V_SP_New_Strong + V_SP_New_Weak,2)

print("Predicted Seat-Party Matrix: ")
print(V_SP_New)

# Count new seats of each party
Seat_Counter_New = np.zeros([N_parties])
for i in range(0,N_seats):
    index = np.argmax(V_SP_New[i])
    Seat_Counter_New[index] +=1
for i in range(0,N_parties):
    diff = int(Seat_Counter_New[i] - Seat_Counter[i])
    print("Party: ", i+1, " Seats: ", int(Seat_Counter_New[i]), "(",diff,")")
print("New Leader is Party: ", np.argmax(Seat_Counter_New)+1)





x1 = np.arange(V_SP.shape[0])
dx1 = (np.arange(V_SP.shape[1])-V_SP.shape[1]/2.)/(V_SP.shape[1]+2.)
d1 = 1./(V_SP.shape[1]+2.)

fig1, ax=plt.subplots()
for i in range(V_SP.shape[1]):
    ax.bar(x1+dx1[i],V_SP[:,i], width=d1, label="Party {}".format(i+1))
plt.xlabel("Seats")
plt.ylabel("Previous Total Vote Share")
plt.legend(framealpha=1).set_draggable(True)

x2 = np.arange(V_SP_New.shape[0])
dx2 = (np.arange(V_SP_New.shape[1])-V_SP_New.shape[1]/2.)/(V_SP_New.shape[1]+2.)
d2 = 1./(V_SP_New.shape[1]+2.)

fig2, ax=plt.subplots()
for i in range(V_SP_New.shape[1]):
    ax.bar(x2+dx2[i],V_SP_New[:,i], width=d2, label="Party {}".format(i+1))
plt.xlabel("Seats")
plt.ylabel("Predicted Total Vote Share")
plt.legend(framealpha=1).set_draggable(True)




plt.show()

# some final thoughts on how to extend/improve this model.
# STM assumes a global strong voter threshold alpha that applies the same to each seat. This is probbaly not well reflected in realisty
# Some seats are obviously much safer for certain parties than others. So the model could perhaps be imporved by accounting for regional
# variability in alpha. However if one did want to assume a global alpha, some hypothesis testing and paramter estimation
# could be implemented as part of a larger analysis to determine a more representative value of alpha.
