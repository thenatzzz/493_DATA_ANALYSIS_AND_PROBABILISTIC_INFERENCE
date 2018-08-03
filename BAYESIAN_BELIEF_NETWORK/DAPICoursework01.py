#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python
from DAPICourseworkLibrary import *
from numpy import *
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
# Coursework 1 task 1 should be inserted here
    prior = zeros((noStates[root]), float ) #Set array to hold number of state

    for row in range(theData.shape[0]):      #This 'for loop' goes row by row in theData
        for state in range(noStates[root]):  #This 'for loop' goes into all states of root parameter
            if(theData[row][0] == state):
                prior[state] += 1  #count number of all state and put in each individual array
    sum_all_state = prior.sum()
#    print(prior/sum_all_state)
    return prior/sum_all_state

# end of Coursework 1 task 1
# ------------------------------------------------------------------------------

# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
# Coursework 1 task 2 should be inserte4d here
    count_list = []                      # This following code works similar to Prior function
    for state in range(noStates[varP]):
        count_list.append(0)
        for row in range(theData.shape[0]):
            if state == theData[row][0] :
                count_list[state] += 1
    for row in range(noStates[varC]):   # row ,or each state in Child node
        for column in range(noStates[varP]):   # column ,or each state in Parent node
            cPT[row][column]= getValue(column,theData,varC,row)/float(count_list[column])
    #print(cPT)
    return cPT

def getValue(which_parent,theData,varC,child_state):
    count  = 0
    for row in range(theData.shape[0]):        # going down row by row until the end of matrix
        if(theData[row][0] == which_parent):   # going down row by row of state 0 of parent
            if theData[row][varC]==child_state: # going down row by row of state 'varC' of child
                count += 1                          # if child_state of child matches parent states,we count +1
    return count
    # end of coursework 1 task 2
# ----------------------------------------------------------------------------------------
# Function to calculate the joint probability table of two variables in the data set

def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )

#Coursework 1 task 3 should be inserted here
    number_elem  = theData.shape[0]
    for row in range(noStates[varRow]):
        for column in range(noStates[varCol]):
            jPT[row][column] = getValue(column,theData,varRow,row)/float(number_elem)
# end of coursework 1 task 3
    #print(jPT)
    return jPT
# -------------------------------------------------------------------------
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here
    for index, element in enumerate(aJPT):
        for num_prior_state in range(len(prior)):
            aJPT[index][num_prior_state]=element[num_prior_state]/prior[num_prior_state]
    #print(aJPT)
# coursework 1 taks 4 ends here
    return aJPT
#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = ones((naiveBayes[0].shape[0]),float)
# Coursework 1 task 5 should be inserted here
    prior = naiveBayes[0]
    cPTs = naiveBayes[1:]

    for index, element in enumerate(theQuery):
        for prior_state in range(rootPdf.size):
            rootPdf[prior_state] = rootPdf[prior_state]*cPTs[index][element][prior_state]
    rootPdf = multiply(rootPdf,prior)

    alpha = 1/sum(rootPdf)
    for prior_state in range(rootPdf.size):
        rootPdf[prior_state] = rootPdf[prior_state]*alpha

    #print(rootPdf)
# end of coursework 1 task 5
    return rootPdf
#
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
# Coursework 2 task 1 should be inserted here


# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here


# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here


# end of coursework 2 task 3
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []

    return array(spanningTree)
#
# End of coursework 2
#

#
# main program part for Coursework 1
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
theData = array(datain)

# ----------------------- 1 ----------------------------
AppendString("DAPIResults01.txt","Coursework One Results by Nattapat Juthaprachakul(nj2217), MSc Computing Science")
AppendString("DAPIResults01.txt","") #blank line
AppendString("DAPIResults01.txt","The prior probability of node 0")
prior = Prior(theData, 0, noStates)
AppendList("DAPIResults01.txt", prior)
#
# continue as described
#
# ----------------------- 2----------------------
y= CPT(theData,2,0,noStates)
AppendString("DAPIResults01.txt","The CPT")
AppendArray("DAPIResults01.txt", y)

# ---------------------  3------------------------
x = JPT(theData,2,0,noStates)
AppendString("DAPIResults01.txt","The JPT")
AppendArray("DAPIResults01.txt", x)
#------------------------ 4 ----------------------------
aJPT = JPT(theData,2,0,noStates)
aJPT2CPT = JPT2CPT(aJPT)
AppendString("DAPIResults01.txt","The JPT2CPT")
AppendArray("DAPIResults01.txt", aJPT2CPT)
#-------------------------5------------------------------------
theQuery = [[4,0,0,0,5],[6,5,2,5,5]]
naiveBayes = []
for i in range(1,len(theQuery[0])+1):
    cpt = CPT(theData,i,0,noStates)
    naiveBayes.append(cpt)
naiveBayes = [prior]+naiveBayes
first_query  = Query(theQuery[0],naiveBayes)
second_query  = Query(theQuery[1],naiveBayes)
AppendString("DAPIResults01.txt","The Query[0]")
AppendList("DAPIResults01.txt", first_query)
AppendString("DAPIResults01.txt","The Query[1]")
AppendList("DAPIResults01.txt", second_query)
