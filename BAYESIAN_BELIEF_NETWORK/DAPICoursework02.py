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
    prior = zeros((noStates[root]), float )
    for i in range(theData.shape[0]):
        for j in range(noStates[root]):
            if(theData[i][0] == j):
                prior[j] += 1
    sum_all_state = prior.sum()
    return prior/sum_all_state

# end of Coursework 1 task 1
# ------------------------------------------------------------------------------

# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
# Coursework 1 task 2 should be inserte4d here
    count_list = []
    for i in range(noStates[varP]):
        count_list.append(0)
        for j in range(theData.shape[0]):
            if i == theData[j][0] :
                count_list[i] += 1

    for i in range(noStates[varC]):
        for j in range(len(count_list)):
            cPT[i][j]= getValue(j,theData,varC,i)/float(count_list[j])
    return cPT

def getValue(value,theData,varC,i):
    count  = 0
    for j in range(theData.shape[0]):
        if(theData[j][0] == value):
            if theData[j][varC]==i:
                count += 1
    return count
    # end of coursework 1 task 2
# ----------------------------------------------------------------------------------------
# Function to calculate the joint probability table of two variables in the data set

def JPT1(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )

#Coursework 1 task 3 should be inserted here
    number_elem  = theData.shape[0]
    for i in range(noStates[varRow]):
        for j in range(noStates[varCol]):
            jPT[i][j] = getValue(j,theData,varRow,i)/float(number_elem)
# end of coursework 1 task 3
    return jPT
def JPT(theData, varRow, varCol, noStates):
     jPT = zeros((noStates[varRow], noStates[varCol]), float )
     for data in theData:
         row = data[varRow]                   # the state value indicates its index in the
         col = data[varCol]                   # probability matrix
         jPT[row][col] += (1.0/len(theData))
     return jPT

# -------------------------------------------------------------------------
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here
    for index, element in enumerate(aJPT):
        for n in range(len(prior)):
            aJPT[index][n]=element[n]/prior[n]
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
        for m in range(rootPdf.size):
            rootPdf[m] = rootPdf[m]*cPTs[index][element][m]
    rootPdf = multiply(rootPdf,prior)

    alpha = 1/sum(rootPdf)
    for n in range(rootPdf.size):
        rootPdf[n] = rootPdf[n]*alpha

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
    sum_col_array = sum(jP,axis= 0)  # this function used for summing of each col
    for index_row,row in enumerate(jP):
        for index_col in range(jP.shape[1]):
            if(jP[index_row][index_col]==0):    # skip the element that has zero value
                continue                        # in order to skip calculating with log function
            left_side = jP[index_row][index_col]
            right_side = math.log(left_side/(row.sum()*sum_col_array[index_col]),2)
            mi = mi + (left_side*right_side)
# end of coursework 2 task 1
    return mi
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)

var_c = JPT(theData,2,8,noStates)
print(MutualInformation(var_c))


# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    for index_i in range(noVariables):
        for index_j in range(noVariables):
            temp_JPT = JPT(theData,index_i,index_j,noStates)
            MIMatrix[index_i][index_j] = MutualInformation(temp_JPT)

# end of coursework 2 task 2
    return MIMatrix

# Function to compute an ordered list of dependencies

def DependencyList(depMatrix):
    depMatrix = depMatrix.tolist()
    depList=[]
    temp_depList = []
# Coursework 2 task 3 should be inserted here
    for index,row in enumerate(depMatrix):
        for index_in_row in range(index+1,len(row)):
            temp_depList = []   # create temporary array
            temp_depList.append(depMatrix[index][index_in_row]) # append dependency value to first index of temp array
            temp_depList.append(index)            # append vertices from Matrix row
            temp_depList.append(index_in_row)     # append vertices from Matrix column
            depList.append(temp_depList)    # append temporary list to main list
    depList.sort(key=lambda x: x[0], reverse = True) # sort from highest to lowest dependency value

# end of coursework 2 task 3
    return  depList
    #return array(depList2)

def findParent(connections, i): #find the root node of the connection
	if connections[i] == -1:
		return i
	if connections[i] != -1:
		return findParent(connections, connections[i])

def union(connections, Node1, Node2): #connect the 2 nodes if it is not in a cycle
	Root = findParent(connections, Node1)
	Child = findParent(connections, Node2)
	connections[Root] = Child

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    connections = [-1] * noVariables #the parent of variable [index] is stored in this array
                                            #initally, starts off with -1 indicating not connected
    edges = []
    for element in depList:
        edges.append([element[1], element[2]])

    for edge in edges:
    	Node1 = edge[0]
        Node2 = edge[1]
        Node1 = int(Node1)
        Node2 = int(Node2)
        x = (findParent(connections, Node1)) #find root node
        y = (findParent(connections, Node2)) #find root node
        #print(x, "and" ,  y)
        if(x != y): #if both root nodes are not equal and therefore not a cycle
            union(connections, Node1, Node2) # connect the 2 nodes in connections
            #print(connections)
            spanningTree.append(edge) #append edges to spanningTree
    return array(spanningTree)

#

# End of coursework 2
#


noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)

var_c = JPT(theData,0,2,noStates)
MutualInformation(var_c)
# ----------------------- 1 ----------------------------
AppendString("DAPIResults02.txt","Coursework Two Results by ")
AppendString("DAPIResults02.txt","Nattapat Juthaprachakul(nj2217) and Weixiong Tay(wt814)")
AppendString("DAPIResults02.txt","")

# ------------------------- 2 -------------------------
dep_matrix = DependencyMatrix(theData,noVariables,noStates)
AppendString("DAPIResults02.txt","The DependencyMatrix")
AppendArray("DAPIResults02.txt", dep_matrix)

# ------------------------- 3 -------------------------
dep_list = DependencyList(dep_matrix)
AppendString("DAPIResults02.txt","The DependencyList")
AppendArray("DAPIResults02.txt", asarray(dep_list))

# ------------------------- 4 -----------------------------
spanning_tree = SpanningTreeAlgorithm(dep_list,noVariables)
AppendString("DAPIResults02.txt","SpanningTreeAlgorithm")
AppendArray("DAPIResults02.txt", spanning_tree)
