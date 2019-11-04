import csv
import numpy as np
import pandas as pd
import math
import sys

class Tree(object):
    def __init__(self):
      self.left = None
      self.right = None
      self.val = None
      self.attr = None
      self.common = None

def preProcessCsv(filename, ratio, first):
  line = pd.read_csv(filename, engine='python',header=None)
  dataset = line.values

  if first:
    num = int(round(len(dataset) * ratio / 100.0))
    dataset = dataset[0:num]
  else:
    num = int(round(len(dataset) * (100 - ratio) / 100.0))
    dataset = dataset[num:len(dataset)]
  dataT = dataset.T[:-1]

  #collect unique values in each attribute
  #update the data with binary attribute
  attrSet = []
  for attr in dataT:
    tempAttr = []
    for val in attr:
      if val not in tempAttr:
        tempAttr.append(val)
    attrSet.append(tempAttr)

  df = pd.DataFrame()

  index = 0
  for i in range(len(attrSet)):
    for val in attrSet[i]:
      newCol = []
      for row in dataset:
        if row[i] == val:
          newCol.append(1)
        else:
          newCol.append(0)
      df[val] = newCol

  df["class"] = dataset.T[-1]

  return df

def getEntropy (dataset, attr):
  pOne = 0.0
  pZero= 0.0

  oneSet = dataset[dataset[attr] == 1]
  zeroSet = dataset[dataset[attr] == 0]

  pOne = len(oneSet.index)/len(dataset.index)
  pZero = len(zeroSet.index)/len(dataset.index)

  entropyOne = 0.0
  if pOne != 0:
    pLess = 0.0
    for row in oneSet.values:
      if row[-1] == ' <=50K':
        pLess += 1
    pLess = pLess / len(oneSet.index)
    pGreater = 1 - pLess
    if (pLess == 1) or (pLess == 0):
      entropyOne = 0.0
    else:
      entropyOne = -pOne * (-pLess * math.log(pLess) - pGreater * math.log(pGreater))

  entropyZero = 0.0
  if pZero != 0:
    pLess = 0.0
    for row in zeroSet.values:
      if row[-1] == ' <=50K':
        pLess += 1
    pLess = pLess / len(zeroSet.index)
    pGreater = 1 - pLess
    if (pLess == 1) or (pLess == 0):
      entropyZero = 0.0
    else:
      entropyZero = -pZero * (-pLess * math.log(pLess) - pGreater * math.log(pGreater))

  return entropyZero + entropyOne


def infoGain(dataset):
  numLess = 0.0

  for row in dataset.values:
    if row[-1] == ' <=50K':
      numLess = numLess + 1
  
  pLess = numLess / len(dataset.index)
  pGreater = 1- pLess

  if (pLess == 1) or (pLess == 0):
    entropyClass = 0.0
  else:
    entropyClass = -pLess*math.log(pLess)-pGreater*math.log(pGreater)

  gainDict = {}

  for header in list(dataset):
    if header == 'class':
      continue
    gain = entropyClass + getEntropy(dataset, header)

    gainDict[header] = gain

  bestAttr = max(gainDict, key=gainDict.get)
  bestGain = gainDict[bestAttr]

  return bestAttr, pLess, bestGain


def id3(dataset):
  # check if all instance have the same class label
  # print(dataset.shape)

  if len(dataset['class'].unique()) == 1:
    root = Tree()
    root.val = dataset["class"].values[0]
    # print(f"root has value: {check}")
    return root

  #find the best attribute
  bestAttr, pLess, bestGain = infoGain(dataset)
  # print(bestGain)

  if pLess > 0.5:
    mostCommon = ' <=50K'
  else:
    mostCommon = ' >50K'

  if bestGain == 0:
    newNode = Tree()
    newNode.val = mostCommon
    return newNode
  
  oneSet = dataset[dataset[bestAttr] == 1]
  zeroSet = dataset[dataset[bestAttr] == 0]

  #remove the selected attribute
  oneSet = oneSet.drop(columns = bestAttr)
  zeroSet = zeroSet.drop(columns = bestAttr)

  newNode = Tree()
  newNode.attr = bestAttr
  newNode.common = mostCommon
  # print(f"root has name: {bestAttr}")

  if len(oneSet.index) == 0:
    newNode.right = Tree()
    newNode.right.val = mostCommon
    # print(f"left root has val: {mostCommon}")
  else:
    newNode.right = id3(oneSet)

  if len(zeroSet.index) == 0:
    newNode.left = Tree()
    newNode.left.val = mostCommon
    # print(f"right root has val: {mostCommon}")
  else:
    newNode.left = id3(zeroSet)

  return newNode

def id3Depth(dataset, depth):
  # check if all instance have the same class label
  # print(dataset.shape)

  if len(dataset['class'].unique()) == 1:
    root = Tree()
    root.val = dataset["class"].values[0]
    # print(f"root has value: {check}")
    return root

  #find the best attribute
  bestAttr, pLess, bestGain = infoGain(dataset)
  # print(bestGain)

  if pLess > 0.5:
    mostCommon = ' <=50K'
  else:
    mostCommon = ' >50K'

  if depth == 0:
    newNode = Tree()
    newNode.val = mostCommon
    return newNode

  if bestGain == 0:
    newNode = Tree()
    newNode.val = mostCommon
    return newNode
  
  oneSet = dataset[dataset[bestAttr] == 1]
  zeroSet = dataset[dataset[bestAttr] == 0]

  #remove the selected attribute
  oneSet = oneSet.drop(columns = bestAttr)
  zeroSet = zeroSet.drop(columns = bestAttr)

  newNode = Tree()
  newNode.attr = bestAttr
  newNode.common = mostCommon
  # print(f"root has name: {bestAttr}")

  if len(oneSet.index) == 0:
    newNode.right = Tree()
    newNode.right.val = mostCommon
    # print(f"left root has val: {mostCommon}")
  else:
    newNode.right = id3Depth(oneSet, depth - 1)

  if len(zeroSet.index) == 0:
    newNode.left = Tree()
    newNode.left.val = mostCommon
    # print(f"right root has val: {mostCommon}")
  else:
    newNode.left = id3Depth(zeroSet, depth - 1)

  return newNode


def searchTree(root, row, column):
  if root.val is not None:
    return root.val

  if root.attr not in column:
    if root.common == 1:
      return searchTree(root.right, row, column)
    else:
      return searchTree(root.left, row, column)

  if row[column.index(root.attr)] == 1:
    return searchTree(root.right, row, column)
  else:
    return searchTree(root.left, row, column)


def getAccuracy(dataset, root):
  matrix = dataset.values
  column = dataset.columns.tolist()
  accuracy = 0.0
  for row in matrix:
    label = searchTree(root, row, column)
    if label == row[-1]:
      accuracy += 1
  accuracy = accuracy / len(matrix)
  return accuracy


def postPruning(root, node, valiSet):

  if node.left is not None:
    if node.left.val is None:
      postPruning(root, node.left, valiSet)
  if node.right is not None:
    if node.right.val is None:
      postPruning(root, node.right, valiSet)

  oldAcc = getAccuracy(valiSet, root)
  node.val = node.common
  newAcc = getAccuracy(valiSet, root)

  if newAcc >= oldAcc:
    node.left = None
    node.right = None
  else:
    node.val = None
  
  return


if __name__ == "__main__":
  train = sys.argv[1]
  test = sys.argv[2]
  ratio = float(sys.argv[4])
  if sys.argv[3] == 'vanilla':
    trainSet = preProcessCsv(train, ratio, 1)
    testSet = preProcessCsv(test, 100, 1)
    root = id3(trainSet)

    trainAccuracy = getAccuracy(trainSet, root)
    print(f"Train set accuracy: {trainAccuracy}")

    testAccuracy = getAccuracy(testSet, root)
    print(f"Test set accuracy: {testAccuracy}")

  if sys.argv[3] == 'prune':
    trainSet = preProcessCsv(train, ratio, 1)
    ratioV = float(sys.argv[5])
    valiSet = preProcessCsv(train, ratioV, 0)
    testSet = preProcessCsv(test, 100, 1)

    root = root = id3(trainSet)
    postPruning(root, root, valiSet)

    trainAccuracy = getAccuracy(trainSet, root)
    print(f"Train set accuracy: {trainAccuracy}")

    testAccuracy = getAccuracy(testSet, root)
    print(f"Test set accuracy: {testAccuracy}")

  if sys.argv[3] == 'maxDepth':
    trainSet = preProcessCsv(train, ratio, 1)
    ratioV = float(sys.argv[5])
    valiSet = preProcessCsv(train, ratioV, 0)
    testSet = preProcessCsv(test, 100, 1)
    depth = float(sys.argv[6])

    root = id3Depth(trainSet, depth)

    trainAccuracy = getAccuracy(trainSet, root)
    print(f"Train set accuracy: {trainAccuracy}")

    testAccuracy = getAccuracy(testSet, root)
    print(f"Test set accuracy: {testAccuracy}\n")

    # Following part is for validation
    # the part above is just training with the given
    # depth then start testing.

    # accList = []
    # depthList = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # print("Start validation:")
    # for i in depthList:
    #   root = id3Depth(trainSet, i)
    #   print(f"Depth is {i}")
    #   trainAccuracy = getAccuracy(trainSet, root)
    #   print(f"Train set accuracy: {trainAccuracy}")

    #   valiAccuracy = getAccuracy(valiSet, root)
    #   print(f"Validation set accuracy: {valiAccuracy}\n")
    #   accList.append(valiAccuracy)

    # bestDepth = depthList[accList.index(max(accList))]
    # print(f"Best depth is {bestDepth}\n")

    # root = id3Depth(trainSet, bestDepth)
    # testAccuracy = getAccuracy(testSet, root)
    # print(f"Test set accuracy with best depth: {testAccuracy}\n")




  # following part is used only in part2 and it no longer works
  # trainSet = preProcessCsv(train, 100, 1)
  # nnode = Tree()
  # nnode.val = 0
  # root = id3(trainSet, nnode)
  # base = root.common
  # print("start")


  # for ratio in [40, 50, 60, 70, 80, 100]:
  #   nnode = Tree()
  #   nnode.val = 0
  #   trainSet = preProcessCsv(train, ratio, 1)
  #   valiSet = preProcessCsv(train, 20, 0)
  #   testSet = preProcessCsv(test, 100, 1)
  #   root = id3(trainSet, nnode)

  #   trainAccuracy = getAccuracy(trainSet, root)
  #   print(f"ogTrain set accuracy {ratio} : {trainAccuracy}")

  #   testAccuracy = getAccuracy(testSet, root)
  #   print(f"ogTest set accuracy {ratio} : {testAccuracy}")
  #   print(nnode.val)

  #   print("prnnn:")

  #   postPruning(root, root, valiSet, nnode)

  #   trainAccuracy = getAccuracy(trainSet, root)
  #   print(f"Train set accuracy {ratio} : {trainAccuracy}")

  #   testAccuracy = getAccuracy(testSet, root)
  #   print(f"Test set accuracy {ratio} : {testAccuracy}")

  #   baseAcc = 0.0
  #   for c in testSet['class']:
  #     if c == base:
  #       baseAcc+=1.0
    
  #   baseAcc = baseAcc/len(testSet.index)
  #   print(f"Base set accuracy {ratio} : {baseAcc}")
  #   print(nnode.val)
  #   print()
