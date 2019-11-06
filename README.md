# DecisionTree

11/05/2019 Update:
Usage:
python3 ID3.py "train file" "test file" [vanilla | prune | maxDepth] trainPercentage testPercentage depth

flags:
vanilla for tree without pruning
prune for tree with post pruning (reduced error pruning)
maxDepth for tree without a pruning but with limited number of depth

This program will build a binary tree, so it convert all attribute to a binary one by creating a new column
for every unique value in the original attribute then delete the original column.

This program will be optimize in the future
