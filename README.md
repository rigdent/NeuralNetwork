Optical character recognition through backpropagation of errors in a neural network.
By Tenzin Rigden and Christopher Gorman Winter

The algorithm is taken from "Artificial Intelligence A modern Approach" by Peter Norvig and Stuart Jonathan Russell on page 734.

We use a data set from here, http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits.

The data set contain around 1000 examples of handwritten numbers ranging from 0 to 9 converted to a 32 by 32 bitmap. We resized the bitmap to make it 16 by 16 to speed up our program due to fears of the size of the bitmap slowing our program.

By running the program with no arguments, it will create the neural network and then run some tests.

If it is run with an argument containing a bitmap of a number, it will return what it believes the number is. The bitmap must be a txt file that is 16 characters by 16 characters containing only 1's and 0's.
