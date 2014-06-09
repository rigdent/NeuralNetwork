Optical character recognition through backpropagation of errors in a neural network.
By Tenzin Rigden and Christopher Gorman Winter

The algorithm is taken from "Artificial Intelligence A modern Approach" by Peter Norvig and Stuart Jonathan Russell on page 734.

We use a data set from here, http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits.

The data set contain around 1000 examples of handwritten numbers ranging from 0 to 9 converted to a 32 by 32 bitmap. We resized the bitmap to make it 16 by 16 to speed up our program due to fears of the size of the bitmap slowing our program (32 by 32 would be 1024 input nodes).

By running the program with no arguments, it will create the neural network and then run some tests.

If it is run with an argument containing a bitmap of a number, it will return what it believes the number is. The bitmap must be a txt file that is 16 characters by 16 characters containing only 1's and 0's.


We tested our algorithm using leave-one-out cross validation, with the first 101 examples in the examplesList (the first 101 examples from smallData.txt). Having the neural network iterate 100 times for each test yields a 75% correct identification rate. This number is probably lower than the actual number the network could correctly identify. First, more iterations would probably lead to a more accurate network. Second, we use a sigmoid function to set the output of a node. If the sigmoid function returns greater than 0.9, we set the output to 1, and if it returns less than 0.1 we set the output to 0. If the sigmoid function returns something in between, we set the output to be whatever was returned. However, if, when testing an example, we don't encounter a 1 anywhere in the vector (because the output of all the nodes was always less than 0.9), we treat it as a failure. However, some of the nodes had all 0s except for a value in the output vector that was greater than 0.1 but less than 0.9. If we took the largest value in the output vector to be the answer, instead of looking for a 1, our success rate would likely increase. Also, this method took a very long time, we waited several hours for it to finish. Note that this test was run before we changed our criteria for stopping, and so ran for 100 iterations every time.
