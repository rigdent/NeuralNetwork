Optical character recognition through backpropagation of errors in a neural network.
By Tenzin Rigden and Christopher Gorman Winter

Sources We Used:

The algorithm is taken from "Artificial Intelligence A modern Approach" by Peter Norvig and Stuart Jonathan Russell on page 734.

We use a data set from here:
http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

The data set contain around 1000 examples of handwritten numbers ranging from 0 to 9 converted to a 32 by 32 bitmap. We resized the bitmap to make it 16 by 16 to speed up our program due to fears of the size of the bitmap slowing our program (32 by 32 would be 1024 input nodes).

How To Run the Program:

Running the program without changing anything will build the outline of a neural network, train that network on the examples, and then test using k-fold cross validation. The default is to use a test set of 30 and then train with all of the other examples. After running, it will tell you the percentage of tests that we successful (across all of the validation runs). The number taken out for each test can be changed. On the lab computers, under pypy, this takes around 10 minutes and yields an accuracy rate of 89%.

We also included a leave-one-out cross validation function, which will again run and print out the percentage of successes. However, this will take a very long time, and so we do not recommend it. When we ran this function, each validation run used 100 iterations, and we were using only the first 101 examples. This had an accuracy rate of 75%, but took several hours to run. Using the test on the mean squared sum of the errors that we implemented later might speed this up a little, but would still likely take a long time. It is also possible that our accuracy rate was actually larger, but we interpreted an output vector that had no values greater than 0.9 to be a failure, even if one of the components was much larger than the others. Looking simply for the biggest component might increase our accuracy rate.

Running in pypy will significantly speed up the run time.

How Our Program Works:

Our program first constructs the neural network.

The algorithm that trains the network is based on the algorithm given in the book. We set alpha to decay to 0.999 times its previous value every time the weights are updated. We found that a decay rate any smaller (such as 0.99) caused our accuracy to suffer.

We included a test that stops the iterations if the mean squared sum of the error drops below a certain threshold. This significantly reduces the number of iterations that we do, at a small cost in accuracy (at the default settings for k-fold cross validation our accuracy rate dropped from ~90% to ~89%). The time speed up is likely worth it, especially at the larger data set sizes. If the network undergoes 100 iterations but the mean squared sum hasn't fallen below this threshold (set to 100 iterations by default), the iterations are also stopped. In our testing, the mean squared sum cutoff occured at about 10-15 iterations, so well below this limit.

In particular, we determined the cutoff by examining the mean squared sum of the errors after 100 iterations. This indicates that the error is reaching a value after between 10 and 15 iterations, and it is at that point so small that it isn't changing significantly for the next 85-90 iterations. This is where the significant cost savings come in.

We adjusted the original data set (nnData.txt) slightly. It was originally a 32 by 32 bitmap, which is 1024 inputs. This takes a long time, we we averaged groups of 4 pixels to shrink it down to 256 inputs, which is more reasonable (smallData.txt).

We started with 96 hidden nodes based examples we found of neural networks being used for OCR. However, 96 hidden nodes took a long time to run, and switching to 50 maintained a high level of accuracy while speeding up the training dramatically.


What We Included:

-This README
-our program, which is called network.py
-an earlier attempt, which is called backPropLearning.py [THIS IS NOT WHAT YOU SHOULD RUN]
-the original data set (nnData.txt)
-the smaller data set (smallData.txt)
-a test piece of data (testData.txt)
