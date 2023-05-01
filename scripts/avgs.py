import numpy as np

# results through 10 epoch chain thaw finetuning on SS-YouTube dataset

times_128 = [ 2256.3748264312744, 2660.2772188186646, 
             3151.567939043045, 1194.9679341316223, 
             1246.0099127292633 ]

accuracies_128 = [ 0.8076628099173554, 0.8141421487603304,
                  0.8278214876033058, 0.8140628099173555, 
                  0.8125950413223141 ]

print("128 average time", np.mean(times_128))
print("128 average accuracy", np.mean(accuracies_128))

times_256 = [ 1114.1578307151794, 1053.1693451404572, 
             1031.6299607753754, 1105.7955691814423, 
             1126.968327999115 ]

accuracies_256 = [ 0.7119338842975207, 0.7102280991735537, 
                  0.7093487603305784, 0.700601652892562, 
                  0.7198545454545455 ]

print("256 average time", np.mean(times_256))
print("256 average accuracy", np.mean(accuracies_256))

# the original layer count - reproducing paper results
times_512 = [ 1793.095605134964, 1874.8625202178955, 
             1955.3379085063934, 2060.577109336853, 
             2112.0171954631805 ]

accuracies_512 = [ 0.9220297520661156, 0.9228297520661156, 
                  0.9228561983471074, 0.9212297520661157,
                  0.9204297520661158 ]

print("512 average time", np.mean(times_512))
print("512 average accuracy", np.mean(accuracies_512))

times_1024 = [ 3407.019469976425 ]

accuracies_1024 = [ 0.8157421487603305 ]

print("1024 average time", np.mean(times_1024))
print("1024 average accuracy", np.mean(accuracies_1024))