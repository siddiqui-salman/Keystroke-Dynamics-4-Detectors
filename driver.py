import pandas as pd
import statistics
import ManhattanDetector
import EuclideanDetector
import EuclideanDetectorNorm
import ManhattanDetectorScaled

df = pd.read_excel('DSL-StrongPasswordData.xls')
users = df['subject'].unique()
sample_size = int(input('Enter the size of the sample: '))
n = [0, 2 , 4]
color = ['r','b','black']

print()
print("-----------------------------------------------------------------------------------------")
print("These are the detectors implemented \n"
      "1. Manhattan Detector\n"
      "2. Manhattan Detector (Scaled)\n"
      "3. Euclidean Detector\n"
      "4. Euclidean Detector (Normalized)")
userChoice= int(input("Enter your choice (1-4): "))
if userChoice == 1:
    fpr,ipr,tpr,threshold,eer=ManhattanDetector.manhattandetector(df,sample_size,users)
    ManhattanDetector.visualdet(fpr, ipr, n, threshold, sample_size, color)
    ManhattanDetector.visualroc(fpr, tpr, n, color, sample_size)
elif userChoice == 2:
    fpr,ipr,tpr,threshold,eer=ManhattanDetectorScaled.manhattandetectorscaled(df,sample_size,users)
    ManhattanDetectorScaled.visualdet(fpr, ipr, n, threshold, sample_size, color)
    ManhattanDetectorScaled.visualroc(fpr, tpr, n, color, sample_size)
elif userChoice ==3:
    fpr,ipr,tpr,threshold,eer = EuclideanDetector.euclideandetector(df,sample_size,users)
    EuclideanDetector.visualdet(fpr, ipr, n, threshold, sample_size, color)
    EuclideanDetector.visualroc(fpr, tpr, n, color, sample_size)
elif userChoice==4:
    fpr, ipr, tpr, threshold, eer = EuclideanDetectorNorm.euclideandetectornorm(df, sample_size, users)
    EuclideanDetectorNorm.visualdet(fpr, ipr, n, threshold, sample_size, color)
    EuclideanDetectorNorm.visualroc(fpr, tpr, n, color, sample_size)
else:
    print("Invalid Choice")

print()
print("---------------------------------------------------------------------------------------")


if userChoice in [1,2,3,4]:
    if userChoice==1:
        print("MANHATTAN DETECTOR")
    elif userChoice==2:
        print("MANHATTAN DETECTOR (SCALED)")
    elif userChoice==3:
        print("EUCLIDEAN DETECTOR")
    elif userChoice==4:
        print("EUCLIDEAN DETECTOR (NORMALIZED)")

    print("Equal error rate was (N=" + "{}".format(sample_size) + ") " + "{:.3f}".format(sum(eer)/len(eer)))
    print("With a standard deviation (N="+"{}".format(sample_size) + ") of " + "{:.3f}".format(statistics.stdev(eer)))

    print("------------------------------------------------------------------------------------")
    for i in n:
        print("Equal error rate for user " + "{}".format(i+1) + " was " + "{:.3f}".format(eer[i]))


