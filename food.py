from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
import cv2
import numpy as np
import os

def getData(directory,target):
  imarray = [];
  tararray = [];
  for subdir, dirs, files in os.walk(directory):
    for f in files:
      path = os.path.join(subdir, f)
      image = cv2.imread(path)
      if image != None and image.shape[:2] != (0,0):
        image = cv2.resize(image, (200,200));
        imarray.append(np.ravel(image))
        tararray.append(target)
  return (imarray,tararray)


good,targood = getData("food",[1])
bad,tarbad = getData("nonfoods",[0])

DS = ClassificationDataSet(200*200*3, 1, 2,class_labels=['food', 'nonfood'])
DS.setField('input',good + bad)
DS.setField('target',targood + tarbad)

tstdata, trndata = DS.splitWithProportion( 0.50 )
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0] 

if os.path.isfile('food.xml'): 
  print "previous xml found:" 
  fnn = NetworkReader.readFrom('food.xml') 
else:
  fnn = buildNetwork( trndata.indim, 64 , trndata.outdim, outclass=SoftmaxLayer )

trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
trainer.trainEpoch(50)
print 'Percent Error on Test dataset: ' , percentError( trainer.testOnClassData (
             dataset=tstdata )
                        , tstdata['class'] )
NetworkWriter.writeToFile(fnn, 'food.xml')
# ticks = arange(-3.,6.,0.2)
# X, Y = meshgrid(ticks, ticks)
# # need column vectors in dataset, not arrays
# griddata = ClassificationDataSet(2,1, nb_classes=3)
# for i in xrange(X.size):
#     griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
# griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy
# 
# for i in range(20):
#     trainer.train()
#     trnresult = percentError( trainer.testOnClassData(), 
#                               trndata['class'] )
#     tstresult = percentError( trainer.testOnClassData( 
#            dataset=tstdata ), tstdata['class'] )
# 
#     print "epoch: %4d" % trainer.totalepochs, \
#           "  train error: %5.2f%%" % trnresult, \
#           "  test error: %5.2f%%" % tstresult
# 
#     out = fnn.activateOnDataset(griddata)
#     out = out.argmax(axis=1)  # the highest output activation gives the class
#     out = out.reshape(X.shape) 
#     
# 
#     figure(1)
#     ioff()  # interactive graphics off
#     clf()   # clear the plot
#     hold(True) # overplot on
#     for c in [0,1,2]:
#         here, _ = where(tstdata['class']==c)
#         plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
#     if out.max()!=out.min():  # safety check against flat field
#         contourf(X, Y, out)   # plot the contour
#     ion()   # interactive graphics on
#     draw()  # update the plot
#     
# ioff()
# show()  
