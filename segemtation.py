import cv2
import numpy as np


img = cv2.imread('Assets/img2.jpg')
img = cv2.resize(img , (540,540))

#-------Define the Useful parameters--------#
iter = 5
thresh = 0.5
conf = 0.5

# -------Loading our model files--------#
classes_ = 'Model/object_detection_classes_coco.txt'
LABELS = open(classes_).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weights = 'Model/frozen_inference_graph.pb'
config = 'Model/mask_rcnn_inception_v2_coco.pbtxt'
    # instantiating our model
net = cv2.dnn.readNetFromTensorflow(weights, config)

# creating blob and setting it as input
blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)
net.setInput(blob)

# creating predictions -> Mask R-CNN
(boxes, masks) = net.forward(["detection_out_final",
	"detection_masks"])

# loop over the number of detected objects
for i in range(0, boxes.shape[2]):
	classID = int(boxes[0, 0, i, 1])
	confidence = boxes[0, 0, i, 2]
	mask = masks[i,classID]
    # Check for minimum confidence threshold for detecting an object
	if confidence > conf:
        # extracting the image dimensions
		(H, W) = img.shape[:2]
        # converting the bounding box to match the image dimensions
		box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
         
		(startX, startY, endX, endY) = box.astype("int")
		boxW = endX - startX
		boxH = endY - startY  

		# resize the mask and store create a separate mask using the box dimensions
		mask = cv2.resize(mask,(boxW,boxH) , interpolation=cv2.INTER_CUBIC)
		mask = (mask > thresh).astype("uint8") * 255
		rcnnMask = np.zeros(img.shape[:2], dtype="uint8")
		rcnnMask[startY:endY, startX:endX] = mask

		# # Uncomment the following block of code to also show the mask Rnn without 
		# # the grab-cut method
		# # to apply the mask to image -> use the bitwise_and
		# rcnnOutput = cv2.bitwise_and(img, img, mask=rcnnMask)

		# # show the output of the Mask R-CNN and bitwise AND operation
		# mask_ = cv2.hconcat([img,rcnnOutput])
		
		# cv2.imshow("R-CNN Mask without Grab-cut", mask_)


		gcMask = rcnnMask.copy()
		gcMask[gcMask > 0] = cv2.GC_PR_FGD
		gcMask[gcMask == 0] = cv2.GC_BGD
		# allocate memory for two arrays that the GrabCut algorithm
		# internally uses when segmenting the foreground from the
		# background and then apply GrabCut using the mask
		# segmentation method
		print("[INFO] applying GrabCut to '{}' ROI...".format(
			LABELS[classID]))
		fgModel = np.zeros((1, 65), dtype="float")
		bgModel = np.zeros((1, 65), dtype="float")
		(gcMask, bgModel, fgModel) = cv2.grabCut(img, gcMask,
			None, bgModel, fgModel, iterCount=iter,
			mode=cv2.GC_INIT_WITH_MASK)
		# set all definite background and probable background pixels
		# to 0 while definite foreground and probable foreground
		# pixels are set to 1, then scale the mask from the range
		# [0, 1] to [0, 255]
		outputMask = np.where(
			(gcMask == cv2.GC_BGD) | (gcMask == cv2.GC_PR_BGD), 0, 1)
		outputMask = (outputMask * 255).astype("uint8")
		# apply a bitwise AND to the image using our mask generated
		# by GrabCut to generate our final output image
		output = cv2.bitwise_and(img, img, mask=outputMask)
		# show the output GrabCut mask as well as the output of
		# applying the GrabCut mask to the original input image
		g_mask = cv2.hconcat([img,output])
		cv2.putText(g_mask , f'{LABELS[classID]}',(40,40),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2,(0,255,255),2)
		cv2.imshow('', g_mask)
		# cv2.imshow("GrabCut Mask", outputMask)
		# cv2.imshow("Output", output)
		cv2.waitKey(0)

