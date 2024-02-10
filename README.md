The key training metric used in the code is loss. The loss is calculated during each training iteration and is used to optimize the model parameters. 
Specifically, the binary cross-entropy loss (BCELoss) is computed between the model's output and the ground truth binary masks.
BCELoss is one of the most widely used loss for image segmentation , thats why I chose that as the loss.
There are instances where it couldnt detect the shape if the colour difference between background and shape is low.
For fixing this , I think I will have to reconsider the learning rate.
