# DA6401 Assignment-2 Skeleton Guide

This repository is an instructional skeleton for building the complete visual perception pipeline on Oxford-IIIT Pet.


### ADDITIONAL INSTRUCTIONS FOR ASSIGNMENT2:
- Ensure VGG11 is implemented according to the official paper(https://arxiv.org/abs/1409.1556). The only difference being injecting BatchNorm and CustomDropout layers is your design choice.
- Train all the networks on normalized images as input (as the test set given by autograder will be normalized images).
- The output of Localization model = [x_center, y_center, width, height] all these numbers are with respect to image coordinates, in pixel space (not normalized)
- Train the object localization network with the following loss function: MSE + custom_IOU_loss.
- Make sure the custom_IOU loss is in range: [0,1]
- In the custom IOU loss, you have to implement all the two reduction types: ["mean", "sum"] and the default reduction type should be "mean". You may include any other reduction type as well, which will help your network learn better.
- multitask.py shd load the saved checkpoints (classifier.pth, localizer.pth, unet.pth), initialize the shared backbone and heads with these trained weights and do prediction.
- Keep paths as relative paths for loading in multitask.py
- Assume input image size is fixed according to vgg11 paper(can be hardcoded need not pass as args)
- Stick to the arguments of the functions and classes given in the github repo, if you include any additional arguments make sure they always have some default value.
- Do not import any other python packages apart from the ones mentioned in assignment pdf, if you do so the autograder will instantly crash and your submission will not be evaluated.
- The following classes will be used by autograder: 
    ```
        from models.vgg11 import VGG11
        from models.layers import CustomDropout
        from losses.iou_loss import IoULoss
        from multitask import MultiTaskPerceptionModel
    ```
- The submission link for this assignment will be available by Saturday(04/04/2026) on gradescope





### GENERAL INSTRUCTIONS:
- From this assignment onwards, if we find any wandb report which is private/inaccessible while grading, there wont be any second chance, that submission will be marked 0 for wandb marks.
- The entireity of plots presented in the wandb report should be interactive and logged in the wandb project. Any screenshot or images of plots will straightly be marked 0 for that question.
- Gradescope offers an option to activate whichever submission you want to, and that submission will be used for evaluation. Under any circumstances, no requests to be raised to TAs to activate any of your prior submissions. It is the student's responsibility to do so(if required) before submission deadline.
- Assignment2 discussion forum has been opened on moodle for any doubt clarification/discussion.   




## Contact

For questions or issues, please contact the teaching staff or post on the course forum.

---

Good luck with your implementation!
