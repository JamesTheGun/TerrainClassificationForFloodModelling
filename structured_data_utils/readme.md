Utils for unet/yolo
Idea is that we can structure the data such that it looks like an image, with a channel for depths
This is different to regular las data where points are just in random places
Plan is that we coerce them into a set of possible positions depending upon the granularity we need for the problem
