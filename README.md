# Object-Detection-for-security-purposes
This program will detect and classify objects in the frame of the camera system and will classify the object as attended or abandoned. If it is abandoned, then it sends an email to alarm the authorities

It uses mobilenet_ssd as the object detection algorithm. Currently it will detect static as well as dynamic objects. The object detection is limited to people and their luggage as those are our points of interest from a security standpoint.

It will detect people and their luggage, classify their luggage as abandoned or attended, and if the luggage is abandoned it will notify the authorities via email.
