Supplementary data for the paper 
De Winter, J., Hoogmoed, J., Stapel, J., Dodou, D., & Bazilinskyy, P. (2023). Predicting perceived risk of traffic scenes using computer vision. Transportation Research Part F: Psychology and Behaviour. https://doi.org/10.1016/j.trf.2023.01.014

* read_and_process.m      	MATLAB script that produces all figures and analyses for the paper
* data.mat	 		     	.mat file with raw data of the experiment per participant. The Worker ID and IP address have been masked
* yoloresults.mat		     	.mat file with outcomes of the YOLOv4 algorithm (bounding boxes [x,y,w,h], classification scores, and labels)
* ini.m			     	MATLAB script with vehicle speed per image (m/s), road type (1 = City, 2 = Residential, 3 = Road) per image, and class label names
* Overview_of_images.xlsx    	Excel file with information about the images
* survey.pdf 		     	Preview of the Appen survey
* \images 			     	Folder with the images that were used in the experiment. The images were taken from the KITTI dataset (Geiger et al., 2013)

Note
* The MATLAB scripts were tested using MATLAB R2022b
* The images from the KITTI dataset are copyrighted by the original authors (Geiger et al., 2013) and published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License

References
* Geiger, A., Lenz, P., Stiller, C., & Urtasun, R. (2013). Vision meets robotics: The KITTI dataset. The International Journal of Robotics Research, 32, 1231–1237. https://doi.org/10.1177/0278364913491297