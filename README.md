# Naive Bayes Texture Classification
A simple image processing / ML project to classify image patches as one of 16 textures.

1. The image is subdivided into the true textures
2. Each texture is subdivided further into [16x16] blocks (64 per texture)
3. 10 Features are extracted for each block
4. The feature mean and covariance matrix for each texture are computed
5. All blocks in the image (64 x 4 x 4 = 1024) are classified using Bayes rule
6. The classification results are overlayed with colour on the original image
