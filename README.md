![](Web_app/find_your_art.png)
Discover artwork around you using transfer learning

Find Your Art is a machine learning tool that classifies art into 25 different styles:

* Abstract Art
* Abstract Expressionism
* Art Informel
* Art Nouveau (Modern)
* Baroque
* Color Field Painting
* Cubism
* Early Renaissance
* Expressionism
* High Renaissance
* Impressionism
* Magic Realism
* Mannerism (Late Renaissance)
* Minimalism
* Naive Art (Primitivism)
* Neoclassicism
* Northern Renaissnce
* Pop Art
* Post-Impressionism
* Realism
* Rococo
* Romanticism
* Surrealism
* Symbolism
* Ukiyo-e


After scrapping / cleaning images and metadata from (https://www.wikiart.org/), a transfer learning approach was taken. Bottleneck features from the VGG16 convolutional neural network (CNN; pre-trained on the imagenet dataset) were extracted for each image, and a customized top classifier CNN was trained on the new labels. 

The web app allows users to upload an image of a piece of art they like, and get the style of that image from the CNN running in the backend, as well as a recommendation of other artwork they might like.   It is hosted at http://find-your-art.us/

## Acknowledgments

** https://github.com/lucasdavid/wikiart  
** https://github.com/chasingbob/keras-visuals
