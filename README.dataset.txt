# futbol players > 2024-11-16 12:11pm
https://universe.roboflow.com/ilyes-talbi-ptwsp/futbol-players

Provided by a Roboflow user
License: Public Domain

## Background Information
This dataset was curated and annotated by [Ilyes Talbi](https://www.linkedin.com/in/ilyes-talbi-ba2451135/), Head of [La revue IA](https://larevueia.fr/), a French publication focused on stories of machine learning applications.

Main objetive is to identify if soccer (futbol) players, the referree and the soccer ball (futbol).

The original custom dataset *(v1)* is composed of 163 images.
* Class 0 = players
* Class 1 = referree
* Class 2 = soccer ball (or futbol)

The dataset is available under the Public License.

## Getting Started
You can download this dataset for use within your own projects, or fork it into a workspace on Roboflow to create your own model.

## Dataset Versions
### Version 7 (v7) - 163 images (raw images)
* **Preprocessing:** Auto-Orient, Modify Classes: 3 remapped, 0 dropped
	* Modified Classes: *Class 0 = players, Class 1 = referree, Class 2 = futbol*
* **Augmentations:** *No augmentations applied*
* **Training Metrics:** *This version of the dataset was not trained*

### Version 2 (v2) - 163 images
* **Preprocessing:** Auto-Orient and Resize (Stretch to 416x416)
* **Augmentations:** *No augmentations applied*
* **Training Metrics:** *This version of the dataset was not trained*

### Version 3 (v3) - 391 images
* **Preprocessing:** Auto-Orient and Resize (Stretch to 416x416), Modify Classes: 3 remapped, 0 dropped
	* Modified Classes: *Class 0 = players, Class 1 = referree, Class 2 = futbol*
* **Augmentations:**
	* Outputs per training example: 3
	* Rotation: Between -25° and +25°
	* Shear: ±15° Horizontal, ±15° Vertical
	* Brightness: Between -25% and +25%
	* Blur: Up to 0.75px
	* Noise: Up to 1% of pixels
	* Bounding Box: Blur: Up to 0.5px
* **Training Metrics:** 86.4%mAP, 51.8% precision, 90.4% recall

### Version 4 (v4) - 391 images
* **Preprocessing:** Auto-Orient and Resize (Stretch to 416x416), Modify Classes: 3 remapped, 0 dropped
	* Modified Classes: *Class 0 = players, Class 1 = referree, Class 2 = futbol*
* **Augmentations:**
 	* Outputs per training example: 3
	* Rotation: Between -25° and +25°
	* Shear: ±15° Horizontal, ±15° Vertical
	* Brightness: Between -25% and +25%
	* Blur: Up to 0.75px
	* Noise: Up to 1% of pixels
	* Bounding Box: Blur: Up to 0.5px
* **Training Metrics:** 84.6% mAP, 52.3% precision, 85.3% recall

### Version 5 (v5) - 391 images
* **Preprocessing:** Auto-Orient and Resize (Stretch to 416x416), Modify Classes: 3 remapped, 2 dropped
	* Modified Classes: *Class 0 = players, Class 1 = referree, Class 2 = futbol*
		* Only *Class 0*, which was remapped to *players* was included in this version
* **Augmentations:**
 	* Outputs per training example: 3
	* Rotation: Between -25° and +25°
	* Shear: ±15° Horizontal, ±15° Vertical
	* Brightness: Between -25% and +25%
	* Blur: Up to 0.75px
	* Noise: Up to 1% of pixels
	* Bounding Box: Blur: Up to 0.5px
* **Training Metrics:** Trained from the COCO Checkpoint in Public Models ("[transfer learning](https://blog.roboflow.com/a-primer-on-transfer-learning/)") on Roboflow
	* 98.8%mAP, 76.3% precision, 99.2% recall

### Version 6 (v6) - 391 images
* **Preprocessing:** Auto-Orient and Resize (Stretch to 416x416), Modify Classes: 3 remapped, 2 dropped
	* Modified Classes: *Class 0 = players, Class 1 = referree, Class 2 = futbol*
		* Only *Class 0*, which was remapped to *players* was included in this version
* **Augmentations:**
 	* Outputs per training example: 3
	* Rotation: Between -25° and +25°
	* Shear: ±15° Horizontal, ±15° Vertical
	* Brightness: Between -25% and +25%
	* Blur: Up to 0.75px
	* Noise: Up to 1% of pixels
	* Bounding Box: Blur: Up to 0.5px
* **Training Metrics:** Trained from Scratch (*no transfer learning employed*)
	* 95.5%mAP, 67.8% precision, 95.5% recall

Ilyes Talbi - [LinkedIn](https://www.linkedin.com/in/ilyes-talbi-ba2451135/) | [La revue IA](https://larevueia.fr/)