## Step by step on how to train, test and demo for KITTI and DETRAC Dataset

### Step 1) Setup Conda Environment
- Open Conda console, navigate to repository root folder
* Enter this command to create environment
   * **conda env create -f environment.yaml**
* Activate newly created environment
    * **conda activate trtmot**

### Step 2) Download DarkNet-53 model
* DarkNet-53 ImageNet pre-trained model: [Link](https://pjreddie.com/media/files/darknet53.conv.74)
* Save DarkNet model to ~/Towards-Realtime-MOT/weights/ folder

### Step 3) Download DETRAC and/or KITTI dataset
* For DETRAC Dataset:
    * Go to to http://detrac-db.rit.albany.edu/download
    * Download [Train Images](http://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip), [Test Images](http://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip), [Train Annotations XML](http://detrac-db.rit.albany.edu/Data/DETRAC-Train-Annotations-XML-v3.zip)
    * Create folders in the following structure
    * Folder structure:
        * ~/Towards-Realtime-MOT/DETRAC
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----images
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----MVI_20011
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img00001.jpg
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---...
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img0000N.jpg
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----MVI_....
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----MVI_0000N
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img00001.jpg
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---...
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img0000N.jpg
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----labels_with_ids
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----MVI_20011
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img00001.txt
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---...
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img0000N.txt
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----MVI_....
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----MVI_0000N
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img00001.txt
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---...
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img0000N.txt
    * Unzip the downloaded Train Images to the /DETRAC/images folder
    * Run the **JDE_DETRAC_Dataset_Formatter.ipynb** notebook to generate the labels, copy all generated **MVI_XXXXXX** folders to the /DETRAC/labels_with_ids folder
    * **[NOTE] The following videos have some missing frame ID in the annotation xml files, either remove the video images and lables before training or filter out the images**
        * MVI_39761
            * 542 - 610
            * 897 - 1080
            * 1297 - 1380
        * MVI_39781
            * 1862 - 1865
        * MVI_39811
            * 207 - 260
            * 342 - 510
            * 597 - 655
            * 787 - 1070
        * MVI_39851
            * 162 - 295
        * MVI_39931
            * 677 - 835
            * 977 - 1005
        * MVI_40152
            * 1607 - 1610
        * MVI_40162
            * 322 - 360
        * MVI_40211
            * 222 - 235
        * MVI_40213
            * 1167 - 1170
            * 1287 - 1290
        * MVI_40991
            * 272 - 335
            * 1567 - 1655
        * MVI_40992
        * MVI_63544
            * 892 - 1095

* For KITTI Dataset:
    * Go to http://www.cvlibs.net/datasets/kitti/eval_tracking.php
    * Download [Train Images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip), [Training Label](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip)
    * Create folders in the following structure
    * Folder structure:
    * ~/Towards-Realtime-MOT/KITTI
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----images
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----0000
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img00001.jpg
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---...
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img0000N.jpg
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----....
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----00NN
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img00001.jpg
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---...
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img0000N.jpg
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----labels_with_ids
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----0000
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img00001.txt
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---...
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img0000N.txt
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----....
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----00NN
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img00001.txt
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---...
        * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----img0000N.txt
    * Unzip the downloaded Train Images to the /KITTI/images folder
    * Run the **JDE_KITTI_Dataset_Formatter.ipynb** notebook to generate the labels, copy all generated **00NN** folders to the /KITTI/labels_with_ids folder

### Step 4) Generate training and validation manifest
* Run the **generate_train_labels.ipynb** notebook to generate training manifest from all the images file names in either the KITTI or DETRAC folders.
* To create the validation manifest, duplicate the generated training manifest and rename the extension to .val from .train.
* Open the .train manifest and delete all entries allocated for training and open the .val manifest and delete all entries allocated for validation.
* Save both .train and .val manifest to the ~/Towards-Realtime-MOT/data/ folders

### Step 5) Update ccmcpe.json in ~/Towards-Realtime-MOT/cfg/ folder
* Change "root" to the repository root folder (i.e. /home/<USERNAME>/Towards-Realtime-MOT)
* Clear all entries under "train", "test_emb" and "test" section
* Enter either DETRAC or KITTI manifest location in their respective "train" and "test" section, example as follows,
    * For DETRACT:
        * "train": { "detrac": "./data/detrac.train" }
        * "test": { "detrac": "./data/detrac.val" }
    * For KITTI:
        * "train": { "kitti": "./data/kitti.train" }
        * "test": { "kitti": "./data/kitti.val" }

### Step 6) Run training
* Run following command:
    * For DETRAC:
        * **CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg cfg/yolov3_864x480.cfg --batch-size=32 --epochs=30**
    * For KITTI:
        * **CUDA_VISIBLE_DEVICES=0 python3 train.py --cfg cfg/yolov3_1088x608.cfg --batch-size=32 --epochs=30**
    * Trained weights are saved in **/weights/latest.pt**

### Step 7) Generate demo
* Run following command: (Set cfg file accordingly and replace <VIDEO_PATH> with test video location)
    * **CUDA_VISIBLE_DEVICES=0 python3 demo.py --cfg cfg/yolov3_865x480.cfg --weights weights/latest.pt --input-video <VIDEO_PATH> --output-format video**
* Inferred video, frames and detection list will be saved in **results** folder