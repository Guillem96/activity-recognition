#!/bin/bash

BASE_URL=https://www.crcv.ucf.edu/
DATASET_URL=${BASE_URL}datasets/human-actions/ucf101/UCF101.rar
ANNOTATION_URL=${BASE_URL}wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip

mkdir -p ucf-101/videos

cd ucf-101

echo "Downloading the data..."
curl -O -k ${DATASET_URL}
curl -O -k ${ANNOTATION_URL}

# Unzip the annotations
echo -ne "Moving annotations under the annots directory..."
unzip -q UCF101TrainTestSplits-RecognitionTask.zip
mv ucfTrainTestlist annots
rm UCF101TrainTestSplits-RecognitionTask.zip
echo "done"

# Unrar the dataset containing the video clips
echo -ne "Unzipping the videos and structuring them like image net dataset..."
mv UCF101.rar videos/UCF101.rar
cd videos
unrar -idq e UCF101.rar
rm UCF101.rar
cd ..

for fname in $(ls -1 videos/*.avi | xargs -n 1 basename); do
    readarray -d _ -t file_comp <<<"$fname"
    LABEL=${file_comp[1]}
    mkdir -p videos/${LABEL}
    mv videos/${fname} videos/${LABEL}/${fname}
done

echo "done"

cd ..
