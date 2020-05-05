BASE_URL=https://www.crcv.ucf.edu/data/UCF101/
DATASET_URL=${BASE_URL}UCF101.rar
ANNOTATION_URL=${BASE_URL}UCF101TrainTestSplits-RecognitionTask.zip

mkdir -p ucf-101/videos

cd ucf-101

echo "Downloading the data..."
curl -O ${DATASET_URL}
curl -O ${ANNOTATION_URL}

# Unzip the annotations
echo -ne "Moving annotations under the annots directory..."
unzip UCF101TrainTestSplits-RecognitionTask.zip
mv ucfTrainTestlist annots
rm UCF101TrainTestSplits-RecognitionTask.zip
echo "done"

# Unrar the dataset containing the video clips
echo -ne "Unzipping the videos and structuring them like image net dataset..."
mv UCF101.rar videos/UCF101.rar
cd videos
unrar e UCF101.rar
cd ..

ls -1 videos/*.avi | xargs -n 1 basename | while read x; do 
    IFS='_' read -ra file_comp <<< "$x"
    LABEL=${file_comp[1]}
    mkdir -p videos/${LABEL}
    mv videos/${x} videos/${LABEL}/${x}
done

echo "done"

# rm UCF101.rar

cd ..