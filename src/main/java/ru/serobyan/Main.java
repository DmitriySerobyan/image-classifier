package ru.serobyan;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

public class Main {

    public static void main(String[] args) {
        try {
            var classifier = ImageClassifier.of("C:\\image\\training", "C:\\image\\testing");
            classifier.buildModel();

            var trainingData = classifier.getTrainingDataSetIterator();
            classifier.trainModel(trainingData);

            var testingData = classifier.getTestingDataSetIterator();
            classifier.evaluatingModel(testingData);

            var imageFolder = new File("D:\\PicturesU");
            var imagesForClassification = imageFolder.listFiles();
            for (var image : imagesForClassification) {
                var labelOpt = classifier.predict(image);
                labelOpt.ifPresent(label -> {
                    moveImage(image, label);
                });
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void moveImage(File image, String label) {
        try {
            Files.copy(
                Paths.get(image.getAbsolutePath()),
                Paths.get("D:\\classification_test\\" + label + "\\" + image.getName()),
                StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
